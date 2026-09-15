

import hashlib
import json
import math
import platform
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ctoml.config import validate
from ctoml.data import (
    FeatureDataset, Vocabulary, collate_samples, load_records,
    sample_episode, validate_splits,
)
from ctoml.evaluation import corpus_metrics
from ctoml.losses import difficulty_weights, global_loss, local_loss, task_loss, token_prototypes
from ctoml.models import CtoMLModel
from .checkpoint import load_checkpoint, restore_rng, rng_state, save_checkpoint
from .meta import meta_objective


def write_json(path, value):

    def clean(item):
        if isinstance(item, dict):
            return {k: clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def forward(model, batch):
    return model(batch["features"], batch["feature_lengths"], batch["decoder_inputs"])


def build_datasets(config, vocabulary=None):
    path = Path(config["data"]["manifest"])
    if not path.is_file():
        raise FileNotFoundError(f"Feature manifest missing: {path}")
    records = load_records(path)
    validate_splits(records)
    train_records = [row for row in records if row["split"] == "train"]
    if not train_records:
        raise ValueError("Manifest has no training samples")
    if config["data"]["horizontal_flip"] and any("feature_path_flipped" not in row for row in train_records):
        raise ValueError("GRID horizontal_flip requires feature_path_flipped for every training sample; extract flipped caches first")
    vocabulary = vocabulary or Vocabulary.from_texts([r["transcript"] for r in train_records], mode=config["data"]["tokenizer"])
    if len(vocabulary) <= 4:
        raise ValueError("Training transcripts contain no lexical tokens")
    datasets = {
        split: FeatureDataset([r for r in records if r["split"] == split], vocabulary)
        for split in ("train", "dev", "test")
    }
    if not len(datasets["dev"]):
        raise ValueError("A separate dev split is required for checkpoint selection; do not substitute test")

    for split, dataset in datasets.items():
        for index in range(len(dataset)):
            item = dataset[index]
            if item["features"].shape[-1] != config["model"]["feature_dim"]:
                raise ValueError(f"{split} sample {index}: feature dimension does not match model.feature_dim")
            if item["features"].shape[0] > config["model"]["max_length"]:
                raise ValueError(f"{split} sample {index}: frame count exceeds model.max_length")
            if len(item["tokens"]) + 1 > config["model"]["max_length"]:
                raise ValueError(f"{split} sample {index}: target length exceeds model.max_length")
    datasets["train"].augment = config["data"]["horizontal_flip"]
    return datasets, vocabulary


def data_signature(path):

    digest = hashlib.sha256(Path(path).read_bytes())
    for record in load_records(path):
        for key in ("feature_path", "feature_path_flipped"):
            if key in record:
                feature = Path(record[key])
                stat = feature.stat()
                digest.update(f"{feature.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


@torch.no_grad()
def evaluate_model(model, dataset, batch_size, device, max_length):
    was_training = model.training
    model.eval()
    references, hypotheses, predictions = [], [], []
    loss_sum, valid_count = 0.0, 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_samples)
    for raw in loader:
        batch = to_device(raw, device)
        out = forward(model, batch)
        count = int(batch["targets"].ne(0).sum())
        loss_sum += float(task_loss(out["logits"], batch["targets"])) * count
        valid_count += count
        generated = model.generate(batch["features"], batch["feature_lengths"], max_length=max_length)
        for i, ids in enumerate(generated.cpu().tolist()):
            hypothesis = dataset.vocab.decode(ids)
            reference = raw["transcripts"][i]
            references.append(reference)
            hypotheses.append(hypothesis)
            predictions.append({"sample_id": raw["sample_ids"][i], "reference": reference, "prediction": hypothesis})
    metrics = corpus_metrics(references, hypotheses)
    metrics["loss"] = loss_sum / valid_count if valid_count else 0.0
    metrics["samples"] = len(references)
    model.train(was_training)
    return metrics, predictions


@torch.no_grad()
def estimate_weights(model, dataset, batch_size, device, vocab_size):
    model.eval()
    was_augment = dataset.augment
    dataset.augment = False
    sums, totals = {}, {}
    for raw in DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_samples):
        batch = to_device(raw, device)
        out = forward(model, batch)
        means, counts, domains = token_prototypes(out["token_features"], batch["targets"], batch["domains"], vocab_size)
        for i, domain in enumerate(domains.tolist()):
            if domain not in sums:
                sums[domain] = torch.zeros_like(means[i])
                totals[domain] = torch.zeros_like(counts[i])
            sums[domain] += means[i] * counts[i, :, None]
            totals[domain] += counts[i]
    keys = sorted(sums)
    counts = torch.stack([totals[k] for k in keys])
    means = torch.stack([sums[k] for k in keys]) / counts.clamp_min(1)[..., None]
    weights = difficulty_weights(means, counts)
    dataset.augment = was_augment
    model.train()
    return weights


def no_meta_objective(model, train_batch, test_batch, weights, settings):
    tr, te = forward(model, train_batch), forward(model, test_batch)
    task = task_loss(tr["logits"], train_batch["targets"], weights=weights if settings["use_taw"] else None)
    vectors = torch.cat([x["token_features"].reshape(-1, x["token_features"].shape[-1]) for x in (tr, te)])[:, None, :]
    labels = torch.cat([x["targets"].reshape(-1) for x in (train_batch, test_batch)])[:, None]
    domains = torch.cat([x["domains"][:, None].expand_as(x["targets"]).reshape(-1) for x in (train_batch, test_batch)])
    gl = global_loss(vectors, labels, domains, train_batch["domains"].unique(), test_batch["domains"].unique(), tr["logits"].shape[-1], temperature=settings["temperature"]) if settings["use_global"] else task * 0
    lc = local_loss(vectors, labels, margin=settings["margin"], max_pairs=settings["max_pairs"]) if settings["use_local"] else task * 0
    return {"total": task + settings["contrastive_weight"] * (gl + lc), "task": task, "global": gl, "local": lc}


def _resume_compatible(old, new):
    old, new = deepcopy(old), deepcopy(new)

    for config in (old, new):
        config["train"].pop("epochs", None)
    if old != new:
        raise ValueError("Resume config changed beyond train.epochs; resume in the original run directory")


def train(config, resume=None, stop_after_epochs=None):

    config = validate(deepcopy(config))
    seed_everything(config["seed"], config["threads"])
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --set device=cpu for small checks")
    output_dir = Path(config["output_dir"])
    if (output_dir / "last.pt").exists() and resume is None:
        raise FileExistsError(f"{output_dir}/last.pt exists. Use --resume or a new output directory.")
    state = load_checkpoint(resume) if resume else None
    if state:
        if Path(resume).resolve().parent != output_dir.resolve():
            raise ValueError("Resume checkpoint must belong to output_dir so the best checkpoint is preserved")
        _resume_compatible(state["config"], config)
    vocabulary = Vocabulary.from_dict(state["vocabulary"]) if state else None
    datasets, vocabulary = build_datasets(config, vocabulary)
    signature = data_signature(config["data"]["manifest"])
    if state and signature != state["data_signature"]:
        raise ValueError("Data manifest or feature cache changed since checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    write_json(output_dir / "vocabulary.json", vocabulary.to_dict())
    audit = {}
    for split, dataset in datasets.items():
        ids = [token for row in dataset.records for token in vocabulary.encode(row["transcript"])]
        audit[split] = {
            "samples": len(dataset), "performers": len({r["performer_id"] for r in dataset.records}),
            "tokens": len(ids), "unknown_tokens": ids.count(vocabulary.unk_id),
        }
    write_json(output_dir / "data_audit.json", audit)
    write_json(output_dir / "environment.json", {
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "platform": platform.platform(), "device": str(device), "data_signature": signature,
    })
    model = CtoMLModel(vocab_size=len(vocabulary), **config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["warmup_lr"])
    sampler = random.Random(config["seed"])
    weights = torch.ones(len(vocabulary), device=device)
    weights_ready = False
    start_epoch, step, history, best_metric = 0, 0, [], float("inf")
    training = config["train"]
    warmup_epochs = 0 if training["mode"] == "base" else training["warmup_epochs"]
    total_epochs = warmup_epochs + training["epochs"]
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])

        weights = state["weights"].to(device)
        weights_ready = state["weights_ready"]
        start_epoch, step = state["epoch_completed"], state["step"]
        history, best_metric = state["history"], state["best_metric"]
        restore_rng(state["rng"], sampler)
    else:
        initial, _ = evaluate_model(model, datasets["dev"], training["batch_size"], device, config["decode"]["max_length"])
        write_json(output_dir / "initial_dev.json", initial)
        history.append({"epoch": 0, "stage": "initial", "dev": initial})
    if start_epoch > total_epochs:
        raise ValueError("Checkpoint already exceeds requested training duration")
    steps_per_epoch = training["steps_per_epoch"] or math.ceil(len(datasets["train"]) / training["batch_size"])
    for epoch in range(start_epoch, total_epochs):
        warmup = epoch < warmup_epochs
        if epoch == warmup_epochs and training["mode"] != "base":
            best_metric = float("inf")
        if not warmup and training["mode"] != "base" and config["meta"]["use_taw"] and not weights_ready:
            weights = estimate_weights(model, datasets["train"], training["batch_size"], device, len(vocabulary))
            weights_ready = True
        lr = training["warmup_lr"] if warmup or training["mode"] == "base" else training["outer_lr"]
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        totals = {key: 0.0 for key in ("total", "task", "global", "local")}
        order = list(range(len(datasets["train"])))
        sampler.shuffle(order)
        partition_seed = sampler.randrange(2**32)
        for batch_index in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            if warmup or training["mode"] == "base":
                begin = (batch_index * training["batch_size"]) % len(order)
                indices = order[begin:begin + training["batch_size"]]
                batch = to_device(collate_samples([datasets["train"][i] for i in indices]), device)
                task = task_loss(forward(model, batch)["logits"], batch["targets"])
                losses = {"total": task, "task": task, "global": task * 0, "local": task * 0}
            else:
                tr, te = sample_episode(datasets["train"], training["batch_size"], sampler, test_domain_count=training["test_domain_count"], partition_seed=partition_seed)
                tr, te = to_device(tr, device), to_device(te, device)
                objective = meta_objective if training["mode"] == "meta" else no_meta_objective
                losses = objective(model, tr, te, weights, config["meta"])
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch+1}, step {step}")
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach())
            step += 1
        dev, _ = evaluate_model(model, datasets["dev"], training["batch_size"], device, config["decode"]["max_length"])
        entry = {"epoch": epoch + 1, "step": step, "stage": "warmup" if warmup else training["mode"],
                 "train": {key: value / steps_per_epoch for key, value in totals.items()}, "dev": dev,
                 "gradient_norm_last": float(grad_norm)}
        history.append(entry)
        metric = dev["wer"] if vocabulary.mode == "word" else dev["cer"]
        stage_start = epoch == (warmup_epochs if not warmup else 0)
        is_best = stage_start or metric < best_metric
        if is_best:
            best_metric = metric
        payload = {
            "format_version": 1, "config": config, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "vocabulary": vocabulary.to_dict(),
            "weights": weights.detach().cpu(), "weights_ready": weights_ready,
            "epoch_completed": epoch + 1, "step": step, "history": history,
            "best_metric": best_metric, "rng": rng_state(sampler), "data_signature": signature,
            "train_performers": sorted({r["performer_id"] for r in datasets["train"].records}),
        }
        save_checkpoint(output_dir / "last.pt", payload)
        if is_best:
            save_checkpoint(output_dir / ("warmup_best.pt" if warmup else "best.pt"), payload)
        write_json(output_dir / "history.json", history)
        print(f"epoch {epoch+1}/{total_epochs} {entry['stage']} loss={entry['train']['total']:.4f} dev_WER={dev['wer']:.4f} dev_CER={dev['cer']:.4f}", flush=True)
        if stop_after_epochs is not None and epoch + 1 >= stop_after_epochs:
            break
    return {"output_dir": str(output_dir), "history": history, "best_dev_metric": best_metric}


def evaluate_checkpoint(checkpoint_path, manifest=None, split="test", device="cpu", output=None):
    state = load_checkpoint(checkpoint_path)
    config = deepcopy(state["config"])
    config["device"] = device
    if manifest:
        config["data"]["manifest"] = str(manifest)
    seed_everything(config["seed"], config["threads"])
    vocabulary = Vocabulary.from_dict(state["vocabulary"])
    records = load_records(config["data"]["manifest"])
    validate_splits(records)
    rows = [r for r in records if r["split"] == split]
    if not rows:
        raise ValueError(f"No samples in split {split!r}")
    if split != "train":
        overlap = set(state["train_performers"]) & {r["performer_id"] for r in rows}
        if overlap:
            raise ValueError(f"Evaluation performers were present in training: {sorted(overlap)}")
    dataset = FeatureDataset(rows, vocabulary)
    model = CtoMLModel(vocab_size=len(vocabulary), **config["model"]).to(device)
    model.load_state_dict(state["model"])
    metrics, predictions = evaluate_model(model, dataset, config["train"]["batch_size"], device, config["decode"]["max_length"])
    result = {"checkpoint": str(checkpoint_path), "epoch": state["epoch_completed"], "split": split,
              "metrics": metrics, "predictions": predictions}
    if output:
        write_json(output, result)
    return result
