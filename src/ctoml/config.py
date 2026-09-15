

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS = {
    "seed": 7,
    "device": "cpu",
    "threads": 1,
    "output_dir": "outputs/run",
    "data": {"manifest": "", "tokenizer": "word", "horizontal_flip": False},
    "model": {
        "feature_dim": 16, "d_model": 32, "nhead": 4,
        "num_encoder_layers": 1, "num_decoder_layers": 1,
        "dim_feedforward": 128, "dropout": 0.0, "max_length": 2048,
    },
    "train": {
        "mode": "meta", "warmup_epochs": 2, "epochs": 2,
        "batch_size": 8, "steps_per_epoch": None, "warmup_lr": 0.003,
        "outer_lr": 0.0005, "grad_clip": 5.0, "test_domain_count": 1,
    },
    "meta": {
        "inner_steps": 10, "inner_lr": 0.001, "inner_optimizer": "adam",
        "first_order": False, "contrastive_weight": 0.005,
        "temperature": 0.2, "margin": 1.0, "max_pairs": 256,
        "use_taw": True, "use_global": True, "use_local": True,
    },
    "decode": {"max_length": 64},
}


def merge(base: dict, update: dict, prefix: str = "") -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key} must be a mapping")
            result[key] = merge(base[key], value, f"{prefix}{key}.")
        else:
            result[key] = value
    return result


def _read(path: Path, ancestors: set[Path]) -> dict:
    path = path.resolve()
    if path in ancestors:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    with path.open(encoding="utf-8") as stream:
        current = yaml.safe_load(stream) or {}
    if not isinstance(current, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    parent = current.pop("extends", None)
    base = _read(path.parent / parent, ancestors | {path}) if parent else DEFAULTS
    return merge(base, current)


def validate(config: dict) -> dict:
    import math

    def positive(value: Any, name: str, integer: bool = False, zero: bool = False):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(value) or (value < 0 if zero else value <= 0):
            raise ValueError(f"Invalid {name}: {value}")
        if integer and not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")

    positive(config["threads"], "threads", True)
    positive(config["seed"], "seed", True, True)
    if config["device"] not in ("cpu", "cuda") and not str(config["device"]).startswith("cuda:"):
        raise ValueError("Supported devices are cpu and cuda[:index]; MPS higher derivatives are unverified")
    if config["data"]["tokenizer"] not in ("word", "character"):
        raise ValueError("data.tokenizer must be word or character")
    if not isinstance(config["data"]["horizontal_flip"], bool):
        raise ValueError("data.horizontal_flip must be a boolean")
    for key, value in config["model"].items():
        if key != "dropout":
            positive(value, f"model.{key}", True)
    if not 0 <= config["model"]["dropout"] < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if config["model"]["d_model"] % config["model"]["nhead"]:
        raise ValueError("model.d_model must be divisible by model.nhead")
    train, meta = config["train"], config["meta"]
    if train["mode"] not in ("base", "meta", "no_meta"):
        raise ValueError("train.mode must be base, meta or no_meta")
    positive(train["warmup_epochs"], "train.warmup_epochs", True, True)
    positive(train["epochs"], "train.epochs", True)
    for key in ("batch_size", "test_domain_count"):
        positive(train[key], f"train.{key}", True)
    if train["steps_per_epoch"] is not None:
        positive(train["steps_per_epoch"], "train.steps_per_epoch", True)
    for key in ("warmup_lr", "outer_lr", "grad_clip"):
        positive(train[key], f"train.{key}")
    positive(meta["inner_steps"], "meta.inner_steps", True)
    positive(meta["max_pairs"], "meta.max_pairs", True)
    for key in ("inner_lr", "temperature", "margin"):
        positive(meta[key], f"meta.{key}")
    positive(meta["contrastive_weight"], "meta.contrastive_weight", zero=True)
    if meta["inner_optimizer"] not in ("adam", "sgd"):
        raise ValueError("meta.inner_optimizer must be adam or sgd")
    for key in ("first_order", "use_taw", "use_global", "use_local"):
        if not isinstance(meta[key], bool):
            raise ValueError(f"meta.{key} must be a boolean")
    positive(config["decode"]["max_length"], "decode.max_length", True)
    if config["decode"]["max_length"] > config["model"]["max_length"]:
        raise ValueError("decode.max_length exceeds model positional capacity")
    return config


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    config = _read(Path(path), set())
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Expected dotted.key=value, got {override!r}")
        key, raw = override.split("=", 1)
        nested = yaml.safe_load(raw)
        for component in reversed(key.split(".")):
            nested = {component: nested}
        config = merge(config, nested)
    return validate(config)
