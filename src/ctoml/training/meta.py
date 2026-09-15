





import torch
from torch.func import functional_call

from ctoml.losses import global_loss, local_loss, task_loss


def _forward(model, parameters, batch):
    return functional_call(model, parameters, (batch["features"], batch["feature_lengths"], batch["decoder_inputs"]))


def meta_objective(model, train_batch, test_batch, weights, settings):
    steps = int(settings.get("inner_steps", 10))
    lr = float(settings.get("inner_lr", 1e-4))
    optimizer = settings.get("inner_optimizer", "adam")
    first_order = bool(settings.get("first_order", False))
    if steps < 0 or lr < 0 or optimizer not in {"adam", "sgd"}:
        raise ValueError("Invalid inner optimizer, step count, or learning rate")
    parameters = dict(model.named_parameters())
    effective_weights = weights if settings.get("use_taw", True) else None
    original = _forward(model, parameters, train_batch)
    task = task_loss(original["logits"], train_batch["targets"], weights=effective_weights)
    use_global, use_local = settings.get("use_global", True), settings.get("use_local", True)
    moments, variances = {}, {}
    for step in range(1, steps + 1):
        output = original if step == 1 else _forward(model, parameters, train_batch)
        inner = task_loss(output["logits"], train_batch["targets"], weights=effective_weights)
        task = inner
        active = {name: p for name, p in parameters.items() if p.requires_grad}
        gradients = torch.autograd.grad(inner, tuple(active.values()), create_graph=not first_order, retain_graph=True, allow_unused=True)
        updated = dict(parameters)
        for (name, parameter), gradient in zip(active.items(), gradients):
            if gradient is None:
                continue
            if first_order:
                gradient = gradient.detach()
            if optimizer == "sgd":
                update = gradient
            else:
                m = 0.9 * moments.get(name, torch.zeros_like(parameter)) + 0.1 * gradient
                v = 0.999 * variances.get(name, torch.zeros_like(parameter)) + 0.001 * gradient.square()
                moments[name], variances[name] = m, v

                denominator = (v / (1 - 0.999 ** step)).clamp_min(torch.finfo(v.dtype).tiny).sqrt() + 1e-8
                update = (m / (1 - 0.9 ** step)) / denominator
            updated[name] = parameter - lr * update
        parameters = updated
    zero = task * 0
    if not use_global and not use_local:
        return {"total": task, "task": task, "global": zero, "local": zero}
    train_output = _forward(model, parameters, train_batch)
    test_output = _forward(model, parameters, test_batch)

    vectors = torch.cat([train_output["token_features"].reshape(-1, train_output["token_features"].shape[-1]), test_output["token_features"].reshape(-1, test_output["token_features"].shape[-1])])[:, None, :]
    targets = torch.cat([train_batch["targets"].reshape(-1), test_batch["targets"].reshape(-1)])[:, None]
    domains = torch.cat([train_batch["domains"][:, None].expand_as(train_batch["targets"]).reshape(-1), test_batch["domains"][:, None].expand_as(test_batch["targets"]).reshape(-1)])
    global_term = global_loss(vectors, targets, domains, train_batch["domains"].unique(), test_batch["domains"].unique(), original["logits"].shape[-1], temperature=settings.get("temperature", 0.2)) if use_global else zero
    local_term = local_loss(vectors, targets, margin=settings.get("margin", 1.0), max_pairs=settings.get("max_pairs", 256)) if use_local else zero
    total = task + settings.get("contrastive_weight", 1.0) * (global_term + local_term)
    return {"total": total, "task": task, "global": global_term, "local": local_term}
