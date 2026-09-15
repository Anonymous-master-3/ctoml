
import torch
from torch.nn import functional as F


def task_loss(logits, targets, pad_id=0, weights=None):
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none", ignore_index=pad_id).reshape_as(targets)
    if weights is not None:
        losses = losses * weights.detach().to(logits)[targets]
    return losses.sum() / targets.ne(pad_id).sum().clamp_min(1)


def _lexical(targets, excluded_ids):
    mask = torch.ones_like(targets, dtype=torch.bool)
    for token in excluded_ids:
        mask = mask & targets.ne(token)
    return mask


def token_prototypes(vectors, targets, domains, vocab_size, excluded_ids=(0, 1, 2, 3)):
    domain_ids, inverse = torch.unique(domains, sorted=True, return_inverse=True)
    if vectors.shape[:-1] != targets.shape or targets.shape[0] != domains.numel():
        raise ValueError("Incompatible vector, target, or domain shapes")
    mask = _lexical(targets, excluded_ids)
    slots = (inverse[:, None].expand_as(targets) * vocab_size + targets)[mask]
    size = domain_ids.numel() * vocab_size
    sums = vectors.new_zeros(size, vectors.shape[-1]).index_add(0, slots, vectors[mask])
    counts = vectors.new_zeros(size).index_add(0, slots, vectors.new_ones(slots.numel()))
    means = sums / counts.clamp_min(1).unsqueeze(-1)
    return means.reshape(-1, vocab_size, vectors.shape[-1]), counts.reshape(-1, vocab_size), domain_ids


@torch.no_grad()
def difficulty_weights(prototypes, counts, excluded_ids=(0, 1, 2, 3), min_domains=2):
    if min_domains < 1:
        raise ValueError("min_domains must be positive")
    present = counts.gt(0)
    n = present.sum(0)
    mean = (prototypes * present[..., None]).sum(0) / n.clamp_min(1)[:, None]
    variance = ((prototypes - mean).square() * present[..., None]).sum(0) / n.clamp_min(1)[:, None]
    weights = variance.sigmoid().mean(-1)
    weights = torch.where(n >= min_domains, weights, torch.ones_like(weights))
    for token in excluded_ids:
        if token < weights.numel():
            weights[token] = 1
    return weights


def global_loss(vectors, targets, domains, train_domains, test_domains, vocab_size, temperature=0.2, excluded_ids=(0, 1, 2, 3)):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    prototypes, counts, ids = token_prototypes(vectors, targets, domains, vocab_size, excluded_ids)
    total = vectors.sum() * 0
    train_ids, test_ids = set(int(x) for x in train_domains), set(int(x) for x in test_domains)
    if train_ids & test_ids:
        raise ValueError("Meta-train and meta-test domains must be disjoint")
    for i, domain_i in enumerate(ids.tolist()):
        if domain_i not in train_ids:
            continue
        for j, domain_j in enumerate(ids.tolist()):
            if domain_j not in test_ids:
                continue
            shared = counts[i].gt(0) & counts[j].gt(0)
            p = F.log_softmax(prototypes[i, shared] / temperature, dim=-1)
            q = F.log_softmax(prototypes[j, shared] / temperature, dim=-1)
            total = total + ((p.exp() - q.exp()) * (p - q)).sum()
    return total / (2 * vocab_size)


def local_loss(vectors, targets, margin=1.0, max_pairs=256, excluded_ids=(0, 1, 2, 3)):
    if margin < 0 or max_pairs < 0:
        raise ValueError("margin and max_pairs must be nonnegative")
    mask = _lexical(targets, excluded_ids)
    selected, labels = vectors[mask], targets[mask]
    n = min(selected.shape[0] // 2, max_pairs)
    if n == 0:
        return vectors.sum() * 0
    order = torch.randperm(selected.shape[0], device=vectors.device)[:2 * n].reshape(n, 2)
    a, b = order[:, 0], order[:, 1]
    squared = (selected[a] - selected[b]).square().sum(-1)

    distance = torch.where(squared > 0, squared.clamp_min(torch.finfo(vectors.dtype).tiny).sqrt(), torch.zeros_like(squared))
    return torch.where(labels[a] == labels[b], distance, F.relu(margin - distance).square()).mean()


__all__ = ["task_loss", "token_prototypes", "difficulty_weights", "global_loss", "local_loss"]
