import math
import torch
from ctoml.losses import task_loss, token_prototypes, difficulty_weights, global_loss, local_loss


def test_weighted_ce_uses_valid_count_and_fixed_weights():
    logits = torch.zeros(1, 3, 6, requires_grad=True)
    targets = torch.tensor([[4, 2, 0]])
    weights = torch.tensor([1., 1., 1., 1., .5, 1.], requires_grad=True)
    loss = task_loss(logits, targets, weights=weights)
    torch.testing.assert_close(loss, torch.tensor(math.log(6) * .75))
    loss.backward()
    assert weights.grad is None
    assert logits.grad[0, 2].count_nonzero() == 0
    assert task_loss(logits, torch.zeros_like(targets)) == 0


def test_prototypes_observed_domains_and_variance_hand_calculation():
    vectors = torch.tensor([[[0., 2.], [2., 4.]], [[3., 5.], [10., 10.]], [[99., 99.], [99., 99.]]], requires_grad=True)
    targets = torch.tensor([[4, 4], [4, 5], [0, 2]])
    prototypes, counts, ids = token_prototypes(vectors, targets, torch.tensor([10, 20, 30]), 6)
    assert ids.tolist() == [10, 20, 30]
    torch.testing.assert_close(prototypes[0, 4], torch.tensor([1., 3.]))
    assert counts[:, 4].tolist() == [2, 1, 0]
    weights = difficulty_weights(prototypes, counts)
    torch.testing.assert_close(weights[4], torch.tensor(1.).sigmoid())
    assert weights[[0, 1, 2, 3, 5]].tolist() == [1.] * 5
    assert not weights.requires_grad
    prototypes.sum().backward()
    assert vectors.grad[2].count_nonzero() == 0


def test_global_symmetric_kl_divides_vocabulary_not_domain_pairs():
    vectors = torch.tensor([[[0., math.log(3)]], [[math.log(3), 0.]], [[math.log(3), 0.]]], requires_grad=True)
    targets = torch.full((3, 1), 4)

    loss = global_loss(vectors, targets, torch.tensor([1, 2, 3]), [1], [2, 3], 6, temperature=1)
    torch.testing.assert_close(loss, torch.tensor(math.log(3) / 6))
    loss.backward()
    assert torch.isfinite(vectors.grad).all()
    empty = global_loss(vectors, torch.tensor([[4], [5], [5]]), torch.tensor([1, 2, 3]), [1], [2, 3], 6)
    assert empty.requires_grad and empty == 0


def test_local_euclidean_positive_and_squared_hinge_negative():
    vectors = torch.tensor([[[0., 0.], [3., 4.]]], requires_grad=True)
    torch.testing.assert_close(local_loss(vectors, torch.tensor([[4, 4]])), torch.tensor(5.))
    torch.testing.assert_close(local_loss(vectors, torch.tensor([[4, 5]]), margin=6), torch.tensor(1.))
    assert local_loss(vectors, torch.tensor([[0, 4]])) == 0
    same = torch.zeros(1, 2, 2, requires_grad=True)
    loss = local_loss(same, torch.tensor([[4, 4]]))
    assert torch.isfinite(torch.autograd.grad(loss, same)[0]).all()
