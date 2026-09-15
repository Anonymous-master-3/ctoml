import torch
import pytest
from torch import nn
from ctoml.models import CtoMLModel
from ctoml.training.meta import meta_objective


class AnalyticModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(.7, dtype=torch.double))

    def forward(self, features, feature_lengths, decoder_inputs):
        x = features[:, :decoder_inputs.shape[1], 0] * self.theta
        logits = torch.stack([x * 0, x * 0, -x, x * 0, x.square(), x], -1)
        return {"logits": logits, "token_features": torch.stack([x, x.square()], -1)}


def batch(domain, dtype=torch.double):
    return {"features": torch.tensor([[[.4 + domain * .5], [1.2 + domain * .2]]], dtype=dtype), "feature_lengths": torch.tensor([2]), "decoder_inputs": torch.tensor([[1, 4]]), "targets": torch.tensor([[4, 2]]), "domains": torch.tensor([domain])}


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_exact_meta_gradient_matches_finite_difference_without_parameter_pollution(optimizer):
    model = AnalyticModel()
    original = model.theta.detach().clone()
    settings = {"inner_steps": 3, "inner_lr": .03, "inner_optimizer": optimizer, "use_local": False, "temperature": .7}
    objective = meta_objective(model, batch(0), batch(1), None, settings)
    torch.testing.assert_close(model.theta, original)
    assert model.theta.grad is None
    derivative = torch.autograd.grad(objective["total"], model.theta)[0]
    epsilon = 1e-5
    values = []
    for shift in [epsilon, -epsilon]:
        with torch.no_grad():
            model.theta.copy_(original + shift)
        values.append(meta_objective(model, batch(0), batch(1), None, settings)["total"].detach())
    with torch.no_grad():
        model.theta.copy_(original)
    numeric = (values[0] - values[1]) / (2 * epsilon)
    torch.testing.assert_close(derivative, numeric, atol=1e-6, rtol=1e-5)
    assert model.theta.grad is None
    torch.testing.assert_close(model.theta, original)
    approximate = meta_objective(model, batch(0), batch(1), None, {**settings, "first_order": True})
    approximate_derivative = torch.autograd.grad(approximate["total"], model.theta)[0]
    assert abs(derivative - approximate_derivative) > 1e-7


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_transformer_ten_inner_steps_backward_and_state_unchanged(optimizer):
    torch.manual_seed(23)
    model = CtoMLModel(1, 6, d_model=4, nhead=1, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=8, dropout=0)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    result = meta_objective(model, batch(0, torch.float32), batch(1, torch.float32), torch.ones(6), {"inner_steps": 10, "inner_lr": .001, "inner_optimizer": optimizer})
    assert set(result) == {"total", "task", "global", "local"}
    result["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, state[name])
    assert model.feature_projection.weight.grad.abs().sum() > 0


def test_no_contrastive_ablation_preserves_task_meta_gradient():
    model = AnalyticModel()
    result = meta_objective(model, batch(0), batch(1), None, {"use_global": False, "use_local": False})
    assert result["total"] is result["task"]
    assert result["local"] == result["global"] == 0
    plain = meta_objective(model, batch(0), batch(1), None, {"inner_steps": 0, "use_global": False, "use_local": False})
    meta_gradient = torch.autograd.grad(result["total"], model.theta)[0]
    plain_gradient = torch.autograd.grad(plain["total"], model.theta)[0]
    assert abs(meta_gradient - plain_gradient) > 1e-7


def test_meta_accepts_different_train_test_sequence_lengths():
    model = AnalyticModel()
    test_batch = batch(1)
    test_batch["features"] = test_batch["features"][:, :1]
    test_batch["feature_lengths"] = torch.tensor([1])
    test_batch["decoder_inputs"] = test_batch["decoder_inputs"][:, :1]
    test_batch["targets"] = test_batch["targets"][:, :1]
    result = meta_objective(model, batch(0), test_batch, None, {"inner_steps": 2, "inner_optimizer": "sgd"})
    result["total"].backward()
    assert torch.isfinite(model.theta.grad)


@pytest.mark.parametrize("optimizer", ["sgd", "adam"])
def test_functional_updates_match_torch_optimizer_values(optimizer):
    from ctoml.losses import task_loss, global_loss
    model = AnalyticModel()
    reference = AnalyticModel()
    train, test = batch(0), batch(1)
    settings = {"inner_steps": 10, "inner_lr": .01, "inner_optimizer": optimizer, "use_local": False, "temperature": .7}
    actual = meta_objective(model, train, test, None, settings)
    optim = (torch.optim.Adam if optimizer == "adam" else torch.optim.SGD)(reference.parameters(), lr=.01)
    for _ in range(10):
        optim.zero_grad()
        output = reference(train["features"], train["feature_lengths"], train["decoder_inputs"])
        last_task = task_loss(output["logits"], train["targets"])
        last_task.backward()
        optim.step()
    train_output = reference(train["features"], train["feature_lengths"], train["decoder_inputs"])
    test_output = reference(test["features"], test["feature_lengths"], test["decoder_inputs"])
    contrast = global_loss(torch.cat([train_output["token_features"], test_output["token_features"]]), torch.cat([train["targets"], test["targets"]]), torch.tensor([0, 1]), [0], [1], 6, temperature=.7)
    torch.testing.assert_close(actual["task"], last_task, atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(actual["global"], contrast, atol=1e-10, rtol=1e-9)
