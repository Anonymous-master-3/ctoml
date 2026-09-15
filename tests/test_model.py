import torch
import pytest
from ctoml.models import CtoMLModel


def small_model():
    torch.manual_seed(7)
    return CtoMLModel(3, 8, d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=12, dropout=0, max_length=32).eval()


def test_causal_attention_and_source_padding():
    model = small_model()
    source = torch.randn(2, 5, 3)
    lengths = torch.tensor([3, 5])
    tokens = torch.tensor([[1, 4, 5, 0], [1, 6, 4, 2]])
    output = model(source, lengths, tokens)
    assert output["token_features"].shape == (2, 4, 8)
    changed = tokens.clone()
    changed[:, 2:] = 7
    torch.testing.assert_close(output["logits"][:, :2], model(source, lengths, changed)["logits"][:, :2])
    perturbed = source.clone()
    perturbed[0, 3:] = 1e4
    torch.testing.assert_close(output["logits"], model(perturbed, lengths, tokens)["logits"])

    padded_tokens = torch.cat([tokens, torch.zeros(2, 2, dtype=torch.long)], 1)
    torch.testing.assert_close(output["logits"], model(source, lengths, padded_tokens)["logits"][:, :4])


def test_token_features_are_last_cross_attention_output():
    model = small_model()
    captured = []
    hook = model.decoder[-1].cross_attention.register_forward_hook(lambda module, args, output: captured.append(output))
    output = model(torch.randn(1, 3, 3), torch.tensor([3]), torch.tensor([[1, 4]]))
    hook.remove()
    torch.testing.assert_close(output["token_features"], captured[0])


def test_generation_eos_and_training_mode_restore():
    model = small_model().train()
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.bias[2] = 100
    result = model.generate(torch.randn(2, 3, 3), torch.tensor([2, 3]), max_length=8)
    assert result.tolist() == [[2], [2]]
    assert model.training


def test_invalid_zero_length_rejected():
    with pytest.raises(ValueError, match="Feature lengths"):
        small_model()(torch.randn(1, 2, 3), torch.tensor([0]), torch.tensor([[1]]))


def test_generation_pads_rows_after_eos():
    import types
    model = small_model()
    def scripted_forward(self, features, lengths, inputs):
        logits = torch.full((2, inputs.shape[1], 8), -100.)
        logits[0, -1, 2] = 100.
        logits[1, -1, 4 if inputs.shape[1] == 1 else 2] = 100.
        return {"logits": logits}
    model.forward = types.MethodType(scripted_forward, model)
    result = model.generate(torch.randn(2, 3, 3), torch.tensor([3, 3]), max_length=8)
    assert result.tolist() == [[2, 0], [4, 2]]
