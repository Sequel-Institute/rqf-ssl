import copy

import pytest
import torch

from rqf_ssl.objective import WindowedPrediction


def _small_config() -> dict[str, object]:
    from rqf_ssl.sc10_model import sc10_config

    return {
        **sc10_config(),
        "width": 4,
        "layers": 1,
        "sequence_length": 40,
        "window_length": 4,
        "horizon": 1,
        "readout_hidden": 5,
        "fbt_dtype": "fp32",
    }


def test_window_mean_has_no_trainable_pooler() -> None:
    """Catches replacing fixed real/imag window means with a learned pooler."""
    from rqf_ssl.sc10_model import SC10Representation

    model = SC10Representation(_small_config())
    x = torch.randn(2, 40, 1)
    y = model.encoder(x)
    want = torch.cat([y.real, y.imag], -1).reshape(2, 10, 4, 8).mean(2)

    torch.testing.assert_close(model.summarize(x), want)
    assert not [name for name, _ in model.named_parameters() if "pool" in name]


def test_classifier_applies_relu_before_time_mean() -> None:
    """Catches averaging states before the nonlinear downstream head."""
    from rqf_ssl.sc10_model import SC10Classifier

    model = SC10Classifier(_small_config())
    states = torch.zeros(1, 40, 4, dtype=torch.complex64)
    states[0, 0, 0] = 2 + 0j
    states[0, 1, 0] = -2 + 0j
    with torch.no_grad():
        first, _, last = model.readout
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 0] = 1
        last.weight.zero_()
        last.bias.zero_()
        last.weight[0, 0] = 1

    assert model.classify_states(states)[0, 0].item() == pytest.approx(0.05)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_bf16_readout_pools_in_float32_and_preserves_gradients(
    device: str, autocast: bool
) -> None:
    """Catches FP32/BF16 handoff failures and rounding before temporal pooling."""
    from rqf_ssl.sc10_model import SC10Classifier

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    model = SC10Classifier(_small_config()).to(device)
    with torch.no_grad():
        first, _, last = model.readout
        for parameter in model.readout.parameters():
            parameter.zero_()
        first.weight[0, 0] = 1
        last.weight[0, 0] = 1
    features = torch.zeros(8, 1, 4, dtype=torch.bfloat16, device=device)
    features[0, 0] = torch.tensor([1.0, 1.0, 1.0, 1.0078125], device=device)
    features.requires_grad_()

    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        logits = model._classify_channel_first(features)

    # These logits are exact in BF16, but their mean is exact only in FP32.
    # Four steps avoid division-rounding differences between CPU and CUDA.
    # Casting only after the mean would give 1.0 instead.
    want = torch.zeros(1, 10, device=device)
    want[0, 0] = 4.0078125 / 4
    torch.testing.assert_close(logits, want, rtol=0, atol=0)
    logits[0, 0].backward()
    want_gradient = torch.zeros_like(features)
    want_gradient[0, 0] = 1 / 4
    torch.testing.assert_close(features.grad, want_gradient, rtol=0, atol=0)
    for parameter in model.readout.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.count_nonzero() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_default_cuda_classifier_forward_without_autocast() -> None:
    """Catches a native BF16 encoder output incompatible with the fresh readout."""
    from rqf_ssl.sc10_model import SC10Classifier, sc10_config

    model = SC10Classifier(sc10_config()).cuda()
    with torch.no_grad():
        logits = model(torch.randn(1, 16000, 1, device="cuda"))
    assert logits.shape == (1, 10)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_ssl_has_only_encoder_and_objective_tokens_detach_targets() -> None:
    """Catches allocating a downstream readout in SSL or backpropagating targets."""
    from rqf_ssl.sc10_model import SC10Representation

    representation = SC10Representation(_small_config())
    assert set(representation._modules) == {"encoder"}
    tokens = representation.summarize(torch.randn(2, 40, 1)).unsqueeze(2)
    tokens.retain_grad()
    loss, _ = WindowedPrediction(8, horizon=1, negatives=2)(tokens)
    loss.backward()

    assert tokens.grad[:, :-1].abs().sum() > 0
    assert tokens.grad[:, -1].count_nonzero() == 0


def test_frozen_encoder_is_byte_stable_for_in_bound_parameters() -> None:
    """Catches a frozen downstream update changing valid transferred encoder weights."""
    from rqf_ssl.sc10_model import SC10Classifier

    model = SC10Classifier(_small_config())
    model.freeze_encoder()
    before = copy.deepcopy(model.encoder.state_dict())
    optimizer = torch.optim.AdamW(model.readout.parameters(), lr=1e-2)
    logits = model(torch.randn(2, 40, 1))
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    optimizer.step()
    model.clamp_parameters()

    for name, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_frozen_encoder_still_projects_out_of_bound_filter_parameters() -> None:
    """Catches skipping the source post-step clamp just because encoder is frozen."""
    from rqf_ssl.sc10_model import SC10Classifier

    model = SC10Classifier(_small_config())
    model.freeze_encoder()
    filter_ = model.encoder.blocks[0].rqf_fwd
    with torch.no_grad():
        filter_.log_gamma.fill_(100)
    model.clamp_parameters()

    assert filter_.log_gamma.max() <= torch.log(filter_._max_gamma).max()


def test_default_sc10_parameter_counts() -> None:
    """Catches an encoder or pointwise readout geometry change."""
    from rqf_ssl.model import parameter_count
    from rqf_ssl.sc10_model import SC10Classifier, SC10Representation, sc10_config

    assert parameter_count(SC10Representation(sc10_config()).encoder) == 33664
    assert parameter_count(SC10Classifier(sc10_config()).readout) == 35594


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"classes": 2}, "ten classes"),
        ({"window_length": 7}, "divisible"),
        ({"horizon": 40}, "horizon"),
        ({"fbt_dtype": "float64"}, "fbt_dtype"),
        ({"fbt_dtype": []}, "fbt_dtype"),
        ({"fbt_dtype": {}}, "fbt_dtype"),
    ],
)
def test_config_rejects_invalid_sc10_geometry(
    override: dict[str, object], message: str
) -> None:
    """Catches accepting a configuration outside the fixed SC10 contract."""
    from rqf_ssl.sc10_model import SC10Representation, sc10_config

    with pytest.raises(ValueError, match=message):
        SC10Representation({**sc10_config(), **override})
