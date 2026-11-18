import copy

import pytest
import torch

from rqf_ssl.model import RQFSystem, default_config, parameter_count
from rqf_ssl.objective import WindowedPrediction


def small_config():
    config = default_config()
    config.update(width=8, layers=2, sequence_length=64, window_length=16)
    return config


def test_paper_parameter_counts():
    model = RQFSystem(default_config())
    assert parameter_count(model.encoder) == 33664
    assert parameter_count(model) == 110002


def test_causal_states_and_full_bptt():
    torch.manual_seed(7)
    model = RQFSystem(small_config())
    x = torch.randn(2, 64, 1, requires_grad=True)
    y = model.encoder(x)
    altered = x.detach().clone()
    altered[:, 32:] += 5
    torch.testing.assert_close(y[:, :32], model.encoder(altered)[:, :32])
    y[:, -1].abs().square().mean().backward()
    assert x.grad[:, :16].abs().sum() > 0
    assert torch.isfinite(x.grad).all()


def test_pooling_and_readout_backward():
    model = RQFSystem(small_config())
    x = torch.randn(2, 64, 16)
    tokens = model.summarize(x)
    assert tokens.shape == (2, 4, 16, 8)
    output = model(x)
    assert output.shape == (2, 2)
    output.square().mean().backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all()
        else:
            assert p.grad is None


def test_objective_detaches_targets_and_trains_source():
    objective = WindowedPrediction(8, horizon=1, negatives=3)
    tokens = torch.randn(2, 4, 2, 8, requires_grad=True)
    loss, metrics = objective(tokens)
    loss.backward()
    assert torch.isfinite(loss)
    assert tokens.grad[:, :-1].abs().sum() > 0
    assert tokens.grad[:, -1].count_nonzero() == 0
    assert metrics["nce"] > 0


def test_freezing_preserves_encoder_and_pooler():
    model = RQFSystem(small_config())
    model.freeze_representation()
    before = copy.deepcopy(model.encoder.state_dict())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    model(torch.randn(2, 64, 16)).sum().backward()
    optimizer.step()
    for name, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(p.grad is None for p in model.pooler.parameters())


def test_invalid_window_geometry():
    config = small_config()
    config["window_length"] = 17
    with pytest.raises(ValueError):
        RQFSystem(config)
