from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

import rqf_ssl.sc10_train as sc10_train_module
import rqf_ssl.train as train_module
from rqf_ssl.dense import (
    DensePrediction,
    _BoundedSampledDot,
    calibration_loader,
    cross_example_negatives,
    dense_recipe,
    masked_infonce,
    preserved_rng,
    select_anchors,
    temporal_negatives,
    window_log_power,
)
from rqf_ssl.model import RQFSystem, SpatialReadout
from rqf_ssl.sc10_model import SC10Representation

from test_sc10_train import _write_config as _write_sc10_config
from test_sc10_train import _write_manifest as _write_sc10_manifest
from test_train import _tiny_config, _write_dataset


def _tiny_recipe(name: str) -> dict[str, int]:
    recipe = dense_recipe(name)
    recipe.update(offset=8, window=4, anchor_count=16, min_separation=4)
    recipe["calibration_examples"] = 2
    return recipe


def test_anchors_are_seeded_sorted_distinct_and_leave_room_for_the_target() -> None:
    """Catches anchors whose future window runs off the sequence or ignores the seed."""
    anchors = select_anchors(100, offset=20, window=10, count=30, seed=1701)
    expected = torch.randperm(71, generator=torch.Generator().manual_seed(1701))[:30]
    assert torch.equal(anchors, expected.sort().values)
    assert anchors.dtype == torch.int64
    assert int(anchors.max()) + 20 + 10 <= 100
    with pytest.raises(ValueError, match="exceeds"):
        select_anchors(100, offset=20, window=10, count=72, seed=0)


def test_window_log_power_matches_a_direct_mean() -> None:
    """Catches an off-by-one in the prefix-sum target window."""
    power = torch.rand(3, 50, 4, dtype=torch.float64)
    anchors = torch.tensor([0, 7, 30])
    target = window_log_power(power, anchors, shift=5, window=6, epsilon=1e-6)
    for position, anchor in enumerate(anchors.tolist()):
        direct = power[:, anchor + 5 : anchor + 11].mean(1)
        torch.testing.assert_close(target[:, position], torch.log(1e-6 + direct))


def test_temporal_negatives_mask_only_nearby_rows_of_the_same_sequence() -> None:
    """Catches overlapping same-sequence futures entering the softmax as negatives."""
    anchors = torch.tensor([0, 3, 10, 20])
    indices, admitted = temporal_negatives(
        anchors, sequences=3, negatives=64, min_separation=5, device=torch.device("cpu")
    )
    rows = torch.arange(12)[:, None]
    assert not bool((indices == rows).any())
    same = indices // 4 == rows // 4
    close = (anchors.repeat(3)[indices] - anchors.repeat(3)[rows]).abs() < 5
    assert torch.equal(admitted, ~(same & close))


def test_cross_example_negatives_keep_the_channel_and_change_the_example() -> None:
    """Catches multichannel negatives drawn from another channel or the same example."""
    anchors = torch.arange(5)
    indices, admitted = cross_example_negatives(
        anchors, sequences=12, channels=4, negatives=50, device=torch.device("cpu")
    )
    positive = torch.arange(60)[:, None] // 5
    negative = indices // 5
    assert bool((negative % 4 == positive % 4).all())
    assert bool((negative // 4 != positive // 4).all())
    assert bool(admitted.all())


def test_bounded_sampled_dot_matches_a_materialized_gather() -> None:
    """Catches wrong gradients in the chunked recompute-in-backward kernel."""
    queries = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    keys = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    indices = torch.randint(7, (7, 5))
    assert torch.autograd.gradcheck(
        lambda q, k: _BoundedSampledDot.apply(q, k, indices), (queries, keys)
    )


def test_masked_infonce_equals_cross_entropy_without_masked_columns() -> None:
    """Catches masked negatives leaking into the partition function."""
    queries = F.normalize(torch.randn(4, 3), dim=-1)
    keys = F.normalize(torch.randn(4, 3), dim=-1)
    indices = torch.tensor([[1, 2], [0, 3], [3, 1], [2, 0]])
    admitted = torch.tensor([[True, False], [True, True], [False, True], [True, True]])
    loss, _ = masked_infonce(queries, keys, indices, admitted, 0.1)
    expected = []
    for row in range(4):
        columns = [keys[row]] + [
            keys[int(j)] for j, keep in zip(indices[row], admitted[row]) if keep
        ]
        logits = torch.stack(columns) @ queries[row] / 0.1
        expected.append(-torch.log_softmax(logits, 0)[0])
    torch.testing.assert_close(loss, torch.stack(expected).mean())


def test_calibration_uses_the_first_declared_sequences_and_gates_the_loss() -> None:
    """Catches standardization fitted on the wrong rows or skipped altogether."""
    objective = DensePrediction(
        2,
        sequence_length=40,
        offset=8,
        window=4,
        anchor_count=8,
        min_separation=4,
        calibration_examples=3,
        negatives=4,
    )
    features = torch.randn(2, 40, 4)
    power = torch.rand(2, 40, 2)
    with pytest.raises(RuntimeError, match="calibrated"):
        objective(features, power)
    batches = [(features, power), (features + 1, power * 3)]
    objective.calibrate(batches)
    _, first = objective.pairs(*batches[0])
    _, second = objective.pairs(*batches[1])
    rows = torch.cat([first, second[:1]]).reshape(-1, 2).double()
    torch.testing.assert_close(objective.target_mean, rows.mean(0).float())
    std = rows.std(0, unbiased=False).float()
    torch.testing.assert_close(objective.target_std, std)
    features.requires_grad_(True)
    loss, _ = objective(features, power)
    loss.backward()
    assert features.grad is not None and bool(features.grad.abs().sum() > 0)


def test_calibration_subset_and_rng_are_fixed() -> None:
    """Catches a calibration subset that moves or consumes training draws."""
    def drawn() -> list[int]:
        loader = calibration_loader(list(range(20)), examples=6, seed=3, batch_size=4)
        return [int(x) for batch in loader for x in batch]

    first = drawn()
    assert first == drawn() and len(set(first)) == 6
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    with preserved_rng():
        torch.rand(100)
    assert torch.equal(torch.rand(3), expected)


def test_sc10_dense_features_expose_nonnegative_final_layer_power() -> None:
    """Catches prescan power taken from the wrong layer or with the wrong layout."""
    config = {
        "width": 3,
        "layers": 2,
        "sequence_length": 40,
        "window_length": 4,
        "horizon": 1,
        "readout_hidden": 2,
        "fbt_dtype": "fp32",
    }
    model = SC10Representation(config)
    signal = torch.randn(2, 40, 1)
    features, power = model.dense_features(signal)
    assert features.shape == (2, 40, 6) and power.shape == (2, 40, 3)
    assert bool((power >= 0).all())
    states = model.states(signal)
    torch.testing.assert_close(features, torch.cat([states.real, states.imag], -1))


def test_sc10_dense_pretraining_saves_calibrated_objective_and_transfers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches dense SC10 pretraining skipping calibration or breaking transfer."""
    monkeypatch.setattr(sc10_train_module, "dense_recipe", _tiny_recipe)
    manifest, config = _write_sc10_manifest(tmp_path), _write_sc10_config(tmp_path)
    paths = sc10_train_module.pretrain(
        manifest,
        tmp_path / "ssl",
        config_path=config,
        epochs=1,
        batch_size=2,
        negatives=4,
        objective="dense",
        device="cpu",
    )
    payload = torch.load(paths["pretrained"], weights_only=True)
    assert payload["training_options"]["objective"] == "dense"
    assert bool(payload["objective_state"]["calibrated"])
    model, kind = sc10_train_module.load_for_finetune(
        paths["pretrained"], mode="frozen", seed=1
    )
    assert kind == "pretrained" and not any(
        p.requires_grad for p in model.encoder.parameters()
    )


def test_chunked_readout_matches_the_unchunked_moments() -> None:
    """Catches chunked accumulation diverging from whole-sequence pooling."""
    torch.manual_seed(0)
    whole = SpatialReadout(6, 2)
    chunked = SpatialReadout(6, 2, chunk_size=5)
    chunked.load_state_dict(whole.state_dict())
    tokens = torch.randn(3, 17, 16, 6, dtype=torch.float64)
    torch.testing.assert_close(
        chunked.double()(tokens), whole.double()(tokens), rtol=1e-6, atol=1e-6
    )


def test_tuab_dense_support_has_no_pooler_and_flattens_channels_example_major() -> None:
    """Catches channel rows in an order the negative sampler does not expect."""
    model = RQFSystem({**_tiny_config(), "temporal_support": "dense"})
    assert model.pooler is None and model.representation_modules() == [model.encoder]
    signal = torch.randn(2, 64, 16)
    features, power = model.dense_features(signal)
    assert features.shape == (32, 64, 16) and power.shape == (32, 64, 8)
    single, _ = model.dense_features(signal[1:, :, 3:4].expand(1, 64, 16).contiguous())
    torch.testing.assert_close(features[1 * 16 + 3], single[0], rtol=1e-5, atol=1e-5)
    assert model(signal).shape == (2, 2)


def test_tuab_dense_pretraining_transfers_the_encoder_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches dense checkpoints loading a windowed model or freezing too little."""
    monkeypatch.setattr(train_module, "dense_recipe", _tiny_recipe)
    manifest = _write_dataset(tmp_path)
    paths = train_module.pretrain(
        manifest,
        tmp_path / "ssl",
        config_path=tmp_path / "tiny.json",
        epochs=1,
        batch_size=2,
        negatives=4,
        objective="dense",
    )
    payload = torch.load(paths["pretrained"], weights_only=True)
    assert payload["model_config"]["temporal_support"] == "dense"
    assert bool(payload["objective_state"]["calibrated"])
    model, _ = train_module.load_for_finetune(
        paths["pretrained"], mode="frozen", seed=0
    )
    assert model.pooler is None
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.readout.parameters())
    result = train_module.finetune(
        manifest,
        paths["pretrained"],
        tmp_path / "frozen",
        mode="frozen",
        epochs=1,
        batch_size=2,
    )
    assert result["best"].is_file()
