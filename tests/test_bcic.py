from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import rqf_ssl.bcic_train as bcic_train
import rqf_ssl.objective as objective_module
from rqf_ssl.alignment import align_sessions, euclidean_alignment_matrix
from rqf_ssl.bcic_data import BCICDataset, read_trials, run_assignments
from rqf_ssl.model import RQFSystem, default_config
from rqf_ssl.objective import WindowedPrediction, same_channel_negatives
from rqf_ssl.train import cosine_factor


def _small_config() -> dict[str, object]:
    return dict(
        width=4,
        layers=1,
        sample_rate=200.0,
        sequence_length=800,
        window_length=100,
        horizon=2,
        classes=4,
        fbt_dtype="fp32",
        temporal_support="windowed",
        montage="bcic22",
    )


def _write_bcic(root: Path) -> Path:
    rows, generator = [], np.random.default_rng(3)
    for subject in range(1, 10):
        split = "train" if subject <= 5 else "val" if subject <= 7 else "test"
        for session in ("T", "E"):
            for run in range(6):
                rows.append([subject, session, run, 0, run % 4, split])
    signals = generator.normal(size=(len(rows), 22, 800)).astype(np.float32)
    for index, row in enumerate(rows):
        signals[index, row[4] * 5] += 2.0
    root.mkdir()
    np.save(root / "signals.npy", signals, allow_pickle=False)
    with (root / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject", "session", "run", "trial", "label", "split"])
        writer.writerows(rows)
    return root


def test_alignment_whitens_each_session_to_its_mean_power() -> None:
    """Catches alignment matrices that are not the power-scaled inverse square root."""
    generator = torch.Generator().manual_seed(0)
    mixing = torch.randn(5, 5, generator=generator, dtype=torch.float64)
    trials = [
        (mixing @ torch.randn(5, 200, generator=generator, dtype=torch.float64)).float()
        for _ in range(30)
    ]
    reference = sum(t.double() @ t.double().T / 200 for t in trials) / len(trials)
    matrix = euclidean_alignment_matrix(trials).double()
    aligned = sum((matrix @ t.double()) @ (matrix @ t.double()).T / 200 for t in trials)
    power = torch.trace(reference) / 5
    identity = torch.eye(5, dtype=torch.float64)
    torch.testing.assert_close(
        aligned / len(trials), power * identity, rtol=1e-4, atol=1e-4
    )
    signals = torch.stack(trials).numpy()
    sessions = ["a"] * 15 + ["b"] * 15
    result, counts = align_sessions(signals, sessions)
    assert counts == {"a": 15, "b": 15}
    first = euclidean_alignment_matrix(trials[:15])
    np.testing.assert_allclose(result[0], (first @ trials[0]).numpy(), rtol=1e-6)


def test_same_channel_negatives_are_distinct_on_both_sampling_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches negatives from the anchor's own example, another channel, or twice."""
    for limit in (1 << 26, 0):
        monkeypatch.setattr(objective_module, "_EXACT_POOL_ELEMENTS", limit)
        batch, times, channels = 5, 4, 3
        cpu = torch.device("cpu")
        indices = same_channel_negatives(batch, times, channels, 12, cpu)
        rows = torch.arange(batch * channels * times)[:, None]
        assert indices.shape == (60, 12)
        assert bool((indices // (channels * times) != rows // (channels * times)).all())
        assert bool(((indices // times) % channels == (rows // times) % channels).all())
        ordered = indices.sort(dim=1).values
        assert bool((ordered[:, 1:] != ordered[:, :-1]).all())
    with pytest.raises(ValueError, match="exceed"):
        same_channel_negatives(2, 3, 2, 4, torch.device("cpu"))


def test_same_channel_objective_backpropagates_to_sources() -> None:
    """Catches the bounded sampled product cutting gradients to the predictor path."""
    tokens = torch.randn(4, 8, 3, 6, requires_grad=True)
    objective = WindowedPrediction(
        6, horizon=2, negatives=5, negative_policy="same_channel"
    )
    loss, _ = objective(tokens)
    loss.backward()
    assert tokens.grad is not None and bool(tokens.grad[:, :6].abs().sum() > 0)


def test_cosine_schedule_warms_up_then_decays_towards_the_floor() -> None:
    """Catches an off-by-one in the warmup or a decay that ends at the wrong level."""
    factors = [
        cosine_factor(e, epochs=60, warmup_epochs=5, final_fraction=0.05)
        for e in range(60)
    ]
    assert factors[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert factors[5] == pytest.approx(0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi / 56)))
    assert all(a > b for a, b in zip(factors[4:], factors[5:]))
    assert factors[-1] > 0.05


def test_run_assignment_groups_48_trials_per_recording_run() -> None:
    """Catches runs counted from EOG-only recording blocks."""
    records = [0, 100, 200] + [1000 * (run + 1) for run in range(6)]
    trials = [1000 * (run + 1) + 10 * t for run in range(6) for t in range(48)]
    expected = [run for run in range(6) for _ in range(48)]
    assert run_assignments(records, trials) == expected


def test_ssl_trials_exclude_the_monitoring_run_and_alignment_is_per_session(
    tmp_path: Path,
) -> None:
    """Catches SSL reading held-out or non-training trials, or pooling sessions."""
    root = _write_bcic(tmp_path / "bcic")
    ssl = BCICDataset(root, "train", ssl=True)
    assert len(ssl) == 5 * 2 * 6 - 5
    assert not any(t.session == "E" and t.run == 5 for t in ssl.trials)
    aligned = BCICDataset(root, "val", alignment="euclidean")
    assert set(aligned.session_counts) == {"A06T", "A06E", "A07T", "A07E"}
    signal, label = aligned[0]
    assert signal.shape == (800, 22) and label == aligned.trials[0].label


def test_models_read_microvolts_scaled_by_the_paper_factor(tmp_path: Path) -> None:
    """Catches trials reaching the model in raw microvolts, 100x the paper's input."""
    root = _write_bcic(tmp_path / "bcic")
    raw = np.load(root / "signals.npy")
    trials = BCICDataset(root, "train")
    first = next(i for i, row in enumerate(read_trials(root)) if row.split == "train")
    signal, _ = trials[0]
    np.testing.assert_array_equal(signal, raw[first].T / np.float32(100.0))


def test_bcic_pretraining_transfer_and_evaluation_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches in-domain or cross-corpus sources failing to reach test evaluation."""
    monkeypatch.setattr(bcic_train, "bcic_config", _small_config)
    root = _write_bcic(tmp_path / "bcic")
    paths = bcic_train.pretrain(
        root, tmp_path / "ssl", epochs=2, batch_size=8, warmup_epochs=1, negatives=4
    )
    init = torch.load(paths["init"], weights_only=True)["model_state"]
    tuab = RQFSystem({**default_config(), "width": 4, "layers": 1, "fbt_dtype": "fp32"})
    tuab_path = tmp_path / "tuab.pt"
    torch.save({"kind": "initialization", "model_state": tuab.state_dict()}, tuab_path)
    paired = bcic_train.pretrain(
        root, tmp_path / "paired", epochs=2, batch_size=8, warmup_epochs=1,
        negatives=4, init_from=tuab_path,
    )  # fmt: skip
    paired_init = torch.load(paired["init"], weights_only=True)["model_state"]
    for name, value in tuab.encoder.state_dict().items():
        copied = paired_init[f"encoder.{name}"]
        torch.testing.assert_close(copied, value, rtol=0, atol=0)
    encoder_keys = [key for key in init if key.startswith("encoder.")]
    assert any(not torch.equal(init[k], paired_init[k]) for k in encoder_keys)
    for source, alignment in ((paths["pretrained"], "euclidean"), (tuab_path, "none")):
        out = tmp_path / f"down-{alignment}"
        result = bcic_train.finetune(
            root, source, out, alignment=alignment, epochs=1, batch_size=8, patience=1
        )
        model = torch.load(result["best"], weights_only=True)
        assert model["training_options"]["alignment"] == alignment
        assert model["model_config"]["window_length"] == 400
        metrics = bcic_train.evaluate(root, result["best"], batch_size=8)
        assert metrics["samples"] == 24 and 0.0 <= metrics["balanced_accuracy"] <= 1.0
