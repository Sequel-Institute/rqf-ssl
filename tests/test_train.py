from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

import rqf_ssl.train as train_module
from rqf_ssl.model import RQFSystem
from rqf_ssl.train import (
    _epoch_factor,
    _save_checkpoint,
    _set_epoch_learning_rates,
    classification_metrics,
    evaluate,
    finetune,
    load_for_finetune,
    pretrain,
)


def _tiny_config() -> dict[str, object]:
    return {
        "width": 8,
        "layers": 2,
        "sample_rate": 128.0,
        "sequence_length": 64,
        "window_length": 16,
        "horizon": 1,
        "classes": 2,
        "fbt_dtype": "fp32",
    }


def _write_dataset(root: Path) -> Path:
    rows: list[dict[str, object]] = []
    generator = np.random.default_rng(14)
    for split in ("train", "val", "test"):
        for label in (0, 1):
            signal = generator.normal(size=(64, 16)).astype(np.float32)
            signal[:, label] += 0.75
            filename = f"{split}_{label}.npy"
            np.save(root / filename, signal, allow_pickle=False)
            rows.append(
                {
                    "path": filename,
                    "patient_id": f"{split}-{label}",
                    "split": split,
                    "label": label,
                }
            )
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "patient_id", "split", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)
    config = root / "tiny.json"
    config.write_text(json.dumps(_tiny_config()), encoding="utf-8")
    return manifest


def _state_changed(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> bool:
    return any(not torch.equal(left[name], right[name]) for name in left)


def _write_source_checkpoint(path: Path) -> Path:
    torch.manual_seed(0)
    model = RQFSystem(_tiny_config())
    torch.save(
        {
            "kind": "initialization",
            "model_config": _tiny_config(),
            "seed": 0,
            "training_options": {},
            "epoch": 0,
            "model_state": model.state_dict(),
        },
        path,
    )
    return path


def test_training_pipeline_preserves_split_and_transfer_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _write_dataset(tmp_path)
    loaded_paths: list[str] = []
    original_load = np.load

    def recording_load(path: object, *args: object, **kwargs: object) -> np.ndarray:
        loaded_paths.append(os.fspath(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)
    ssl_dir = tmp_path / "ssl"
    pretrain(
        manifest,
        ssl_dir,
        config_path=tmp_path / "tiny.json",
        epochs=1,
        batch_size=2,
        negatives=2,
        device="cpu",
    )
    pretrain_progress = json.loads(capsys.readouterr().out)
    assert pretrain_progress["phase"] == "pretrain"
    assert pretrain_progress["epoch"] == 1
    assert np.isfinite(pretrain_progress["loss"])

    init = torch.load(ssl_dir / "init.pt", map_location="cpu", weights_only=True)
    pretrained = torch.load(
        ssl_dir / "pretrained.pt", map_location="cpu", weights_only=True
    )
    assert init["kind"] == "initialization"
    assert pretrained["kind"] == "pretrained"
    assert init["epoch"] == 0
    torch.manual_seed(0)
    expected = RQFSystem(_tiny_config()).state_dict()
    for name, value in init["model_state"].items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    init_representation = {
        name: value
        for name, value in init["model_state"].items()
        if name.startswith(("encoder.", "pooler."))
    }
    pretrained_representation = {
        name: value
        for name, value in pretrained["model_state"].items()
        if name.startswith(("encoder.", "pooler."))
    }
    assert _state_changed(init_representation, pretrained_representation)
    assert not any(Path(path).name.startswith("test_") for path in loaded_paths)

    from_init, init_kind = load_for_finetune(
        ssl_dir / "init.pt", mode="full-ft", seed=19, device="cpu"
    )
    from_pretrained, pretrained_kind = load_for_finetune(
        ssl_dir / "pretrained.pt", mode="full-ft", seed=19, device="cpu"
    )
    assert (init_kind, pretrained_kind) == ("initialization", "pretrained")
    for name, value in from_init.readout.state_dict().items():
        torch.testing.assert_close(
            value, from_pretrained.readout.state_dict()[name], rtol=0, atol=0
        )

    frozen_dir = tmp_path / "frozen"
    finetune(
        manifest,
        ssl_dir / "pretrained.pt",
        frozen_dir,
        mode="frozen",
        seed=19,
        epochs=1,
        batch_size=2,
        device="cpu",
    )
    frozen_progress = json.loads(capsys.readouterr().out)
    assert frozen_progress["phase"] == "finetune"
    assert frozen_progress["epoch"] == 1
    assert set(frozen_progress) >= {
        "train_loss",
        "validation_loss",
        "validation_auroc",
        "validation_balanced_accuracy",
    }
    frozen = torch.load(frozen_dir / "best.pt", map_location="cpu", weights_only=True)
    assert frozen["kind"] == "downstream"
    assert frozen["source_kind"] == "pretrained"
    for name, value in pretrained_representation.items():
        torch.testing.assert_close(frozen["model_state"][name], value, rtol=0, atol=0)

    full_dir = tmp_path / "full"
    finetune(
        manifest,
        ssl_dir / "pretrained.pt",
        full_dir,
        mode="full-ft",
        seed=19,
        epochs=1,
        batch_size=2,
        device="cpu",
    )
    json.loads(capsys.readouterr().out)
    full = torch.load(full_dir / "best.pt", map_location="cpu", weights_only=True)
    full_representation = {
        name: value
        for name, value in full["model_state"].items()
        if name.startswith(("encoder.", "pooler."))
    }
    assert _state_changed(pretrained_representation, full_representation)
    assert not any(Path(path).name.startswith("test_") for path in loaded_paths)

    result = evaluate(
        manifest,
        frozen_dir / "best.pt",
        batch_size=2,
        device="cpu",
    )
    assert set(result) == {"auroc", "balanced_accuracy", "samples"}
    assert result["samples"] == 2
    assert any(Path(path).name.startswith("test_") for path in loaded_paths)


def test_classification_metrics_match_hand_calculation_and_reject_one_class() -> None:
    logits = torch.tensor([[0.5, 0.1], [0.5, 0.4], [0.5, 0.35], [0.5, 0.8]])
    targets = torch.tensor([0, 0, 1, 1])

    assert classification_metrics(logits, targets) == {
        "auroc": 0.75,
        "balanced_accuracy": 0.75,
    }
    with pytest.raises(ValueError, match="both classes"):
        classification_metrics(logits[:2], targets[:2])


def test_auroc_uses_binary_logit_delta_and_ignores_samplewise_offsets() -> None:
    logits = torch.tensor([[0.0, 2.0], [100.0, 3.0]])
    targets = torch.tensor([0, 1])

    expected = {"auroc": 0.0, "balanced_accuracy": 0.0}
    assert classification_metrics(logits, targets) == expected
    offsets = torch.tensor([[17.0], [-31.0]])
    assert classification_metrics(logits + offsets, targets) == expected


def test_epoch_factors_match_five_iteration_linear_warmup_then_decay() -> None:
    assert [_epoch_factor(epoch) for epoch in range(8)] == pytest.approx(
        [0.2, 0.36, 0.52, 0.68, 0.84, 1.0, 0.99, 0.9801]
    )


def test_epoch_schedule_applies_exact_factor_to_both_parameter_groups() -> None:
    upstream = torch.nn.Parameter(torch.zeros(()))
    readout = torch.nn.Parameter(torch.zeros(()))
    base_rates = [2e-4, 2e-3]
    optimizer = torch.optim.AdamW(
        [
            {"params": [upstream], "lr": base_rates[0]},
            {"params": [readout], "lr": base_rates[1]},
        ]
    )

    observed: list[list[float]] = []
    for epoch in range(8):
        _set_epoch_learning_rates(optimizer, base_rates, epoch)
        observed.append(
            [
                group["lr"] / base_rate
                for group, base_rate in zip(
                    optimizer.param_groups, base_rates, strict=True
                )
            ]
        )

    expected = [0.2, 0.36, 0.52, 0.68, 0.84, 1.0, 0.99, 0.9801]
    assert [row[0] for row in observed] == pytest.approx(expected)
    assert [row[1] for row in observed] == pytest.approx(expected)


def test_atomic_checkpoint_failure_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "last.pt"
    _save_checkpoint(checkpoint, {"epoch": 1, "value": torch.tensor([4.0])})

    def fail_after_partial_write(payload: object, destination: object) -> None:
        del payload
        if hasattr(destination, "write"):
            destination.write(b"partial")
        else:
            Path(destination).write_bytes(b"partial")
        raise OSError("simulated serialization failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    with pytest.raises(OSError, match="simulated"):
        _save_checkpoint(checkpoint, {"epoch": 2})

    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert restored["epoch"] == 1
    torch.testing.assert_close(restored["value"], torch.tensor([4.0]))
    assert list(tmp_path.glob(".last.pt.tmp-*")) == []


def test_finetune_persists_completed_epoch_before_later_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_dataset(tmp_path)
    source = _write_source_checkpoint(tmp_path / "source.pt")
    real_evaluate = train_module._evaluate_loader
    evaluations = 0

    def fail_second_evaluation(*args: object, **kwargs: object):
        nonlocal evaluations
        evaluations += 1
        if evaluations == 2:
            raise RuntimeError("simulated interruption")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(train_module, "_evaluate_loader", fail_second_evaluation)
    output = tmp_path / "fit"
    with pytest.raises(RuntimeError, match="interruption"):
        finetune(
            manifest,
            source,
            output,
            mode="frozen",
            epochs=2,
            batch_size=2,
            device="cpu",
        )

    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    history = json.loads((output / "history.json").read_text(encoding="utf-8"))
    assert best["epoch"] == 1
    assert history["best_epoch"] == 1
    assert [entry["epoch"] for entry in history["epochs"]] == [1]


def test_pretrain_rejects_nonfinite_gradients_before_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_dataset(tmp_path)

    class FiniteValueNonfiniteGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:
            del ctx
            return value

        @staticmethod
        def backward(ctx: object, gradient: torch.Tensor) -> torch.Tensor:
            del ctx
            return torch.full_like(gradient, torch.nan)

    original_forward = train_module.WindowedPrediction.forward

    def poisoned_forward(self: object, tokens: torch.Tensor):
        loss, metrics = original_forward(self, tokens)
        return FiniteValueNonfiniteGradient.apply(loss), metrics

    monkeypatch.setattr(train_module.WindowedPrediction, "forward", poisoned_forward)
    with pytest.raises(RuntimeError, match="non-finite"):
        pretrain(
            manifest,
            tmp_path / "ssl",
            config_path=tmp_path / "tiny.json",
            epochs=1,
            batch_size=2,
            negatives=2,
            device="cpu",
        )


def test_outputs_are_exclusive_and_checkpoint_kind_is_validated(tmp_path: Path) -> None:
    manifest = _write_dataset(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        pretrain(
            manifest,
            existing,
            config_path=tmp_path / "tiny.json",
            epochs=1,
            batch_size=2,
            negatives=2,
        )

    bad_checkpoint = tmp_path / "bad.pt"
    torch.save({"kind": "mystery"}, bad_checkpoint)
    with pytest.raises(ValueError, match="checkpoint kind"):
        load_for_finetune(bad_checkpoint, mode="frozen", seed=0, device="cpu")

    result_path = tmp_path / "result.json"
    result_path.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        evaluate(manifest, bad_checkpoint, output_json=result_path)
    assert result_path.read_text(encoding="utf-8") == "keep"


def test_example_cli_entry_displays_help() -> None:
    repository = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    completed = subprocess.run(
        [sys.executable, str(repository / "examples" / "train.py"), "--help"],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "{pretrain,finetune,evaluate}" in completed.stdout


def test_label_budget_finetuning_records_its_patient_subset(tmp_path: Path) -> None:
    """Catches a label budget that is silently ignored or left out of the checkpoint."""
    manifest = _write_dataset(tmp_path)
    source = _write_source_checkpoint(tmp_path / "source.pt")
    result = finetune(
        manifest,
        source,
        tmp_path / "budget",
        mode="frozen",
        epochs=1,
        batch_size=2,
        label_fraction=0.25,
        label_seed=0,
    )
    options = torch.load(result["best"], weights_only=True)["training_options"]
    assert options["label_budget"] == {
        "fraction": 0.25,
        "seed": 0,
        "patients": 2,
        "segments": 2,
    }
    with pytest.raises(ValueError, match="below one"):
        finetune(
            manifest, source, tmp_path / "all", mode="frozen", label_fraction=1.0
        )
