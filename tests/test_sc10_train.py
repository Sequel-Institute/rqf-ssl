from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch


_CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def _write_manifest(root: Path) -> Path:
    (root / "clips").mkdir()
    splits: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    generator = np.random.default_rng(12)
    for split in splits:
        for label in range(10):
            name = f"{split}_{label}.npy"
            signal = generator.normal(size=(16_000, 1)).astype(np.float32)
            if split == "test":
                signal.fill(np.nan)
            np.save(root / "clips" / name, signal, allow_pickle=False)
            splits[split].append(
                {
                    "source": f"dataset/{_CLASSES[label]}/{split}_{label}.wav",
                    "path": f"clips/{name}",
                    "label": label,
                }
            )
    manifest = {
        "format": "rqf_ssl.sc10.v1",
        "dataset": "speech_commands_v0.02_sc10",
        "canonical": False,
        "classes": _CLASSES,
        "sample_rate": 16_000,
        "sample_shape": [16_000, 1],
        "dtype": "float32",
        "normalization": {
            "fit_split": "train",
            "mean": 0.0,
            "std": 1.0,
            "std_correction": 1,
            "std_floor": 1e-5,
        },
        "archive": {"sha256": "0" * 64},
        "counts": {"train": 10, "val": 10, "test": 10, "total": 30},
        "splits": splits,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_config(root: Path) -> Path:
    path = root / "small.json"
    path.write_text(
        json.dumps(
            {
                "width": 1,
                "layers": 1,
                "readout_hidden": 2,
                "horizon": 1,
                "fbt_dtype": "fp32",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_sc10_training_paths_transfer_only_encoder_and_defer_test_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches test-set reads during fitting and transfers or reuses an SSL readout."""
    from rqf_ssl.sc10_train import evaluate, finetune, load_for_finetune, pretrain

    manifest, config = _write_manifest(tmp_path), _write_config(tmp_path)
    loads: list[str] = []
    original_load = np.load

    def record_load(path: object, *args: object, **kwargs: object) -> np.ndarray:
        loads.append(os.fspath(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", record_load)
    ssl = pretrain(
        manifest,
        tmp_path / "ssl",
        config_path=config,
        epochs=1,
        batch_size=2,
        negatives=2,
        device="cpu",
    )
    assert not any(Path(value).name.startswith("test_") for value in loads)
    init_model, _ = load_for_finetune(ssl["init"], mode="frozen", seed=5)
    pretrained_model, _ = load_for_finetune(ssl["pretrained"], mode="full-ft", seed=5)
    for name, value in init_model.readout.state_dict().items():
        torch.testing.assert_close(
            value, pretrained_model.readout.state_dict()[name], rtol=0, atol=0
        )
    results = {}
    for source in ("init", "pretrained"):
        for mode in ("frozen", "full-ft"):
            results[(source, mode)] = finetune(
                manifest,
                ssl[source],
                tmp_path / f"{source}-{mode}",
                mode=mode,
                seed=5,
                epochs=1,
                batch_size=2,
                device="cpu",
            )
    assert all(result["best"].is_file() for result in results.values())
    assert not any(Path(value).name.startswith("test_") for value in loads)
    selected_best = results[("pretrained", "frozen")]["best"]
    with pytest.raises(ValueError, match="finite"):
        evaluate(manifest, selected_best, batch_size=2, device="cpu")
    assert any(Path(value).name.startswith("test_") for value in loads)


def test_multiclass_accuracy_is_argmax_fraction() -> None:
    """Catches a binary-only metric in the ten-class evaluation API."""
    from rqf_ssl.sc10_train import classification_metrics

    logits = torch.tensor(
        [
            [4.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2.0, 1.0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert classification_metrics(logits, torch.tensor([0, 1, 2])) == pytest.approx(
        {"accuracy": 2 / 3}
    )


def test_pretraining_seed_controls_encoder_initialization_despite_ambient_rng(
    tmp_path: Path,
) -> None:
    """Catches init checkpoints depending on random draws made before pretraining."""
    from rqf_ssl.sc10_train import pretrain

    manifest, config = _write_manifest(tmp_path), _write_config(tmp_path)
    torch.rand(19)
    first = pretrain(
        manifest,
        tmp_path / "first",
        config_path=config,
        epochs=1,
        batch_size=2,
        negatives=2,
        seed=23,
        device="cpu",
    )
    torch.rand(71)
    second = pretrain(
        manifest,
        tmp_path / "second",
        config_path=config,
        epochs=1,
        batch_size=2,
        negatives=2,
        seed=23,
        device="cpu",
    )
    left = torch.load(first["init"], weights_only=True)["model_state"]
    right = torch.load(second["init"], weights_only=True)["model_state"]
    for name, value in left.items():
        torch.testing.assert_close(value, right[name], rtol=0, atol=0)


def test_full_finetuning_reenables_fixed_encoder_tau(tmp_path: Path) -> None:
    """Catches full fine-tuning silently leaving the encoder tau fixed."""
    from rqf_ssl.sc10_model import SC10Representation
    from rqf_ssl.sc10_train import load_for_finetune

    config = {
        "width": 1,
        "layers": 1,
        "sequence_length": 40,
        "window_length": 4,
        "horizon": 1,
        "readout_hidden": 2,
        "fbt_dtype": "fp32",
    }
    source = SC10Representation(config)
    checkpoint = tmp_path / "source.pt"
    torch.save(
        {
            "kind": "initialization",
            "config": config,
            "model_state": source.state_dict(),
            "epoch": 0,
            "seed": 0,
            "training_options": {},
        },
        checkpoint,
    )

    model, _ = load_for_finetune(checkpoint, mode="full-ft", seed=0)

    assert any(
        name.endswith("log_tau") and value.requires_grad
        for name, value in model.named_parameters()
    )
    torch.nn.functional.cross_entropy(
        model(torch.randn(2, 40, 1)), torch.tensor([0, 1])
    ).backward()
    assert any(
        name.endswith("log_tau") and value.grad is not None
        for name, value in model.named_parameters()
    )


def _source_checkpoint(root: Path) -> Path:
    from rqf_ssl.sc10_model import SC10Representation

    config = {
        "width": 1,
        "layers": 1,
        "readout_hidden": 2,
        "horizon": 1,
        "fbt_dtype": "fp32",
    }
    model = SC10Representation(config)
    path = root / "source.pt"
    torch.save(
        {
            "kind": "initialization",
            "config": config,
            "model_state": model.state_dict(),
            "epoch": 0,
            "seed": 0,
            "training_options": {},
        },
        path,
    )
    return path


@pytest.mark.parametrize("manifest_name", ("missing.json", "malformed.json"))
def test_pretrain_invalid_manifest_does_not_create_output(
    tmp_path: Path, manifest_name: str
) -> None:
    """Catches an unusable SSL run directory surviving manifest validation failure."""
    from rqf_ssl.sc10_train import pretrain

    manifest = tmp_path / manifest_name
    if manifest_name == "malformed.json":
        manifest.write_text("{}", encoding="utf-8")
    output = tmp_path / f"ssl-{manifest.stem}"

    with pytest.raises(ValueError):
        pretrain(manifest, output, config_path=_write_config(tmp_path), epochs=1)

    assert not output.exists()


@pytest.mark.parametrize("failure", ("mode", "checkpoint", "device", "manifest"))
def test_finetune_invalid_input_does_not_create_output(
    tmp_path: Path, failure: str
) -> None:
    """Catches an empty downstream directory blocking a corrected retry."""
    from rqf_ssl.sc10_train import finetune

    manifest = _write_manifest(tmp_path)
    checkpoint = _source_checkpoint(tmp_path)
    kwargs: dict[str, object] = {"mode": "frozen", "device": "cpu"}
    if failure == "mode":
        kwargs["mode"] = "unsupported"
    elif failure == "checkpoint":
        checkpoint = tmp_path / "bad.pt"
        torch.save({"kind": "wrong"}, checkpoint)
    elif failure == "device":
        kwargs["device"] = "not-a-device"
    else:
        manifest = tmp_path / "missing.json"
    output = tmp_path / f"fit-{failure}"

    with pytest.raises((ValueError, RuntimeError)):
        finetune(manifest, checkpoint, output, **kwargs)

    assert not output.exists()
