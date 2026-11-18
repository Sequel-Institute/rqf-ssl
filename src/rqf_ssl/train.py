"""Standalone self-supervised training, downstream fitting, and evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from rqf_ssl.data import SegmentDataset, label_budget_indices
from rqf_ssl.dense import (
    DensePrediction,
    calibration_loader,
    dense_recipe,
    preserved_rng,
)
from rqf_ssl.model import TEMPORAL_SUPPORTS, RQFSystem, default_config
from rqf_ssl.objective import NEGATIVE_POLICIES, WindowedPrediction


_SOURCE_KINDS = frozenset({"initialization", "pretrained"})
_DOWNSTREAM_KIND = "downstream"


def _set_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _positive_int(name: str, value: int, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _finite_number(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    converted = float(value)
    if not math.isfinite(converted) or (positive and converted <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return converted


def _resolve_device(device: str | torch.device) -> torch.device:
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid device: {device}") from exc
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return resolved


def _load_config(path: str | Path | None) -> dict[str, Any]:
    config = default_config()
    if path is not None:
        with Path(path).open(encoding="utf-8") as handle:
            overrides = json.load(handle)
        if not isinstance(overrides, dict):
            raise ValueError("architecture config must be a JSON object")
        unknown = sorted(set(overrides).difference(config))
        if unknown:
            raise ValueError(f"unknown architecture option: {unknown[0]}")
        config.update(overrides)
    integer_fields = (
        "width",
        "layers",
        "sequence_length",
        "window_length",
        "horizon",
        "classes",
    )
    for name in integer_fields:
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"architecture option {name!r} must be a positive integer")
    sample_rate = config.get("sample_rate")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, float)):
        raise ValueError("architecture option 'sample_rate' must be numeric")
    if not math.isfinite(float(sample_rate)) or float(sample_rate) <= 0:
        raise ValueError(
            "architecture option 'sample_rate' must be positive and finite"
        )
    if config["classes"] != 2:
        raise ValueError("standalone training requires exactly two classes")
    if config["sequence_length"] % config["window_length"]:
        raise ValueError("sequence length must be divisible by window length")
    windows = config["sequence_length"] // config["window_length"]
    if windows <= config["horizon"]:
        raise ValueError("prediction horizon must be smaller than the window count")
    if not isinstance(config.get("fbt_dtype"), str) or not config["fbt_dtype"]:
        raise ValueError("architecture option 'fbt_dtype' must be a non-empty string")
    if config.get("montage") != "tcp16":
        raise ValueError("the TUAB/TUEG scripts require the 'tcp16' montage")
    if config.get("temporal_support") not in TEMPORAL_SUPPORTS:
        raise ValueError(
            f"architecture option 'temporal_support' must be one of {TEMPORAL_SUPPORTS}"
        )
    return config


def _new_output_directory(path: str | Path) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {output.parent}")
    output.mkdir()
    return output


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.tmp-",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(_cpu_tree(dict(payload)), temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    return payload


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.tmp-",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _loader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def _batch(batch: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    signal, label = batch
    return signal.to(device=device, dtype=torch.float32), label.to(
        device=device, dtype=torch.long
    )


def _validate_training_options(
    *,
    epochs: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
) -> None:
    _positive_int("epochs", epochs)
    _positive_int("batch_size", batch_size)
    _positive_int("num_workers", num_workers, allow_zero=True)
    _finite_number("learning_rate", learning_rate, positive=True)
    decay = _finite_number("weight_decay", weight_decay)
    if decay < 0:
        raise ValueError("weight_decay must be non-negative")
    _finite_number("gradient_clip", gradient_clip, positive=True)


def cosine_factor(
    epoch_index: int, *, epochs: int, warmup_epochs: int, final_fraction: float
) -> float:
    """Per-epoch linear warmup to the peak, then cosine decay towards a floor.

    Epoch ``e < warmup_epochs`` uses ``(e + 1) / warmup_epochs``. Later epochs
    follow a half cosine over ``epochs - warmup_epochs + 1`` steps from one
    to ``final_fraction``.
    """
    if epoch_index < warmup_epochs:
        return (epoch_index + 1) / warmup_epochs
    decay_epochs = epochs - warmup_epochs + 1
    progress = min(1.0, (epoch_index - warmup_epochs + 1) / decay_epochs)
    return final_fraction + (1.0 - final_fraction) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def _check_schedule(
    epochs: int, warmup_epochs: int | None, final_lr_fraction: float
) -> None:
    if warmup_epochs is None:
        return
    _positive_int("warmup_epochs", warmup_epochs)
    if warmup_epochs >= epochs:
        raise ValueError("warmup_epochs must be below epochs")
    fraction = _finite_number("final_lr_fraction", final_lr_fraction, positive=True)
    if fraction > 1:
        raise ValueError("final_lr_fraction must not exceed one")


def _calibrate(
    model: RQFSystem,
    objective: DensePrediction,
    loader: DataLoader[Any],
    device: torch.device,
) -> None:
    """Refit the dense target statistics in eval mode without touching the RNG."""
    with preserved_rng(), torch.no_grad():
        model.eval()
        objective.eval()
        objective.calibrate(
            model.dense_features(_batch(raw_batch, device)[0]) for raw_batch in loader
        )


def pretrain(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    seed: int = 0,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    negatives: int = 256,
    temperature: float = 0.1,
    variance_weight: float = 0.04,
    gradient_clip: float = 1.0,
    include_validation: bool = False,
    objective: str | None = None,
    negative_policy: str = "uniform",
    warmup_epochs: int | None = None,
    final_lr_fraction: float = 0.05,
    device: str | torch.device = "cpu",
    num_workers: int = 0,
) -> dict[str, Path]:
    """Train future prediction without reading any test signals.

    ``negative_policy`` chooses the windowed objective's negatives (see
    ``WindowedPrediction``). With ``warmup_epochs`` the learning rate follows
    ``cosine_factor``; without it the rate stays constant.

    ``objective`` selects windowed prediction or dense future log-power
    prediction and sets the checkpoint's ``temporal_support`` to match, which
    fixes the downstream architecture; ``None`` follows the configuration.
    The dense objective does not use ``variance_weight``.
    """
    if objective is not None and objective not in TEMPORAL_SUPPORTS:
        raise ValueError(f"objective must be one of {TEMPORAL_SUPPORTS}")
    _validate_training_options(
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
    )
    _positive_int("negatives", negatives)
    _finite_number("temperature", temperature, positive=True)
    if _finite_number("variance_weight", variance_weight) < 0:
        raise ValueError("variance_weight must be non-negative")
    if negative_policy not in NEGATIVE_POLICIES:
        raise ValueError(f"negative_policy must be one of {NEGATIVE_POLICIES}")
    _check_schedule(epochs, warmup_epochs, final_lr_fraction)
    if not isinstance(include_validation, bool):
        raise ValueError("include_validation must be boolean")
    _set_seed(seed)
    resolved_device = _resolve_device(device)
    config = _load_config(config_path)
    if objective is not None:
        config["temporal_support"] = objective
    dense = config["temporal_support"] == "dense"
    output = _new_output_directory(output_dir)

    model = RQFSystem(config).to(resolved_device)
    options: dict[str, Any] = {
        "objective": config["temporal_support"],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "negatives": negatives,
        "temperature": float(temperature),
        "variance_weight": float(variance_weight),
        "gradient_clip": float(gradient_clip),
        "include_validation": include_validation,
        "num_workers": num_workers,
    }
    if negative_policy != "uniform":
        options["negative_policy"] = negative_policy
    if warmup_epochs is not None:
        options["schedule"] = {
            "warmup_epochs": warmup_epochs,
            "final_lr_fraction": float(final_lr_fraction),
        }
    train_dataset = SegmentDataset(manifest, "train")
    datasets: list[Any] = [train_dataset]
    if include_validation:
        datasets.append(SegmentDataset(manifest, "val"))
    dataset: Any = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    calibration: DataLoader[Any] | None = None
    if dense:
        recipe = dense_recipe("tuab")
        calibration_seed = recipe.pop("calibration_seed")
        prediction: WindowedPrediction | DensePrediction = DensePrediction(
            config["width"],
            sequence_length=config["sequence_length"],
            negatives=negatives,
            temperature=temperature,
            **{key: value for key, value in recipe.items() if key != "negatives"},
        )
        options["dense"] = {
            **recipe,
            "negatives": negatives,
            "calibration_seed": calibration_seed,
        }
        calibration = calibration_loader(
            dataset,
            examples=prediction.calibration_examples,
            seed=calibration_seed,
            batch_size=batch_size,
        )
    else:
        prediction = WindowedPrediction(
            config["width"],
            horizon=config["horizon"],
            negatives=negatives,
            temperature=temperature,
            variance_weight=variance_weight,
            negative_policy=negative_policy,
        )
    prediction = prediction.to(resolved_device)
    init_path = output / "init.pt"
    _save_checkpoint(
        init_path,
        {
            "kind": "initialization",
            "model_config": config,
            "seed": seed,
            "training_options": options,
            "epoch": 0,
            "model_state": _model_state(model),
        },
    )
    loader = _loader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )
    parameters = [
        parameter
        for module in (*model.representation_modules(), prediction)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = None
    if warmup_epochs is not None:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda index: cosine_factor(
                index,
                epochs=epochs,
                warmup_epochs=warmup_epochs,
                final_fraction=float(final_lr_fraction),
            ),
        )
    last_path = output / "last.pt"
    if calibration is not None:
        assert isinstance(prediction, DensePrediction)
        _calibrate(model, prediction, calibration, resolved_device)
    for epoch_index in range(epochs):
        model.train()
        prediction.train()
        total_loss = 0.0
        samples = 0
        for raw_batch in loader:
            signal, _ = _batch(raw_batch, resolved_device)
            optimizer.zero_grad(set_to_none=True)
            if isinstance(prediction, DensePrediction):
                loss, _ = prediction(*model.dense_features(signal))
            else:
                loss, _ = prediction(model.summarize(signal))
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError("pretraining produced a non-finite scalar loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += float(loss.detach()) * len(signal)
            samples += len(signal)
        if scheduler is not None:
            scheduler.step()
        if calibration is not None:
            assert isinstance(prediction, DensePrediction)
            _calibrate(model, prediction, calibration, resolved_device)
        payload = {
            "kind": "pretrained",
            "model_config": config,
            "seed": seed,
            "training_options": options,
            "epoch": epoch_index + 1,
            "model_state": _model_state(model),
            "optimizer_state": optimizer.state_dict(),
            "objective_state": prediction.state_dict(),
        }
        _save_checkpoint(last_path, payload)
        print(
            json.dumps(
                {
                    "epoch": epoch_index + 1,
                    "loss": total_loss / samples,
                    "phase": "pretrain",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    final_path = output / "pretrained.pt"
    _save_checkpoint(final_path, payload)
    return {"init": init_path, "last": last_path, "pretrained": final_path}


def load_for_finetune(
    checkpoint: str | Path,
    *,
    mode: str,
    seed: int,
    device: str | torch.device = "cpu",
) -> tuple[RQFSystem, str]:
    """Create a fresh downstream model and transfer only its representation.

    The representation is the encoder plus, for windowed support, the pooler.
    """
    if mode not in {"frozen", "full-ft"}:
        raise ValueError("mode must be 'frozen' or 'full-ft'")
    resolved_device = _resolve_device(device)
    payload = _load_checkpoint(checkpoint)
    source_kind = payload.get("kind")
    if source_kind not in _SOURCE_KINDS:
        raise ValueError("checkpoint kind is unsuitable for finetuning")
    config = payload.get("model_config")
    state = payload.get("model_state")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("source checkpoint is missing model configuration or state")
    _set_seed(seed)
    model = RQFSystem(config)
    try:
        for prefix, module in (("encoder.", model.encoder), ("pooler.", model.pooler)):
            if module is None:
                continue
            module.load_state_dict(
                {
                    name.removeprefix(prefix): value
                    for name, value in state.items()
                    if isinstance(name, str) and name.startswith(prefix)
                },
                strict=True,
            )
    except (RuntimeError, TypeError) as exc:
        raise ValueError("source checkpoint representation is incompatible") from exc
    if mode == "frozen":
        model.freeze_representation()
    return model.to(resolved_device), str(source_kind)


def classification_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Compute segment-level binary AUROC and argmax balanced accuracy."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("logits must have shape [samples, 2]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0] or not len(targets):
        raise ValueError("targets must have shape [samples]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits contain non-finite values")
    targets_cpu = targets.detach().cpu().numpy()
    if not np.isin(targets_cpu, (0, 1)).all():
        raise ValueError("metric targets must be binary")
    if np.unique(targets_cpu).size != 2:
        raise ValueError("metric targets must contain both classes")
    scores = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
    predictions = logits.argmax(dim=1).detach().cpu().numpy()
    return {
        "auroc": float(roc_auc_score(targets_cpu, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(targets_cpu, predictions)),
    }


def _evaluate_loader(
    model: RQFSystem, loader: DataLoader[Any], device: torch.device
) -> tuple[dict[str, float], float, int]:
    model.eval()
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    losses: list[float] = []
    with torch.no_grad():
        for raw_batch in loader:
            signal, target = _batch(raw_batch, device)
            output = model(signal)
            loss = F.cross_entropy(output, target)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError("evaluation produced a non-finite scalar loss")
            logits.append(output.detach().cpu())
            targets.append(target.detach().cpu())
            losses.append(float(loss) * len(target))
    if not targets:
        raise ValueError("evaluation split is empty")
    all_targets = torch.cat(targets)
    metrics = classification_metrics(torch.cat(logits), all_targets)
    return metrics, sum(losses) / len(all_targets), len(all_targets)


def _epoch_factor(epoch_index: int) -> float:
    if epoch_index <= 5:
        return 0.2 + 0.8 * epoch_index / 5
    return 0.99 ** (epoch_index - 5)


def _set_epoch_learning_rates(
    optimizer: torch.optim.Optimizer, base_rates: list[float], epoch_index: int
) -> None:
    factor = _epoch_factor(epoch_index)
    for group, base_rate in zip(optimizer.param_groups, base_rates, strict=True):
        group["lr"] = base_rate * factor


def finetune(
    manifest: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    mode: str,
    seed: int = 0,
    epochs: int = 20,
    batch_size: int = 4,
    upstream_learning_rate: float = 2e-4,
    readout_learning_rate: float = 2e-3,
    weight_decay: float = 0.01,
    gradient_clip: float = 1.0,
    label_fraction: float | None = None,
    label_seed: int = 0,
    device: str | torch.device = "cpu",
    num_workers: int = 0,
) -> dict[str, Path]:
    """Fit a fresh binary readout, selecting solely by validation AUROC.

    ``label_fraction`` trains on a class-stratified subset of training patients
    (see ``label_budget_indices``); validation and test patients are unchanged.
    """
    _validate_training_options(
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=readout_learning_rate,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
    )
    _finite_number("upstream_learning_rate", upstream_learning_rate, positive=True)
    if label_fraction is not None:
        _finite_number("label_fraction", label_fraction, positive=True)
        if label_fraction >= 1:
            raise ValueError("label_fraction must be below one; omit it for all labels")
    _positive_int("label_seed", label_seed, allow_zero=True)
    output = _new_output_directory(output_dir)
    resolved_device = _resolve_device(device)
    model, source_kind = load_for_finetune(
        checkpoint, mode=mode, seed=seed, device=resolved_device
    )
    train_dataset: Any = SegmentDataset(manifest, "train")
    label_budget: dict[str, Any] | None = None
    if label_fraction is not None:
        patient_ids = train_dataset.patient_ids
        indices = label_budget_indices(
            patient_ids,
            train_dataset.labels,
            fraction=float(label_fraction),
            seed=label_seed,
        )
        label_budget = {
            "fraction": float(label_fraction),
            "seed": label_seed,
            "patients": len({patient_ids[index] for index in indices}),
            "segments": len(indices),
        }
        train_dataset = Subset(train_dataset, indices)
    validation_dataset = SegmentDataset(manifest, "val")
    train_loader = _loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )
    readout_parameters = [p for p in model.readout.parameters() if p.requires_grad]
    groups: list[dict[str, Any]] = [
        {"params": readout_parameters, "lr": readout_learning_rate}
    ]
    base_rates = [float(readout_learning_rate)]
    if mode == "full-ft":
        upstream_parameters = [
            parameter
            for module in model.representation_modules()
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        groups.insert(0, {"params": upstream_parameters, "lr": upstream_learning_rate})
        base_rates.insert(0, float(upstream_learning_rate))
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    options = {
        "mode": mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "upstream_learning_rate": float(upstream_learning_rate),
        "readout_learning_rate": float(readout_learning_rate),
        "weight_decay": float(weight_decay),
        "gradient_clip": float(gradient_clip),
        "warmup_epochs": 5,
        "warmup_start_factor": 0.2,
        "decay_gamma": 0.99,
        "num_workers": num_workers,
        "label_budget": label_budget,
    }
    history: list[dict[str, float | int]] = []
    best_auroc = -math.inf
    best_epoch: int | None = None
    best_path = output / "best.pt"
    history_path = output / "history.json"
    for epoch_index in range(epochs):
        _set_epoch_learning_rates(optimizer, base_rates, epoch_index)
        model.train()
        total_loss = 0.0
        samples = 0
        for raw_batch in train_loader:
            signal, target = _batch(raw_batch, resolved_device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(signal), target)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError("finetuning produced a non-finite scalar loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters, gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            samples += len(target)
        validation_metrics, validation_loss, _ = _evaluate_loader(
            model, validation_loader, resolved_device
        )
        epoch_record: dict[str, float | int] = {
            "epoch": epoch_index + 1,
            "train_loss": total_loss / samples,
            "validation_loss": validation_loss,
            "validation_auroc": validation_metrics["auroc"],
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
        }
        history.append(epoch_record)
        if validation_metrics["auroc"] > best_auroc:
            best_auroc = validation_metrics["auroc"]
            best_epoch = epoch_index + 1
            _save_checkpoint(
                best_path,
                {
                    "kind": _DOWNSTREAM_KIND,
                    "source_kind": source_kind,
                    "model_config": dict(model.config),
                    "seed": seed,
                    "training_options": options,
                    "epoch": best_epoch,
                    "validation_metrics": validation_metrics,
                    "model_state": _model_state(model),
                    "optimizer_state": _cpu_tree(optimizer.state_dict()),
                },
            )
        _save_json(
            history_path,
            {
                "source_kind": source_kind,
                "selection_metric": "validation_auroc",
                "best_epoch": best_epoch,
                "epochs": history,
            },
        )
        print(
            json.dumps({"phase": "finetune", **epoch_record}, sort_keys=True),
            flush=True,
        )
    if best_epoch is None:
        raise RuntimeError("no downstream checkpoint was selected")
    return {"best": best_path, "history": history_path}


def evaluate(
    manifest: str | Path,
    checkpoint: str | Path,
    *,
    output_json: str | Path | None = None,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
    num_workers: int = 0,
) -> dict[str, float | int]:
    """Evaluate a selected downstream checkpoint on the held-out test split."""
    _positive_int("batch_size", batch_size)
    _positive_int("num_workers", num_workers, allow_zero=True)
    destination = Path(output_json) if output_json is not None else None
    if destination is not None:
        if destination.exists():
            raise FileExistsError(f"output file already exists: {destination}")
        if not destination.parent.is_dir():
            raise ValueError(
                f"output parent directory does not exist: {destination.parent}"
            )
    payload = _load_checkpoint(checkpoint)
    if payload.get("kind") != _DOWNSTREAM_KIND:
        raise ValueError("checkpoint kind is unsuitable for evaluation")
    config = payload.get("model_config")
    state = payload.get("model_state")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError(
            "downstream checkpoint is missing model configuration or state"
        )
    resolved_device = _resolve_device(device)
    _set_seed(int(payload.get("seed", 0)))
    model = RQFSystem(config)
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("downstream checkpoint model state is incompatible") from exc
    model.to(resolved_device)
    test_dataset = SegmentDataset(manifest, "test")
    test_loader = _loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=int(payload.get("seed", 0)),
        num_workers=num_workers,
    )
    metrics, _, samples = _evaluate_loader(model, test_loader, resolved_device)
    result: dict[str, float | int] = {**metrics, "samples": samples}
    rendered = json.dumps(result, sort_keys=True)
    if destination is None:
        print(rendered)
    else:
        destination.write_text(rendered + "\n", encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain_parser = subparsers.add_parser(
        "pretrain", help="run self-supervised pretraining"
    )
    pretrain_parser.add_argument("manifest", type=Path)
    pretrain_parser.add_argument("output_dir", type=Path)
    pretrain_parser.add_argument("--config", type=Path)
    pretrain_parser.add_argument("--seed", type=int, default=0)
    pretrain_parser.add_argument("--epochs", type=int, default=30)
    pretrain_parser.add_argument("--batch-size", type=int, default=8)
    pretrain_parser.add_argument("--learning-rate", type=float, default=1e-4)
    pretrain_parser.add_argument("--weight-decay", type=float, default=0.05)
    pretrain_parser.add_argument("--negatives", type=int, default=256)
    pretrain_parser.add_argument("--temperature", type=float, default=0.1)
    pretrain_parser.add_argument("--variance-weight", type=float, default=0.04)
    pretrain_parser.add_argument("--gradient-clip", type=float, default=1.0)
    pretrain_parser.add_argument("--include-validation", action="store_true")
    pretrain_parser.add_argument("--objective", choices=TEMPORAL_SUPPORTS)
    pretrain_parser.add_argument(
        "--negative-policy", choices=NEGATIVE_POLICIES, default="uniform"
    )
    pretrain_parser.add_argument("--warmup-epochs", type=int)
    pretrain_parser.add_argument("--final-lr-fraction", type=float, default=0.05)
    pretrain_parser.add_argument("--device", default="cpu")
    pretrain_parser.add_argument("--num-workers", type=int, default=0)

    finetune_parser = subparsers.add_parser(
        "finetune", help="fit a downstream classifier"
    )
    finetune_parser.add_argument("manifest", type=Path)
    finetune_parser.add_argument("checkpoint", type=Path)
    finetune_parser.add_argument("output_dir", type=Path)
    finetune_parser.add_argument("--mode", choices=("frozen", "full-ft"), required=True)
    finetune_parser.add_argument("--seed", type=int, default=0)
    finetune_parser.add_argument("--epochs", type=int, default=20)
    finetune_parser.add_argument("--batch-size", type=int, default=4)
    finetune_parser.add_argument("--upstream-learning-rate", type=float, default=2e-4)
    finetune_parser.add_argument("--readout-learning-rate", type=float, default=2e-3)
    finetune_parser.add_argument("--weight-decay", type=float, default=0.01)
    finetune_parser.add_argument("--gradient-clip", type=float, default=1.0)
    finetune_parser.add_argument("--label-fraction", type=float)
    finetune_parser.add_argument("--label-seed", type=int, default=0)
    finetune_parser.add_argument("--device", default="cpu")
    finetune_parser.add_argument("--num-workers", type=int, default=0)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate a selected checkpoint"
    )
    evaluate_parser.add_argument("manifest", type=Path)
    evaluate_parser.add_argument("checkpoint", type=Path)
    evaluate_parser.add_argument("--output-json", type=Path)
    evaluate_parser.add_argument("--batch-size", type=int, default=4)
    evaluate_parser.add_argument("--device", default="cpu")
    evaluate_parser.add_argument("--num-workers", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the standalone training command-line interface."""
    args = vars(_parser().parse_args(argv))
    command = args.pop("command")
    if command == "pretrain":
        args["config_path"] = args.pop("config")
        pretrain(**args)
    elif command == "finetune":
        finetune(**args)
    else:
        evaluate(**args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
