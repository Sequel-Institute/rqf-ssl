"""Standalone SC10 self-supervised training, transfer fitting, and evaluation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rqf_ssl.dense import (
    DensePrediction,
    calibration_loader,
    dense_recipe,
    preserved_rng,
)
from rqf_ssl.objective import WindowedPrediction
from rqf_ssl.sc10_data import SC10Dataset
from rqf_ssl.sc10_model import (
    SC10Classifier,
    SC10Representation,
    sc10_config,
    validate_sc10_config,
)
from rqf_ssl.train import (
    _batch,
    _cpu_tree,
    _finite_number,
    _load_checkpoint,
    _loader,
    _model_state,
    _new_output_directory,
    _positive_int,
    _resolve_device,
    _save_checkpoint,
    _save_json,
    _set_seed,
)


_SOURCE_KINDS = frozenset({"initialization", "pretrained"})
_DOWNSTREAM_KIND = "downstream"
_OBJECTIVES = ("windowed", "dense")


def _validate_new_output_directory(path: str | Path) -> Path:
    """Check a fresh output destination without creating it."""
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {output.parent}")
    return output


def _required_dataset(manifest: str | Path, split: str) -> SC10Dataset:
    """Construct one consumed split and reject an empty training/evaluation loop."""
    dataset = SC10Dataset(manifest, split)
    if not len(dataset):
        raise ValueError(f"SC10 {split} split is empty")
    return dataset


def _load_config(path: str | Path | None) -> dict[str, Any]:
    config: Mapping[str, Any] = sc10_config()
    if path is not None:
        with Path(path).open(encoding="utf-8") as handle:
            overrides = json.load(handle)
        if not isinstance(overrides, dict):
            raise ValueError("architecture config must be a JSON object")
        config = {**config, **overrides}
    return validate_sc10_config(config)


@contextmanager
def _tf32(device: torch.device) -> Iterator[None]:
    """Enable CUDA TF32 for a complete API call and restore caller settings."""
    if device.type != "cuda":
        yield
        return
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


@contextmanager
def _precision(device: torch.device, *, mixed: bool) -> Iterator[None]:
    """Use CUDA bf16 autocast for forward-only work; CPU remains fp32."""
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed):
            yield
    else:
        yield


def _check_train_options(
    *,
    epochs: int,
    batch_size: int,
    num_workers: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    _positive_int("epochs", epochs)
    _positive_int("batch_size", batch_size)
    _positive_int("num_workers", num_workers, allow_zero=True)
    _finite_number("learning_rate", learning_rate, positive=True)
    if _finite_number("weight_decay", weight_decay) < 0:
        raise ValueError("weight_decay must be non-negative")


def _checkpoint_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = payload.get("config", payload.get("model_config"))
    if not isinstance(config, dict):
        raise ValueError("checkpoint is missing model configuration")
    return validate_sc10_config(config)


def _encoder_state(state: object) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing model state")
    selected = {
        name.removeprefix("encoder."): value
        for name, value in state.items()
        if isinstance(name, str)
        and name.startswith("encoder.")
        and isinstance(value, torch.Tensor)
    }
    if not selected:
        raise ValueError("checkpoint is missing encoder state")
    return selected


def _pretrain_payload(
    model: SC10Representation,
    *,
    kind: str,
    epoch: int,
    seed: int,
    options: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    objective: WindowedPrediction | DensePrediction | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": kind,
        "config": dict(model.config),
        "model_config": dict(model.config),
        "model_state": _model_state(model),
        "epoch": epoch,
        "seed": seed,
        "training_options": dict(options),
    }
    if optimizer is not None and objective is not None:
        payload["optimizer_state"] = _cpu_tree(optimizer.state_dict())
        payload["objective_state"] = _model_state(objective)
    return payload


def _calibrate(
    model: SC10Representation,
    objective: DensePrediction,
    loader: DataLoader[Any],
    device: torch.device,
) -> None:
    """Refit dense target statistics in eval mode and fp32, leaving the RNG as is."""
    with preserved_rng(), torch.no_grad(), _precision(device, mixed=False):
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
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    negatives: int = 256,
    temperature: float = 0.1,
    variance_weight: float = 0.04,
    objective: str = "windowed",
    device: str | torch.device = "cpu",
    num_workers: int = 0,
) -> dict[str, Path]:
    """Pretrain only on SC10's train clips, selecting the terminal epoch.

    ``objective="dense"`` replaces windowed prediction with dense future
    log-power prediction; ``variance_weight`` then does not apply.
    """
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {_OBJECTIVES}")
    _check_train_options(
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    _positive_int("negatives", negatives)
    _finite_number("temperature", temperature, positive=True)
    if _finite_number("variance_weight", variance_weight) < 0:
        raise ValueError("variance_weight must be non-negative")
    _positive_int("seed", seed, allow_zero=True)
    output_path = _validate_new_output_directory(output_dir)
    config = _load_config(config_path)
    resolved_device = _resolve_device(device)
    train_dataset = _required_dataset(manifest, "train")
    # Construction and loader sampling deliberately have distinct seed points.
    _set_seed(seed)
    model = SC10Representation(config).to(resolved_device)
    options: dict[str, Any] = {
        "objective": objective,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "negatives": negatives,
        "temperature": float(temperature),
        "num_workers": num_workers,
        "validation": "not_used",
    }
    calibration: DataLoader[Any] | None = None
    if objective == "windowed":
        prediction: WindowedPrediction | DensePrediction = WindowedPrediction(
            2 * config["width"],
            horizon=config["horizon"],
            negatives=negatives,
            temperature=temperature,
            variance_weight=variance_weight,
        )
        options["variance_weight"] = float(variance_weight)
    else:
        recipe = dense_recipe("sc10")
        calibration_seed = recipe.pop("calibration_seed")
        prediction = DensePrediction(
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
            train_dataset,
            examples=prediction.calibration_examples,
            seed=calibration_seed,
            batch_size=batch_size,
        )
    prediction = prediction.to(resolved_device)
    _set_seed(seed)
    output = _new_output_directory(output_path)
    init = output / "init.pt"
    _save_checkpoint(
        init,
        _pretrain_payload(
            model, kind="initialization", epoch=0, seed=seed, options=options
        ),
    )
    train_loader = _loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )
    parameters = [*model.encoder.parameters(), *prediction.parameters()]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
    last = output / "last.pt"
    payload: dict[str, Any] | None = None
    with _tf32(resolved_device):
        if calibration is not None:
            assert isinstance(prediction, DensePrediction)
            _calibrate(model, prediction, calibration, resolved_device)
        for epoch_index in range(epochs):
            model.train()
            prediction.train()
            total, samples = 0.0, 0
            for raw_batch in train_loader:
                signal, _ = _batch(raw_batch, resolved_device)
                optimizer.zero_grad(set_to_none=True)
                with _precision(resolved_device, mixed=False):
                    if isinstance(prediction, DensePrediction):
                        loss, _ = prediction(*model.dense_features(signal))
                    else:
                        loss, _ = prediction(model.summarize(signal).unsqueeze(2))
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise ValueError("pretraining produced a non-finite scalar loss")
                loss.backward()
                for parameter in parameters:
                    if parameter.grad is not None and not bool(
                        torch.isfinite(parameter.grad).all()
                    ):
                        raise ValueError("pretraining produced non-finite gradients")
                optimizer.step()
                total += float(loss.detach()) * len(signal)
                samples += len(signal)
            if calibration is not None:
                assert isinstance(prediction, DensePrediction)
                _calibrate(model, prediction, calibration, resolved_device)
            payload = _pretrain_payload(
                model,
                kind="pretrained",
                epoch=epoch_index + 1,
                seed=seed,
                options=options,
                optimizer=optimizer,
                objective=prediction,
            )
            _save_checkpoint(last, payload)
            print(
                json.dumps(
                    {
                        "epoch": epoch_index + 1,
                        "loss": total / samples,
                        "phase": "pretrain",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    assert payload is not None
    pretrained = output / "pretrained.pt"
    _save_checkpoint(pretrained, payload)
    return {"init": init, "last": last, "pretrained": pretrained}


def load_for_finetune(
    checkpoint: str | Path, *, mode: str, seed: int, device: str | torch.device = "cpu"
) -> tuple[SC10Classifier, str]:
    """Construct a fresh downstream readout and transfer the encoder alone."""
    if mode not in {"frozen", "full-ft"}:
        raise ValueError("mode must be 'frozen' or 'full-ft'")
    _positive_int("seed", seed, allow_zero=True)
    payload = _load_checkpoint(checkpoint)
    kind = payload.get("kind")
    if kind not in _SOURCE_KINDS:
        raise ValueError("checkpoint kind is unsuitable for finetuning")
    _set_seed(seed)
    model = SC10Classifier(_checkpoint_config(payload))
    try:
        model.encoder.load_state_dict(
            _encoder_state(payload.get("model_state")), strict=True
        )
    except (RuntimeError, TypeError) as exc:
        raise ValueError("source checkpoint encoder is incompatible") from exc
    if mode == "frozen":
        model.freeze_encoder()
    else:
        # Downstream full fine-tuning deliberately promotes the otherwise
        # fixed filter time constant into the optimizer along with all encoder
        # parameters.
        model.encoder.requires_grad_(True)
    return model.to(_resolve_device(device)), str(kind)


def classification_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Return SC10 clip-level argmax accuracy without binary assumptions."""
    if logits.ndim != 2 or logits.shape[1] != 10:
        raise ValueError("logits must have shape [samples, 10]")
    if targets.ndim != 1 or len(targets) != len(logits) or not len(targets):
        raise ValueError("targets must have shape [samples]")
    if not bool(torch.isfinite(logits).all()) or not bool(
        ((targets >= 0) & (targets < 10)).all()
    ):
        raise ValueError("metrics require finite logits and SC10 targets")
    return {"accuracy": float((logits.argmax(1) == targets).float().mean())}


def _evaluate_loader(
    model: SC10Classifier, loader: DataLoader[Any], device: torch.device
) -> tuple[dict[str, float], float, int]:
    model.eval()
    logits: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    losses: list[float] = []
    with torch.no_grad():
        for raw_batch in loader:
            signal, target = _batch(raw_batch, device)
            with _precision(device, mixed=device.type == "cuda"):
                output = model(signal)
                loss = F.cross_entropy(output, target)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError("evaluation produced a non-finite scalar loss")
            logits.append(output.cpu())
            targets.append(target.cpu())
            losses.append(float(loss) * len(target))
    if not targets:
        raise ValueError("evaluation split is empty")
    all_targets = torch.cat(targets)
    return (
        classification_metrics(torch.cat(logits), all_targets),
        sum(losses) / len(all_targets),
        len(all_targets),
    )


def _optimizer(model: SC10Classifier) -> torch.optim.Optimizer:
    dynamics: list[torch.nn.Parameter] = []
    remaining: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (
                dynamics
                if any(key in name for key in ("log_theta", "log_gamma", "log_tau"))
                else remaining
            ).append(parameter)
    groups = [
        {"params": group, "lr": 1e-2, "weight_decay": 0.0}
        for group in (remaining, dynamics)
        if group
    ]
    return torch.optim.AdamW(groups, lr=1e-2, weight_decay=0.0, eps=1e-4)


def finetune(
    manifest: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    mode: str,
    seed: int = 0,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    patience: int = 199,
    device: str | torch.device = "cpu",
    num_workers: int = 0,
) -> dict[str, Path]:
    """Fit a fresh SC10 readout, selecting strict validation-accuracy improvements."""
    _check_train_options(
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    _positive_int("patience", patience, allow_zero=True)
    _positive_int("seed", seed, allow_zero=True)
    output_path = _validate_new_output_directory(output_dir)
    resolved_device = _resolve_device(device)
    model, source_kind = load_for_finetune(
        checkpoint, mode=mode, seed=seed, device=resolved_device
    )
    train_dataset = _required_dataset(manifest, "train")
    validation_dataset = _required_dataset(manifest, "val")
    output = _new_output_directory(output_path)
    train_loader = _loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
    )
    val_loader = _loader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
    )
    optimizer = _optimizer(model)
    # The published SC10 recipe has one shared lr, including dynamics/rest groups.
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        group["weight_decay"] = weight_decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=1e-6
    )
    options = {
        "mode": mode,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "eps": 1e-4,
        "scheduler": "cosine",
        "t_max": 200,
        "eta_min": 1e-6,
        "patience": patience,
        "num_workers": num_workers,
    }
    best, history_path = output / "best.pt", output / "history.json"
    history: list[dict[str, float | int]] = []
    best_accuracy, best_epoch, stale = -math.inf, None, 0
    with _tf32(resolved_device):
        for epoch_index in range(epochs):
            model.train()
            total = 0.0
            samples = 0
            for raw_batch in train_loader:
                signal, target = _batch(raw_batch, resolved_device)
                optimizer.zero_grad(set_to_none=True)
                with _precision(resolved_device, mixed=resolved_device.type == "cuda"):
                    loss = F.cross_entropy(model(signal), target)
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise ValueError("finetuning produced a non-finite scalar loss")
                loss.backward()
                for parameter in optimizer.param_groups:
                    for value in parameter["params"]:
                        if value.grad is not None and not bool(
                            torch.isfinite(value.grad).all()
                        ):
                            raise ValueError("finetuning produced non-finite gradients")
                optimizer.step()
                model.clamp_parameters()
                total += float(loss.detach()) * len(target)
                samples += len(target)
            metrics, validation_loss, _ = _evaluate_loader(
                model, val_loader, resolved_device
            )
            scheduler.step()
            record: dict[str, float | int] = {
                "epoch": epoch_index + 1,
                "train_loss": total / samples,
                "validation_loss": validation_loss,
                "validation_accuracy": metrics["accuracy"],
            }
            history.append(record)
            if metrics["accuracy"] > best_accuracy:
                best_accuracy, best_epoch, stale = (
                    metrics["accuracy"],
                    epoch_index + 1,
                    0,
                )
                _save_checkpoint(
                    best,
                    {
                        "kind": _DOWNSTREAM_KIND,
                        "source_kind": source_kind,
                        "config": dict(model.config),
                        "model_config": dict(model.config),
                        "model_state": _model_state(model),
                        "epoch": best_epoch,
                        "seed": seed,
                        "training_options": options,
                        "validation_metrics": metrics,
                        "optimizer_state": _cpu_tree(optimizer.state_dict()),
                    },
                )
            else:
                stale += 1
            _save_json(
                history_path,
                {
                    "source_kind": source_kind,
                    "selection_metric": "validation_accuracy",
                    "best_epoch": best_epoch,
                    "epochs": history,
                },
            )
            print(
                json.dumps({"phase": "finetune", **record}, sort_keys=True), flush=True
            )
            if stale > patience:
                break
    if best_epoch is None:
        raise RuntimeError("no downstream checkpoint was selected")
    return {"best": best, "history": history_path}


def evaluate(
    manifest: str | Path,
    checkpoint: str | Path,
    *,
    output_json: str | Path | None = None,
    batch_size: int = 128,
    device: str | torch.device = "cpu",
    num_workers: int = 0,
    include_cross_entropy: bool = False,
) -> dict[str, float | int]:
    """Evaluate one selected downstream checkpoint on the held-out SC10 test split."""
    _positive_int("batch_size", batch_size)
    _positive_int("num_workers", num_workers, allow_zero=True)
    if not isinstance(include_cross_entropy, bool):
        raise ValueError("include_cross_entropy must be boolean")
    destination = Path(output_json) if output_json is not None else None
    if destination is not None and destination.exists():
        raise FileExistsError(f"output file already exists: {destination}")
    if destination is not None and not destination.parent.is_dir():
        raise ValueError(
            f"output parent directory does not exist: {destination.parent}"
        )
    payload = _load_checkpoint(checkpoint)
    if payload.get("kind") != _DOWNSTREAM_KIND:
        raise ValueError("checkpoint kind is unsuitable for evaluation")
    _set_seed(int(payload.get("seed", 0)))
    resolved_device = _resolve_device(device)
    model = SC10Classifier(_checkpoint_config(payload))
    try:
        model.load_state_dict(payload.get("model_state"), strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("downstream checkpoint model state is incompatible") from exc
    loader = _loader(
        SC10Dataset(manifest, "test"),
        batch_size=batch_size,
        shuffle=False,
        seed=int(payload.get("seed", 0)),
        num_workers=num_workers,
    )
    with _tf32(resolved_device):
        metrics, loss, count = _evaluate_loader(
            model.to(resolved_device), loader, resolved_device
        )
    result: dict[str, float | int] = {**metrics, "count": count}
    if include_cross_entropy:
        result["cross_entropy"] = loss
    rendered = json.dumps(result, sort_keys=True)
    if destination is None:
        print(rendered)
    else:
        _save_json(destination, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("pretrain")
    pre.add_argument("manifest", type=Path)
    pre.add_argument("output_dir", type=Path)
    pre.add_argument("--config", type=Path)
    pre.add_argument("--seed", type=int, default=0)
    pre.add_argument("--epochs", type=int, default=50)
    pre.add_argument("--batch-size", type=int, default=32)
    pre.add_argument("--learning-rate", type=float, default=1e-3)
    pre.add_argument("--weight-decay", type=float, default=1e-2)
    pre.add_argument("--negatives", type=int, default=256)
    pre.add_argument("--temperature", type=float, default=0.1)
    pre.add_argument("--variance-weight", type=float, default=0.04)
    pre.add_argument("--objective", choices=_OBJECTIVES, default="windowed")
    pre.add_argument("--device", default="cpu")
    pre.add_argument("--num-workers", type=int, default=0)
    fine = commands.add_parser("finetune")
    fine.add_argument("manifest", type=Path)
    fine.add_argument("checkpoint", type=Path)
    fine.add_argument("output_dir", type=Path)
    fine.add_argument("--mode", choices=("frozen", "full-ft"), required=True)
    fine.add_argument("--seed", type=int, default=0)
    fine.add_argument("--epochs", type=int, default=200)
    fine.add_argument("--batch-size", type=int, default=128)
    fine.add_argument("--learning-rate", type=float, default=1e-2)
    fine.add_argument("--weight-decay", type=float, default=0.0)
    fine.add_argument("--patience", type=int, default=199)
    fine.add_argument("--device", default="cpu")
    fine.add_argument("--num-workers", type=int, default=0)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("manifest", type=Path)
    evaluate_parser.add_argument("checkpoint", type=Path)
    evaluate_parser.add_argument("--output-json", type=Path)
    evaluate_parser.add_argument("--batch-size", type=int, default=128)
    evaluate_parser.add_argument("--device", default="cpu")
    evaluate_parser.add_argument("--num-workers", type=int, default=0)
    evaluate_parser.add_argument("--include-cross-entropy", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
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
