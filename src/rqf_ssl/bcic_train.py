"""BCI-IV-2a in-domain pretraining, encoder transfer, and evaluation.

Pretraining predicts the window summary one second ahead from eight
half-second windows per trial, with distinct negatives from the same channel
of other trials. Downstream training transfers the encoder alone (from an
in-domain, TUAB, or TUEG checkpoint) and trains a fresh pooler and spatial
readout over two two-second windows, selecting by validation Cohen's kappa.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from rqf_ssl.bcic_data import BCICDataset
from rqf_ssl.model import RQFSystem
from rqf_ssl.objective import WindowedPrediction
from rqf_ssl.sc10_train import _precision, _tf32
from rqf_ssl.train import (
    _batch,
    _check_schedule,
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
    _set_epoch_learning_rates,
    _set_seed,
    cosine_factor,
)

_SOURCE_KINDS = frozenset({"initialization", "pretrained"})
_DOWNSTREAM_KIND = "downstream"
DOWNSTREAM_WINDOW = 400
ALIGNMENTS = ("none", "euclidean")


def bcic_config() -> dict[str, Any]:
    """The pretraining architecture: 200 Hz, eight 0.5 s windows, 1 s horizon."""
    return dict(
        width=64,
        layers=4,
        sample_rate=200.0,
        sequence_length=800,
        window_length=100,
        horizon=2,
        classes=4,
        fbt_dtype="bf16",
        temporal_support="windowed",
        montage="bcic22",
    )


def downstream_config() -> dict[str, Any]:
    """The downstream architecture: two non-overlapping two-second windows."""
    return {**bcic_config(), "window_length": DOWNSTREAM_WINDOW}


def _encoder_state(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if payload.get("kind") not in _SOURCE_KINDS:
        raise ValueError("checkpoint kind is unsuitable as an encoder source")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("source checkpoint is missing model state")
    encoder = {
        name.removeprefix("encoder."): value
        for name, value in state.items()
        if isinstance(name, str) and name.startswith("encoder.")
    }
    if not encoder:
        raise ValueError("source checkpoint is missing encoder state")
    return encoder


def _load_encoder(model: RQFSystem, checkpoint: str | Path) -> str:
    """Copy every encoder tensor of a source checkpoint, as in the paper.

    The copy is verbatim, including the frequency-range buffers, so a 128 Hz
    TUAB or TUEG encoder keeps its learned parameters inside the 200 Hz model.
    """
    payload = _load_checkpoint(checkpoint)
    try:
        model.encoder.load_state_dict(_encoder_state(payload), strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("source checkpoint encoder is incompatible") from exc
    return str(payload["kind"])


def pretrain(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    init_from: str | Path | None = None,
    seed: int = 0,
    epochs: int = 60,
    batch_size: int = 240,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.05,
    gradient_clip: float = 1.0,
    warmup_epochs: int = 5,
    final_lr_fraction: float = 0.05,
    negatives: int = 256,
    temperature: float = 0.1,
    variance_weight: float = 0.04,
    device: str | torch.device = "cpu",
    num_workers: int = 0,
) -> dict[str, Path]:
    """Pretrain on unaligned training-subject trials, keeping the terminal epoch.

    ``init_from`` copies the encoder of an existing checkpoint before any
    update, so the pretrained model and its paired initialization share one
    starting point with another run.
    """
    for name, value in (
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("negatives", negatives),
    ):
        _positive_int(name, value)
    _positive_int("seed", seed, allow_zero=True)
    _positive_int("num_workers", num_workers, allow_zero=True)
    _finite_number("learning_rate", learning_rate, positive=True)
    _finite_number("gradient_clip", gradient_clip, positive=True)
    _finite_number("temperature", temperature, positive=True)
    if _finite_number("weight_decay", weight_decay) < 0:
        raise ValueError("weight_decay must be non-negative")
    if _finite_number("variance_weight", variance_weight) < 0:
        raise ValueError("variance_weight must be non-negative")
    _check_schedule(epochs, warmup_epochs, final_lr_fraction)
    resolved_device = _resolve_device(device)
    dataset = BCICDataset(data_dir, "train", ssl=True)
    config = bcic_config()
    _set_seed(seed)
    model = RQFSystem(config)
    source_kind = None if init_from is None else _load_encoder(model, init_from)
    model = model.to(resolved_device)
    objective = WindowedPrediction(
        config["width"],
        horizon=config["horizon"],
        negatives=negatives,
        temperature=temperature,
        variance_weight=variance_weight,
        negative_policy="same_channel",
    ).to(resolved_device)
    options = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "gradient_clip": float(gradient_clip),
        "schedule": {
            "warmup_epochs": warmup_epochs,
            "final_lr_fraction": float(final_lr_fraction),
        },
        "negatives": negatives,
        "negative_policy": "same_channel",
        "temperature": float(temperature),
        "variance_weight": float(variance_weight),
        "init_from": None if init_from is None else str(init_from),
        "init_source_kind": source_kind,
        "trials": len(dataset),
        "num_workers": num_workers,
    }
    output = _new_output_directory(output_dir)
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
        for module in (*model.representation_modules(), objective)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
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
    payload: dict[str, Any] | None = None
    with _tf32(resolved_device):
        for epoch_index in range(epochs):
            model.train()
            objective.train()
            total, samples = 0.0, 0
            for raw_batch in loader:
                signal, _ = _batch(raw_batch, resolved_device)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = objective(model.summarize(signal))
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise ValueError("pretraining produced a non-finite scalar loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters, gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                total += float(loss.detach()) * len(signal)
                samples += len(signal)
            scheduler.step()
            payload = {
                "kind": "pretrained",
                "model_config": config,
                "seed": seed,
                "training_options": options,
                "epoch": epoch_index + 1,
                "model_state": _model_state(model),
                "optimizer_state": _cpu_tree(optimizer.state_dict()),
                "objective_state": _model_state(objective),
            }
            _save_checkpoint(last_path, payload)
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
    final_path = output / "pretrained.pt"
    _save_checkpoint(final_path, payload)
    return {"init": init_path, "last": last_path, "pretrained": final_path}


def classification_metrics(
    logits: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Four-class balanced accuracy, Cohen's kappa, and weighted F1."""
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise ValueError("logits must have shape [samples, 4]")
    if targets.ndim != 1 or len(targets) != len(logits) or not len(targets):
        raise ValueError("targets must have shape [samples]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits contain non-finite values")
    truth = targets.cpu().numpy()
    predicted = logits.argmax(1).cpu().numpy()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "cohen_kappa": float(cohen_kappa_score(truth, predicted)),
        "weighted_f1": float(
            f1_score(truth, predicted, average="weighted", labels=[0, 1, 2, 3])
        ),
    }


def _evaluate_loader(
    model: RQFSystem, loader: DataLoader[Any], device: torch.device
) -> tuple[dict[str, float], float]:
    model.eval()
    logits, targets, losses = [], [], []
    with torch.no_grad():
        for raw_batch in loader:
            signal, target = _batch(raw_batch, device)
            with _precision(device, mixed=True):
                output = model(signal)
            loss = F.cross_entropy(output.float(), target)
            if not bool(torch.isfinite(loss)):
                raise ValueError("evaluation produced a non-finite loss")
            logits.append(output.float().cpu())
            targets.append(target.cpu())
            losses.append(float(loss) * len(target))
    all_targets = torch.cat(targets)
    return (
        classification_metrics(torch.cat(logits), all_targets),
        sum(losses) / len(all_targets),
    )


def _alignment(value: str) -> str | None:
    if value not in ALIGNMENTS:
        raise ValueError(f"alignment must be one of {ALIGNMENTS}")
    return None if value == "none" else value


def finetune(
    data_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    mode: str = "full-ft",
    alignment: str = "euclidean",
    seed: int = 0,
    epochs: int = 50,
    batch_size: int = 32,
    upstream_learning_rate: float = 2e-4,
    readout_learning_rate: float = 2e-3,
    weight_decay: float = 1e-2,
    patience: int = 10,
    device: str | torch.device = "cpu",
    num_workers: int = 0,
) -> dict[str, Path]:
    """Transfer the encoder, train a fresh pooler and readout, select by kappa.

    The best epoch is the first with the highest validation kappa; training
    stops after ``patience`` epochs without a strict improvement.
    """
    if mode not in {"frozen", "full-ft"}:
        raise ValueError("mode must be 'frozen' or 'full-ft'")
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        _positive_int(name, value)
    for name, value in (("seed", seed), ("patience", patience)):
        _positive_int(name, value, allow_zero=True)
    _positive_int("num_workers", num_workers, allow_zero=True)
    _finite_number("upstream_learning_rate", upstream_learning_rate, positive=True)
    _finite_number("readout_learning_rate", readout_learning_rate, positive=True)
    if _finite_number("weight_decay", weight_decay) < 0:
        raise ValueError("weight_decay must be non-negative")
    aligned = _alignment(alignment)
    resolved_device = _resolve_device(device)
    train_dataset = BCICDataset(data_dir, "train", alignment=aligned)
    validation_dataset = BCICDataset(data_dir, "val", alignment=aligned)
    _set_seed(seed)
    model = RQFSystem(downstream_config())
    source_kind = _load_encoder(model, checkpoint)
    # As in the paper's recipe, every encoder parameter trains under full
    # fine-tuning, including the otherwise fixed filter time constants.
    model.requires_grad_(True)
    if mode == "frozen":
        model.encoder.requires_grad_(False)
    model = model.to(resolved_device)
    output = _new_output_directory(output_dir)
    train_loader = _loader(
        train_dataset, batch_size=batch_size, shuffle=True, seed=seed,
        num_workers=num_workers,
    )  # fmt: skip
    validation_loader = _loader(
        validation_dataset, batch_size=batch_size, shuffle=False, seed=seed,
        num_workers=num_workers,
    )  # fmt: skip
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    fresh = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in encoder_ids
    ]
    groups: list[dict[str, Any]] = [{"params": fresh, "lr": readout_learning_rate}]
    base_rates = [float(readout_learning_rate)]
    if mode == "full-ft":
        encoder_group = list(model.encoder.parameters())
        groups.insert(0, {"params": encoder_group, "lr": upstream_learning_rate})
        base_rates.insert(0, float(upstream_learning_rate))
    optimizer = torch.optim.AdamW(groups, weight_decay=weight_decay)
    options = {
        "mode": mode,
        "alignment": alignment,
        "epochs": epochs,
        "batch_size": batch_size,
        "upstream_learning_rate": float(upstream_learning_rate),
        "readout_learning_rate": float(readout_learning_rate),
        "weight_decay": float(weight_decay),
        "warmup_epochs": 5,
        "warmup_start_factor": 0.2,
        "decay_gamma": 0.99,
        "patience": patience,
        "selection_metric": "validation_cohen_kappa",
        "num_workers": num_workers,
    }
    best_path, history_path = output / "best.pt", output / "history.json"
    history: list[dict[str, float | int]] = []
    best_kappa, best_epoch, stale = -math.inf, None, 0
    with _tf32(resolved_device):
        for epoch_index in range(epochs):
            _set_epoch_learning_rates(optimizer, base_rates, epoch_index)
            model.train()
            total, samples = 0.0, 0
            for raw_batch in train_loader:
                signal, target = _batch(raw_batch, resolved_device)
                optimizer.zero_grad(set_to_none=True)
                with _precision(resolved_device, mixed=True):
                    loss = F.cross_entropy(model(signal).float(), target)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("finetuning produced a non-finite loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(target)
                samples += len(target)
            metrics, validation_loss = _evaluate_loader(
                model, validation_loader, resolved_device
            )
            record: dict[str, float | int] = {
                "epoch": epoch_index + 1,
                "train_loss": total / samples,
                "validation_loss": validation_loss,
                **{f"validation_{name}": value for name, value in metrics.items()},
            }
            history.append(record)
            kappa = metrics["cohen_kappa"]
            if math.isfinite(kappa) and kappa > best_kappa:
                best_kappa, best_epoch, stale = kappa, epoch_index + 1, 0
                _save_checkpoint(
                    best_path,
                    {
                        "kind": _DOWNSTREAM_KIND,
                        "source_kind": source_kind,
                        "model_config": dict(model.config),
                        "seed": seed,
                        "training_options": options,
                        "epoch": best_epoch,
                        "validation_metrics": metrics,
                        "model_state": _model_state(model),
                    },
                )
            else:
                stale += 1
            _save_json(
                history_path,
                {
                    "source_kind": source_kind,
                    "selection_metric": "validation_cohen_kappa",
                    "best_epoch": best_epoch,
                    "epochs": history,
                },
            )
            rendered = json.dumps({"phase": "finetune", **record}, sort_keys=True)
            print(rendered, flush=True)
            if stale >= patience:
                break
    if best_epoch is None:
        raise RuntimeError("no downstream checkpoint was selected")
    return {"best": best_path, "history": history_path}


def evaluate(
    data_dir: str | Path,
    checkpoint: str | Path,
    *,
    output_json: str | Path | None = None,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    num_workers: int = 0,
) -> dict[str, float | int]:
    """Evaluate the selected checkpoint once on the test subjects."""
    _positive_int("batch_size", batch_size)
    destination = Path(output_json) if output_json is not None else None
    if destination is not None and destination.exists():
        raise FileExistsError(f"output file already exists: {destination}")
    payload = _load_checkpoint(checkpoint)
    if payload.get("kind") != _DOWNSTREAM_KIND:
        raise ValueError("checkpoint kind is unsuitable for evaluation")
    options = payload.get("training_options", {})
    resolved_device = _resolve_device(device)
    _set_seed(int(payload.get("seed", 0)))
    model = RQFSystem(payload["model_config"])
    try:
        model.load_state_dict(payload["model_state"], strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("downstream checkpoint model state is incompatible") from exc
    test_dataset = BCICDataset(
        data_dir, "test", alignment=_alignment(options.get("alignment", "none"))
    )
    loader = _loader(
        test_dataset, batch_size=batch_size, shuffle=False, seed=0,
        num_workers=num_workers,
    )  # fmt: skip
    with _tf32(resolved_device):
        model = model.to(resolved_device)
        metrics, _ = _evaluate_loader(model, loader, resolved_device)
    result: dict[str, float | int] = {**metrics, "samples": len(test_dataset)}
    if destination is None:
        print(json.dumps(result, sort_keys=True))
    else:
        _save_json(destination, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("pretrain")
    pre.add_argument("data_dir", type=Path)
    pre.add_argument("output_dir", type=Path)
    pre.add_argument("--init-from", type=Path)
    pre.add_argument("--seed", type=int, default=0)
    pre.add_argument("--epochs", type=int, default=60)
    pre.add_argument("--batch-size", type=int, default=240)
    pre.add_argument("--learning-rate", type=float, default=1e-4)
    pre.add_argument("--weight-decay", type=float, default=0.05)
    pre.add_argument("--warmup-epochs", type=int, default=5)
    pre.add_argument("--final-lr-fraction", type=float, default=0.05)
    pre.add_argument("--device", default="cpu")
    pre.add_argument("--num-workers", type=int, default=0)
    fine = commands.add_parser("finetune")
    fine.add_argument("data_dir", type=Path)
    fine.add_argument("checkpoint", type=Path)
    fine.add_argument("output_dir", type=Path)
    fine.add_argument("--mode", choices=("frozen", "full-ft"), default="full-ft")
    fine.add_argument("--alignment", choices=ALIGNMENTS, default="euclidean")
    fine.add_argument("--seed", type=int, default=0)
    fine.add_argument("--epochs", type=int, default=50)
    fine.add_argument("--batch-size", type=int, default=32)
    fine.add_argument("--patience", type=int, default=10)
    fine.add_argument("--device", default="cpu")
    fine.add_argument("--num-workers", type=int, default=0)
    ev = commands.add_parser("evaluate")
    ev.add_argument("data_dir", type=Path)
    ev.add_argument("checkpoint", type=Path)
    ev.add_argument("--output-json", type=Path)
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--num-workers", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = vars(_parser().parse_args(argv))
    command = args.pop("command")
    {"pretrain": pretrain, "finetune": finetune, "evaluate": evaluate}[command](**args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
