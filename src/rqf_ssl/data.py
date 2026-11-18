"""Dataset loading and EEG preprocessing helpers for the TUAB example."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Literal

import numpy as np
import torch


Split = Literal["train", "val", "test"]
_SPLITS = frozenset({"train", "val", "test"})
_MANIFEST_COLUMNS = ["path", "patient_id", "split", "label"]

TCP_AR_BIPOLAR_PAIRS: tuple[tuple[str, str], ...] = (
    ("Fp1", "F7"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("T5", "O1"),
    ("Fp2", "F8"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("T6", "O2"),
    ("Fp1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("Fp2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
)


@dataclass(frozen=True)
class _SegmentRow:
    path: Path
    patient_id: str
    split: str
    label: int


@dataclass(frozen=True)
class _RecordingRow:
    path: Path
    patient_id: str
    split: str
    label: int


def _record_manifest_path(
    assignments: dict[Path, tuple[str, str, int]],
    path: Path,
    patient_id: str,
    split: str,
    label: int,
    kind: str,
) -> None:
    assignment = (patient_id, split, label)
    if path in assignments:
        raise ValueError(
            f"{kind} path occurs multiple times with duplicate or conflicting metadata"
        )
    assignments[path] = assignment


def required_referential_channels() -> list[str]:
    """Return referential channels in the order expected by :func:`derive_bipolar`."""
    channels: list[str] = []
    for pair in TCP_AR_BIPOLAR_PAIRS:
        for channel in pair:
            if channel not in channels:
                channels.append(channel)
    return channels


def derive_bipolar(data: np.ndarray, ref_order: list[str]) -> np.ndarray:
    """Derive the standard TCP-16 bipolar channels from referential EEG."""
    array = np.asarray(data)
    if array.ndim != 2 or array.shape[1] != len(ref_order):
        raise ValueError("referential data must have shape [time, channels]")
    if len(set(ref_order)) != len(ref_order):
        raise ValueError("referential channel names must be unique")
    index = {channel: position for position, channel in enumerate(ref_order)}
    try:
        bipolar = np.stack(
            [
                array[:, index[left]] - array[:, index[right]]
                for left, right in TCP_AR_BIPOLAR_PAIRS
            ],
            axis=1,
        )
    except KeyError as exc:
        raise ValueError(
            f"required referential channel is missing: {exc.args[0]}"
        ) from exc
    return bipolar.astype(np.float32, copy=False)


def robust_q95_scale(segment: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale each channel by its 95th absolute-value percentile."""
    array = np.asarray(segment)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("segment must have shape [time, channels]")
    if not np.isfinite(array).all():
        raise ValueError("segment contains non-finite values")
    q95 = np.quantile(np.abs(array), 0.95, axis=0, keepdims=True)
    scaled = (array / (q95 + eps)).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("scaled segment contains non-finite values")
    return scaled


def segment_bounds(
    sample_count: int,
    segment_samples: int = 12_800,
    step_samples: int = 12_800,
) -> tuple[tuple[int, int], ...]:
    """Return complete crop bounds, omitting any incomplete trailing crop."""
    if sample_count < 0 or segment_samples <= 0 or step_samples <= 0:
        raise ValueError("sample counts must be non-negative and crop sizes positive")
    return tuple(
        (start, start + segment_samples)
        for start in range(0, sample_count - segment_samples + 1, step_samples)
    )


def patient_train_val_split(
    patients: list[str], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Create a deterministic patient-disjoint split for non-test patients."""
    unique = sorted(set(patients))
    if len(unique) < 2:
        raise ValueError("at least two patients are required for train/val splitting")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between zero and one")
    generator = np.random.Generator(np.random.PCG64(seed))
    shuffled = [unique[int(index)] for index in generator.permutation(len(unique))]
    validation_count = min(len(unique) - 1, max(1, round(len(unique) * val_fraction)))
    return shuffled[:-validation_count], shuffled[-validation_count:]


def label_budget_indices(
    patient_ids: list[str], labels: list[int], *, fraction: float, seed: int = 0
) -> list[int]:
    """Keep every segment of a seeded, class-stratified fraction of patients.

    Each class keeps ``max(1, floor(fraction * patients))`` of its patients,
    drawn from one generator (normal patients first). Every fraction reads
    the same permutations, so a smaller budget's patients are always retained
    by a larger one. Returns dataset positions in their original order.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between zero and one")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if len(patient_ids) != len(labels):
        raise ValueError("patient_ids and labels must align")
    label_of: dict[str, int] = {}
    for patient_id, label in zip(patient_ids, labels, strict=True):
        if label_of.setdefault(patient_id, int(label)) != int(label):
            raise ValueError(f"patient {patient_id!r} carries both labels")
    by_label: dict[int, list[str]] = {0: [], 1: []}
    for patient_id, label in label_of.items():
        by_label[label].append(patient_id)
    if not by_label[0] or not by_label[1]:
        raise ValueError("label budgets need patients of both classes")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chosen: set[str] = set()
    for label in (0, 1):
        patients = sorted(by_label[label])
        keep = max(1, int(len(patients) * fraction))
        order = torch.randperm(len(patients), generator=generator).tolist()
        chosen.update(patients[index] for index in order[:keep])
    return [
        position
        for position, patient_id in enumerate(patient_ids)
        if patient_id in chosen
    ]


def _normalize_channel_name(channel: str) -> str:
    normalized = channel.strip().rstrip(".")
    if normalized.upper().startswith("EEG ") or normalized.upper().startswith("EEG-"):
        normalized = normalized[4:]
    for suffix in ("-REF", "-LE"):
        if normalized.upper().endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    canonical = {name.upper(): name for name in required_referential_channels()}
    aliases = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    normalized = aliases.get(
        normalized.upper(), canonical.get(normalized.upper(), normalized)
    )
    return normalized


def _read_recording_manifest(manifest: Path) -> list[_RecordingRow]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _MANIFEST_COLUMNS:
            raise ValueError(
                f"manifest columns must be: {', '.join(_MANIFEST_COLUMNS)}"
            )
        rows = list(reader)

    root = manifest.parent.resolve()
    patients: dict[str, str] = {}
    assignments: dict[Path, tuple[str, str, int]] = {}
    parsed: list[_RecordingRow] = []
    for line_number, row in enumerate(rows, start=2):
        split = row["split"]
        if split not in _SPLITS:
            raise ValueError(f"invalid split on manifest line {line_number}")
        label_text = row["label"]
        if label_text not in {"0", "1"}:
            raise ValueError(f"invalid label on manifest line {line_number}")
        patient_id = row["patient_id"]
        if not patient_id or patient_id.strip() != patient_id:
            raise ValueError(f"invalid patient_id on manifest line {line_number}")
        previous_split = patients.setdefault(patient_id, split)
        if previous_split != split:
            raise ValueError(f"patient {patient_id!r} occurs in multiple splits")

        relative_path = Path(row["path"])
        if relative_path.is_absolute() or relative_path.suffix.casefold() != ".edf":
            raise ValueError(f"invalid EDF path on manifest line {line_number}")
        path = (root / relative_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"invalid EDF path on manifest line {line_number}")
        label = int(label_text)
        _record_manifest_path(assignments, path, patient_id, split, label, "EDF")
        parsed.append(_RecordingRow(path, patient_id, split, label))
    return parsed


def _load_edf(path: Path) -> np.ndarray:
    try:
        import mne
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "EDF preprocessing requires the optional preprocessing dependencies"
        ) from exc

    try:
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    except Exception as exc:
        raise ValueError(f"could not read EDF {path.name!r}") from exc
    try:
        positions: dict[str, int] = {}
        for position, original in enumerate(raw.ch_names):
            channel = _normalize_channel_name(str(original))
            if channel in positions:
                raise ValueError(f"EDF {path.name!r} has duplicate channel {channel!r}")
            positions[channel] = position
        required = required_referential_channels()
        missing = [channel for channel in required if channel not in positions]
        if missing:
            raise ValueError(
                f"EDF {path.name!r} is missing channels: {', '.join(missing)}"
            )

        sample_rate = float(raw.info["sfreq"])
        if not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError(f"EDF {path.name!r} has an invalid sample rate")
        if abs(sample_rate - 128.0) > 1e-6:
            raw.resample(128.0, verbose="ERROR")
        referential = raw.get_data(picks=[positions[channel] for channel in required]).T
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not preprocess EDF {path.name!r}") from exc
    finally:
        raw.close()
    if not np.isfinite(referential).all():
        raise ValueError(f"EDF {path.name!r} contains non-finite samples")
    return derive_bipolar(referential, required)


def preprocess_manifest(input_csv: str | Path, output_dir: str | Path) -> Path:
    """Preprocess EDF recordings into 100-second, 128 Hz TCP-16 segments."""
    manifest = Path(input_csv)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {output.parent}")
    recordings = _read_recording_manifest(manifest)

    with tempfile.TemporaryDirectory(
        prefix=".rqf_ssl_preprocess_", dir=output.parent
    ) as temp:
        staging = Path(temp)
        output_rows: list[dict[str, object]] = []
        segment_number = 0
        for recording in recordings:
            signal = _load_edf(recording.path)
            for start, stop in segment_bounds(signal.shape[0]):
                segment = robust_q95_scale(signal[start:stop])
                filename = f"segment_{segment_number:08d}.npy"
                np.save(staging / filename, segment, allow_pickle=False)
                output_rows.append(
                    {
                        "path": filename,
                        "patient_id": recording.patient_id,
                        "split": recording.split,
                        "label": recording.label,
                    }
                )
                segment_number += 1
        output_manifest = staging / "manifest.csv"
        with output_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(output_rows)
        staging.replace(output)
    return output / "manifest.csv"


def main(argv: list[str] | None = None) -> int:
    """Run the EDF preprocessing command-line interface."""
    parser = argparse.ArgumentParser(
        description="Create 100-second, 128 Hz TCP-16 segments from an EDF manifest."
    )
    parser.add_argument(
        "input_csv", type=Path, help="CSV describing input EDF recordings"
    )
    parser.add_argument(
        "output_dir", type=Path, help="new directory for segments and manifest"
    )
    args = parser.parse_args(argv)
    preprocess_manifest(args.input_csv, args.output_dir)
    return 0


class SegmentDataset:
    """Load preprocessed EEG segments listed in a CSV manifest."""

    def __init__(self, manifest: str | Path, split: Split) -> None:
        if split not in _SPLITS:
            raise ValueError("split must be one of: train, val, test")
        self.manifest = Path(manifest)
        with self.manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != _MANIFEST_COLUMNS:
                raise ValueError(
                    f"manifest columns must be: {', '.join(_MANIFEST_COLUMNS)}"
                )
            rows = list(reader)

        root = self.manifest.parent.resolve()
        patients: dict[str, str] = {}
        assignments: dict[Path, tuple[str, str, int]] = {}
        parsed: list[_SegmentRow] = []
        for line_number, row in enumerate(rows, start=2):
            row_split = row["split"]
            if row_split not in _SPLITS:
                raise ValueError(f"invalid split on manifest line {line_number}")
            label_text = row["label"]
            if label_text not in {"0", "1"}:
                raise ValueError(f"invalid label on manifest line {line_number}")
            patient_id = row["patient_id"]
            if not patient_id or patient_id.strip() != patient_id:
                raise ValueError(f"invalid patient_id on manifest line {line_number}")
            previous_split = patients.setdefault(patient_id, row_split)
            if previous_split != row_split:
                raise ValueError(f"patient {patient_id!r} occurs in multiple splits")

            relative_path = Path(row["path"])
            if relative_path.is_absolute() or relative_path.suffix != ".npy":
                raise ValueError(f"invalid segment path on manifest line {line_number}")
            path = (root / relative_path).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"invalid segment path on manifest line {line_number}")
            label = int(label_text)
            _record_manifest_path(
                assignments, path, patient_id, row_split, label, "segment"
            )
            parsed.append(_SegmentRow(path, patient_id, row_split, label))

        self._rows = [row for row in parsed if row.split == split]
        if not self._rows:
            raise ValueError(f"requested split {split!r} is empty")

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def patient_ids(self) -> list[str]:
        return [row.patient_id for row in self._rows]

    @property
    def labels(self) -> list[int]:
        return [row.label for row in self._rows]

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        row = self._rows[index]
        signal = np.load(row.path, allow_pickle=False)
        if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] != 16:
            raise ValueError(f"segment {row.path.name!r} must have shape [time, 16]")
        if not np.isfinite(signal).all():
            raise ValueError(f"segment {row.path.name!r} contains non-finite values")
        signal = np.asarray(signal, dtype=np.float32)
        if not np.isfinite(signal).all():
            raise ValueError(f"segment {row.path.name!r} is outside the float32 range")
        return signal, row.label
