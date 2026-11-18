"""Exact Speech Commands 10-class preparation and loading."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from typing import Literal, Sequence
import wave

import numpy as np
import torch
from torch.utils.data import Dataset


Split = Literal["train", "val", "test"]
SPLITS = ("train", "val", "test")
CLASSES = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
SAMPLE_RATE = 16_000
SAMPLE_SHAPE = (16_000, 1)
EXPECTED_COUNTS = {"train": 24_478, "val": 5_241, "test": 5_256}
_FORMAT = "rqf_ssl.sc10.v1"
_DATASET = "speech_commands_v0.02_sc10"


def _split_once(
    labels: torch.Tensor,
    indices: torch.Tensor,
    *,
    fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    indexed_labels = labels[indices]
    selected: list[torch.Tensor] = []
    remainder: list[torch.Tensor] = []
    for class_id in torch.unique(indexed_labels):
        positions = torch.nonzero(indexed_labels == class_id).flatten()
        positions = positions[torch.randperm(len(positions), generator=generator)]
        count = int(len(positions) * fraction)
        selected.append(indices[positions[:count]])
        remainder.append(indices[positions[count:]])
    selected_indices = torch.cat(selected)
    remainder_indices = torch.cat(remainder)
    selected_indices = selected_indices[
        torch.randperm(len(selected_indices), generator=generator)
    ]
    remainder_indices = remainder_indices[
        torch.randperm(len(remainder_indices), generator=generator)
    ]
    return selected_indices, remainder_indices


def stratified_split(labels: Sequence[int]) -> dict[str, list[int]]:
    """Return the historical deterministic, clip-level SC10 split."""
    label_tensor = torch.as_tensor(list(labels), dtype=torch.long)
    if label_tensor.ndim != 1 or len(label_tensor) == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if bool(((label_tensor < 0) | (label_tensor >= len(CLASSES))).any()):
        raise ValueError("labels must be SC10 class indices")
    indices = torch.arange(len(label_tensor))
    train, remainder = _split_once(label_tensor, indices, fraction=0.7, seed=0)
    validation, test = _split_once(
        label_tensor,
        remainder,
        fraction=0.15 / (1.0 - 0.7),
        seed=1,
    )
    return {
        "train": train.tolist(),
        "val": validation.tolist(),
        "test": test.tolist(),
    }


def _safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return path


def _decode_selected_wav(payload: bytes, source: str) -> np.ndarray | None:
    try:
        with wave.open(io.BytesIO(payload), "rb") as handle:
            if handle.getcomptype() != "NONE" or handle.getsampwidth() != 2:
                raise ValueError(f"selected WAV {source!r} must use PCM16 encoding")
            if handle.getnchannels() != 1:
                raise ValueError(f"selected WAV {source!r} must be mono")
            if handle.getframerate() != SAMPLE_RATE:
                raise ValueError(
                    f"selected WAV {source!r} must have sample rate {SAMPLE_RATE}"
                )
            frames = handle.getnframes()
            if frames != SAMPLE_SHAPE[0]:
                return None
            raw = handle.readframes(frames)
            if len(raw) != frames * 2 or handle.readframes(1):
                raise ValueError(f"selected WAV {source!r} is malformed")
    except ValueError:
        raise
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"selected WAV {source!r} is malformed") from exc
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return (samples / np.float32(32768.0)).reshape(SAMPLE_SHAPE)


def _archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collect_clips(archive_path: Path) -> tuple[list[str], list[int], list[np.ndarray]]:
    sources: list[str] = []
    labels: list[int] = []
    waveforms: list[np.ndarray] = []
    selected_names: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in archive:
                member_path = _safe_member_name(member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f"archive links are not allowed: {member.name!r}")
                if not member.isfile() or len(member_path.parts) < 2:
                    continue
                class_name = member_path.parts[-2]
                if class_name not in CLASSES or member_path.suffix.casefold() != ".wav":
                    continue
                source = member_path.as_posix()
                if source in selected_names:
                    raise ValueError(f"duplicate selected archive member: {source!r}")
                selected_names.add(source)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"selected WAV {source!r} could not be read")
                waveform = _decode_selected_wav(extracted.read(), source)
                if waveform is None:
                    continue
                sources.append(source)
                labels.append(CLASSES.index(class_name))
                waveforms.append(waveform)
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"could not read archive {archive_path.name!r}") from exc
    if not waveforms:
        raise ValueError("archive contains no usable SC10 clips")
    return sources, labels, waveforms


def prepare_sc10(
    archive: str | Path,
    output_dir: str | Path,
    *,
    require_expected_counts: bool = True,
) -> Path:
    """Prepare an SC10 tar archive into a new manifest-backed directory."""
    archive_path = Path(archive)
    output = Path(output_dir)
    if not archive_path.is_file():
        raise ValueError(f"archive does not exist: {archive_path}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")
    if not output.parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {output.parent}")

    archive_digest = _archive_sha256(archive_path)
    sources, labels, waveforms = _collect_clips(archive_path)
    splits = stratified_split(labels)
    actual_counts = {split: len(splits[split]) for split in SPLITS}
    if require_expected_counts and actual_counts != EXPECTED_COUNTS:
        raise ValueError(f"expected SC10 counts {EXPECTED_COUNTS}, got {actual_counts}")

    train_stack = torch.stack(
        [torch.from_numpy(waveforms[index]) for index in splits["train"]]
    )
    mean = train_stack.mean().item()
    std = train_stack.std(correction=1).item()
    if not math.isfinite(mean) or not math.isfinite(std):
        raise ValueError("training normalization statistics are not finite")

    with tempfile.TemporaryDirectory(
        prefix=".rqf_ssl_sc10_", dir=output.parent
    ) as temp:
        staging = Path(temp)
        clips_dir = staging / "clips"
        clips_dir.mkdir()
        paths: list[str] = []
        for index, waveform in enumerate(waveforms):
            relative = f"clips/clip_{index:08d}.npy"
            np.save(staging / relative, waveform, allow_pickle=False)
            paths.append(relative)
        manifest_splits = {
            split: [
                {"source": sources[index], "path": paths[index], "label": labels[index]}
                for index in splits[split]
            ]
            for split in SPLITS
        }
        manifest = {
            "format": _FORMAT,
            "dataset": _DATASET,
            "canonical": bool(require_expected_counts),
            "classes": list(CLASSES),
            "sample_rate": SAMPLE_RATE,
            "sample_shape": list(SAMPLE_SHAPE),
            "dtype": "float32",
            "normalization": {
                "fit_split": "train",
                "mean": mean,
                "std": std,
                "std_correction": 1,
                "std_floor": 1e-5,
            },
            "archive": {"sha256": archive_digest},
            "counts": {
                **actual_counts,
                "total": len(waveforms),
            },
            "splits": manifest_splits,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        output.mkdir()
        os.rename(clips_dir, output / "clips")
        os.rename(manifest_path, output / "manifest.json")
    return output / "manifest.json"


def _require_dict(value: object, description: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"manifest {description} must be an object")
    return value


class SC10Dataset(Dataset[tuple[torch.Tensor, int]]):
    """Load one ordered split from a prepared SC10 manifest."""

    def __init__(self, manifest: str | Path, split: str) -> None:
        if split not in SPLITS:
            raise ValueError("split must be one of: train, val, test")
        self.manifest = Path(manifest)
        try:
            document = _require_dict(
                json.loads(self.manifest.read_text(encoding="utf-8")), "root"
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("could not read SC10 manifest") from exc
        if document.get("format") != _FORMAT or document.get("dataset") != _DATASET:
            raise ValueError("manifest format or dataset identifier is invalid")
        if document.get("classes") != list(CLASSES):
            raise ValueError("manifest classes are invalid")
        if document.get("sample_rate") != SAMPLE_RATE:
            raise ValueError("manifest sample rate is invalid")
        if document.get("sample_shape") != list(SAMPLE_SHAPE):
            raise ValueError("manifest sample shape is invalid")
        if document.get("dtype") != "float32":
            raise ValueError("manifest dtype is invalid")

        normalization = _require_dict(document.get("normalization"), "normalization")
        if (
            normalization.get("fit_split") != "train"
            or normalization.get("std_correction") != 1
            or normalization.get("std_floor") != 1e-5
        ):
            raise ValueError("manifest normalization declaration is invalid")
        mean = normalization.get("mean")
        std = normalization.get("std")
        if (
            isinstance(mean, bool)
            or not isinstance(mean, (int, float))
            or isinstance(std, bool)
            or not isinstance(std, (int, float))
            or not math.isfinite(mean)
            or not math.isfinite(std)
            or std < 0
        ):
            raise ValueError("manifest normalization statistics are invalid")

        split_rows = _require_dict(document.get("splits"), "splits")
        if set(split_rows) != set(SPLITS):
            raise ValueError("manifest splits are invalid")
        counts = _require_dict(document.get("counts"), "counts")
        root = self.manifest.parent.resolve()
        seen_sources: set[str] = set()
        seen_paths: set[Path] = set()
        parsed: dict[str, list[tuple[Path, int]]] = {name: [] for name in SPLITS}
        for row_split in SPLITS:
            rows = split_rows[row_split]
            if not isinstance(rows, list):
                raise ValueError(f"manifest {row_split} split must be a list")
            if counts.get(row_split) != len(rows):
                raise ValueError(f"manifest {row_split} count is invalid")
            for row in rows:
                row = _require_dict(row, f"{row_split} membership")
                label = row.get("label")
                if (
                    isinstance(label, bool)
                    or not isinstance(label, int)
                    or not 0 <= label < len(CLASSES)
                ):
                    raise ValueError("manifest label is invalid")
                source = row.get("source")
                if not isinstance(source, str) or not source:
                    raise ValueError("manifest source is invalid")
                source_path = _safe_member_name(source)
                if source in seen_sources:
                    raise ValueError("duplicate clip membership across splits")
                if (
                    len(source_path.parts) < 2
                    or source_path.parts[-2] != CLASSES[label]
                ):
                    raise ValueError("manifest source and label disagree")
                seen_sources.add(source)
                relative_text = row.get("path")
                if not isinstance(relative_text, str):
                    raise ValueError("manifest waveform path is invalid")
                relative = Path(relative_text)
                path = (root / relative).resolve()
                if (
                    relative.is_absolute()
                    or relative.suffix != ".npy"
                    or not path.is_relative_to(root)
                ):
                    raise ValueError("manifest waveform path is invalid")
                if path in seen_paths:
                    raise ValueError("duplicate waveform path across multiple splits")
                if not path.is_file():
                    raise ValueError("manifest waveform path does not exist")
                seen_paths.add(path)
                parsed[row_split].append((path, label))
        if counts.get("total") != sum(len(parsed[name]) for name in SPLITS):
            raise ValueError("manifest total count is invalid")

        self._rows = parsed[split]
        self._mean = float(mean)
        self._denominator = max(float(std), 1e-5)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self._rows[index]
        try:
            waveform = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"waveform {path.name!r} could not be loaded") from exc
        if not isinstance(waveform, np.ndarray):
            waveform.close()
            raise ValueError(f"waveform {path.name!r} is not a NumPy array")
        if (
            waveform.shape != SAMPLE_SHAPE
            or waveform.dtype != np.float32
            or not np.isfinite(waveform).all()
        ):
            raise ValueError(
                f"waveform {path.name!r} must be finite float32 with shape {SAMPLE_SHAPE}"
            )
        tensor = torch.from_numpy(waveform)
        return (tensor - self._mean) / self._denominator, label


def main(argv: list[str] | None = None) -> int:
    """Run the SC10 preparation command-line interface."""
    parser = argparse.ArgumentParser(
        description="Prepare an exact SC10 dataset archive."
    )
    parser.add_argument("archive", type=Path, help="Speech Commands tar archive")
    parser.add_argument("output_dir", type=Path, help="new prepared-data directory")
    parser.add_argument(
        "--allow-unexpected-counts",
        action="store_true",
        help="allow noncanonical counts for tests or custom examples",
    )
    args = parser.parse_args(argv)
    prepare_sc10(
        args.archive,
        args.output_dir,
        require_expected_counts=not args.allow_unexpected_counts,
    )
    return 0
