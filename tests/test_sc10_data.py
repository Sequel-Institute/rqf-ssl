from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import wave

import numpy as np
import pytest
import torch

from rqf_ssl.sc10_data import SC10Dataset, prepare_sc10, stratified_split


CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def _wav_bytes(
    samples: np.ndarray,
    *,
    channels: int = 1,
    sample_rate: int = 16_000,
    sample_width: int = 2,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(samples).tobytes())
    return buffer.getvalue()


def _write_archive(
    path: Path,
    members: list[tuple[str, bytes | tarfile.TarInfo]],
) -> Path:
    with tarfile.open(path, "w:gz") as archive:
        for name, value in members:
            if isinstance(value, tarfile.TarInfo):
                archive.addfile(value)
                continue
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    return path


def _valid_clip(value: int = 0, *, frames: int = 16_000) -> bytes:
    return _wav_bytes(np.full(frames, value, dtype="<i2"))


def _manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_loader_fixture(root: Path) -> Path:
    (root / "clips").mkdir()
    np.save(root / "clips/train.npy", np.zeros((16_000, 1), dtype=np.float32))
    np.save(root / "clips/test.npy", np.ones((16_000, 1), dtype=np.float32))
    manifest = {
        "format": "rqf_ssl.sc10.v1",
        "dataset": "speech_commands_v0.02_sc10",
        "canonical": False,
        "classes": CLASSES,
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
        "counts": {"train": 1, "val": 0, "test": 1, "total": 2},
        "splits": {
            "train": [
                {
                    "source": "dataset/yes/train.wav",
                    "path": "clips/train.npy",
                    "label": 0,
                }
            ],
            "val": [],
            "test": [
                {"source": "dataset/no/test.wav", "path": "clips/test.npy", "label": 1}
            ],
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_split_preserves_all_samples_without_overlap() -> None:
    result = stratified_split([0] * 10 + [1] * 10)

    assert [len(result[split]) for split in ("train", "val", "test")] == [14, 2, 4]
    assert sorted(sum(result.values(), [])) == list(range(20))


def test_split_matches_archive_position_sensitive_historical_order() -> None:
    result = stratified_split([0] * 10 + [1] * 10)

    assert result == {
        "train": [3, 19, 4, 0, 9, 13, 5, 17, 14, 16, 7, 12, 1, 18],
        "val": [15, 2],
        "test": [8, 10, 6, 11],
    }


def test_prepare_skips_short_and_unselected_clips_and_records_provenance(
    tmp_path: Path,
) -> None:
    archive = _write_archive(
        tmp_path / "speech.tar.gz",
        [
            ("dataset/yes/first.wav", _valid_clip(1)),
            ("dataset/yes/second.wav", _valid_clip(2)),
            ("dataset/dog/ignored.wav", b"not a wav"),
            ("dataset/no/short.wav", _valid_clip(3, frames=15_999)),
            ("dataset/no/last.wav", _valid_clip(4)),
            ("dataset/no/other.wav", _valid_clip(5)),
        ],
    )

    manifest_path = prepare_sc10(
        archive, tmp_path / "prepared", require_expected_counts=False
    )

    manifest = _manifest(manifest_path)
    assert manifest_path == tmp_path / "prepared/manifest.json"
    assert manifest["format"] == "rqf_ssl.sc10.v1"
    assert manifest["dataset"] == "speech_commands_v0.02_sc10"
    assert manifest["canonical"] is False
    assert manifest["classes"] == CLASSES
    assert manifest["sample_rate"] == 16_000
    assert manifest["sample_shape"] == [16_000, 1]
    assert manifest["dtype"] == "float32"
    assert manifest["archive"] == {
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()
    }
    assert manifest["counts"] == {"train": 2, "val": 0, "test": 2, "total": 4}
    sources = [
        row["source"]
        for split in ("train", "val", "test")
        for row in manifest["splits"][split]
    ]
    assert set(sources) == {
        "dataset/yes/first.wav",
        "dataset/yes/second.wav",
        "dataset/no/last.wav",
        "dataset/no/other.wav",
    }


def test_prepare_uses_archive_order_for_split_membership_and_train_only_stats(
    tmp_path: Path,
) -> None:
    members = []
    for index in range(10):
        members.extend(
            [
                (f"dataset/yes/y{9 - index}.wav", _valid_clip(index + 1)),
                (f"dataset/no/n{9 - index}.wav", _valid_clip(101 + index)),
            ]
        )
    archive = _write_archive(tmp_path / "speech.tar.gz", members)

    manifest_path = prepare_sc10(
        archive, tmp_path / "prepared", require_expected_counts=False
    )
    manifest = _manifest(manifest_path)

    expected_sources = {
        "train": [
            "dataset/yes/y6.wav",
            "dataset/no/n0.wav",
            "dataset/yes/y5.wav",
            "dataset/yes/y9.wav",
            "dataset/yes/y0.wav",
            "dataset/no/n6.wav",
            "dataset/yes/y4.wav",
            "dataset/no/n2.wav",
            "dataset/no/n5.wav",
            "dataset/no/n3.wav",
            "dataset/yes/y2.wav",
            "dataset/no/n7.wav",
            "dataset/yes/y8.wav",
            "dataset/no/n1.wav",
        ],
        "val": ["dataset/no/n4.wav", "dataset/yes/y7.wav"],
        "test": [
            "dataset/yes/y1.wav",
            "dataset/no/n9.wav",
            "dataset/yes/y3.wav",
            "dataset/no/n8.wav",
        ],
    }
    for split, expected in expected_sources.items():
        assert [row["source"] for row in manifest["splits"][split]] == expected

    train_values = (
        torch.tensor(
            [4, 110, 5, 1, 10, 104, 6, 108, 105, 107, 8, 103, 2, 109],
            dtype=torch.float32,
        )
        / 32768.0
    )
    expected_mean = torch.stack([value.expand(16_000) for value in train_values]).mean()
    expected_std = torch.stack([value.expand(16_000) for value in train_values]).std(
        correction=1
    )
    normalization = manifest["normalization"]
    assert normalization["fit_split"] == "train"
    assert normalization["std_correction"] == 1
    assert normalization["std_floor"] == 1e-5
    assert normalization["mean"] == pytest.approx(expected_mean.item())
    assert normalization["std"] == pytest.approx(expected_std.item())

    validation_signal, validation_label = SC10Dataset(manifest_path, "val")[0]
    expected_normalized = (106 / 32768.0 - expected_mean.item()) / max(
        expected_std.item(), 1e-5
    )
    assert validation_signal.shape == (16_000, 1)
    assert validation_signal.dtype == torch.float32
    assert validation_signal[0, 0].item() == pytest.approx(expected_normalized)
    assert validation_label == 1
    assert type(validation_label) is int


@pytest.mark.parametrize(
    ("clip", "message"),
    [
        (_wav_bytes(np.zeros(16_000, dtype="<i2"), sample_rate=8_000), "sample rate"),
        (_wav_bytes(np.zeros(32_000, dtype="<i2"), channels=2), "mono"),
        (_wav_bytes(np.zeros(16_000, dtype=np.uint8), sample_width=1), "PCM16"),
        (b"not a wave file", "WAV"),
    ],
)
def test_prepare_rejects_malformed_selected_wav_geometry(
    tmp_path: Path, clip: bytes, message: str
) -> None:
    archive = _write_archive(
        tmp_path / "speech.tar.gz", [("dataset/yes/bad.wav", clip)]
    )

    with pytest.raises(ValueError, match=message):
        prepare_sc10(archive, tmp_path / "prepared", require_expected_counts=False)

    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("name", ["/dataset/yes/a.wav", "dataset/../yes/a.wav"])
def test_prepare_rejects_unsafe_archive_member_paths(tmp_path: Path, name: str) -> None:
    archive = _write_archive(tmp_path / "speech.tar.gz", [(name, _valid_clip())])

    with pytest.raises(ValueError, match="archive member path"):
        prepare_sc10(archive, tmp_path / "prepared", require_expected_counts=False)


def test_prepare_rejects_archive_links(tmp_path: Path) -> None:
    link = tarfile.TarInfo("dataset/yes/link.wav")
    link.type = tarfile.SYMTYPE
    link.linkname = "dataset/yes/target.wav"
    archive = _write_archive(tmp_path / "speech.tar.gz", [(link.name, link)])

    with pytest.raises(ValueError, match="link"):
        prepare_sc10(archive, tmp_path / "prepared", require_expected_counts=False)


def test_prepare_rejects_duplicate_selected_member_names(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path / "speech.tar.gz",
        [
            ("dataset/yes/a.wav", _valid_clip(1)),
            ("dataset/yes/a.wav", _valid_clip(2)),
        ],
    )

    with pytest.raises(ValueError, match="duplicate"):
        prepare_sc10(archive, tmp_path / "prepared", require_expected_counts=False)


def test_prepare_enforces_expected_counts_by_default_without_publishing(
    tmp_path: Path,
) -> None:
    archive = _write_archive(
        tmp_path / "speech.tar.gz", [("dataset/yes/a.wav", _valid_clip())]
    )

    with pytest.raises(ValueError, match="expected SC10 counts"):
        prepare_sc10(archive, tmp_path / "prepared")

    assert not (tmp_path / "prepared").exists()


def test_prepare_rejects_existing_output_without_changing_it(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "speech.tar.gz", [])
    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError):
        prepare_sc10(archive, output, require_expected_counts=False)

    assert marker.read_text(encoding="utf-8") == "user data"


def test_prepare_treats_broken_output_symlink_as_a_collision(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path / "speech.tar.gz",
        [
            ("dataset/yes/one.wav", _valid_clip(1)),
            ("dataset/yes/two.wav", _valid_clip(2)),
        ],
    )
    output = tmp_path / "prepared"
    output.symlink_to(tmp_path / "missing", target_is_directory=True)

    with pytest.raises(FileExistsError):
        prepare_sc10(archive, output, require_expected_counts=False)

    assert output.is_symlink()


def test_prepare_does_not_replace_output_created_during_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _write_archive(
        tmp_path / "speech.tar.gz",
        [
            ("dataset/yes/one.wav", _valid_clip(1)),
            ("dataset/yes/two.wav", _valid_clip(2)),
        ],
    )
    output = tmp_path / "prepared"
    real_mkdir = os.mkdir
    competing_inode: int | None = None

    def racing_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal competing_inode
        if Path(path) == output and competing_inode is None:
            real_mkdir(path, mode)
            competing_inode = output.stat().st_ino
        if dir_fd is None:
            return real_mkdir(path, mode)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", racing_mkdir)

    with pytest.raises(FileExistsError):
        prepare_sc10(archive, output, require_expected_counts=False)

    assert competing_inode is not None
    assert output.stat().st_ino == competing_inode
    assert list(output.iterdir()) == []


@pytest.mark.parametrize("split", ["", "validation", "TRAIN"])
def test_dataset_rejects_invalid_requested_split(tmp_path: Path, split: str) -> None:
    manifest = _write_loader_fixture(tmp_path)

    with pytest.raises(ValueError, match="split"):
        SC10Dataset(manifest, split)


def test_dataset_rejects_duplicate_or_aliased_membership_before_payload_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _write_loader_fixture(tmp_path)
    manifest = _manifest(manifest_path)
    manifest["splits"]["test"][0]["path"] = "clips/../clips/train.npy"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("payload was read before membership validation")

    monkeypatch.setattr(np, "load", unexpected_load)
    with pytest.raises(ValueError, match="multiple splits|duplicate"):
        SC10Dataset(manifest_path, "train")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda manifest: manifest.update(sample_shape=[16_001, 1]), "shape"),
        (lambda manifest: manifest.update(dtype="float64"), "dtype"),
        (lambda manifest: manifest["splits"]["train"][0].update(label=10), "label"),
    ],
)
def test_dataset_rejects_invalid_declared_geometry_dtype_or_label(
    tmp_path: Path, mutate, message: str
) -> None:
    manifest_path = _write_loader_fixture(tmp_path)
    manifest = _manifest(manifest_path)
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        SC10Dataset(manifest_path, "train")


@pytest.mark.parametrize(
    "payload",
    [
        np.zeros((15_999, 1), dtype=np.float32),
        np.zeros((16_000,), dtype=np.float32),
        np.zeros((16_000, 1), dtype=np.float64),
        np.full((16_000, 1), np.nan, dtype=np.float32),
    ],
)
def test_dataset_rejects_invalid_payload(tmp_path: Path, payload: np.ndarray) -> None:
    manifest = _write_loader_fixture(tmp_path)
    np.save(tmp_path / "clips/train.npy", payload)

    dataset = SC10Dataset(manifest, "train")
    with pytest.raises(ValueError, match="waveform"):
        dataset[0]


def test_train_dataset_does_not_open_test_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_loader_fixture(tmp_path)
    real_load = np.load
    opened: list[str] = []

    def recording_load(path, *args, **kwargs):
        opened.append(Path(path).name)
        if Path(path).name == "test.npy":
            pytest.fail("test payload was opened while loading train data")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)
    dataset = SC10Dataset(manifest, "train")
    signal, label = dataset[0]

    assert opened == ["train.npy"]
    assert signal.shape == (16_000, 1)
    assert label == 0


def test_dataset_loads_without_object_pickle(tmp_path: Path) -> None:
    manifest = _write_loader_fixture(tmp_path)
    object_payload = np.empty((16_000, 1), dtype=object)
    object_payload.fill(0)
    np.save(tmp_path / "clips/train.npy", object_payload, allow_pickle=True)

    dataset = SC10Dataset(manifest, "train")
    with pytest.raises(ValueError, match="waveform"):
        dataset[0]


def test_dataset_rejects_non_array_numpy_payload(tmp_path: Path) -> None:
    manifest = _write_loader_fixture(tmp_path)
    with (tmp_path / "clips/train.npy").open("wb") as handle:
        np.savez(handle, signal=np.zeros((16_000, 1), dtype=np.float32))

    dataset = SC10Dataset(manifest, "train")
    with pytest.raises(ValueError, match="waveform"):
        dataset[0]
