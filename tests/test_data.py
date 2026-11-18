from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from rqf_ssl.data import (
    SegmentDataset,
    derive_bipolar,
    main,
    patient_train_val_split,
    preprocess_manifest,
    required_referential_channels,
    robust_q95_scale,
    segment_bounds,
)


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "patient_id", "split", "label"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_segment_dataset_loads_manifest_relative_float32_signal(tmp_path: Path) -> None:
    expected = np.arange(64, dtype=np.float64).reshape(4, 16)
    np.save(tmp_path / "segment.npy", expected)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [{"path": "segment.npy", "patient_id": "p1", "split": "train", "label": 1}],
    )

    dataset = SegmentDataset(manifest, "train")
    signal, label = dataset[0]

    np.testing.assert_array_equal(signal, expected.astype(np.float32))
    assert signal.shape == (4, 16)
    assert signal.dtype == np.float32
    assert label == 1
    assert type(label) is int


@pytest.mark.parametrize("bad_split", ["validation", "", "TRAIN"])
def test_segment_dataset_rejects_invalid_requested_split(
    tmp_path: Path, bad_split: str
) -> None:
    manifest = _write_manifest(tmp_path / "manifest.csv", [])

    with pytest.raises(ValueError, match="split"):
        SegmentDataset(manifest, bad_split)  # type: ignore[arg-type]


def test_segment_dataset_rejects_empty_requested_split(tmp_path: Path) -> None:
    np.save(tmp_path / "segment.npy", np.zeros((4, 16), dtype=np.float32))
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [{"path": "segment.npy", "patient_id": "p1", "split": "val", "label": 0}],
    )

    with pytest.raises(ValueError, match="empty"):
        SegmentDataset(manifest, "train")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("split", "validation", "split"), ("label", "2", "label")],
)
def test_segment_dataset_rejects_invalid_manifest_values(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    np.save(tmp_path / "segment.npy", np.zeros((4, 16), dtype=np.float32))
    row = {"path": "segment.npy", "patient_id": "p1", "split": "train", "label": 0}
    row[field] = value
    manifest = _write_manifest(tmp_path / "manifest.csv", [row])

    with pytest.raises(ValueError, match=message):
        SegmentDataset(manifest, "train")


def test_segment_dataset_rejects_patient_split_overlap(tmp_path: Path) -> None:
    np.save(tmp_path / "one.npy", np.zeros((4, 16), dtype=np.float32))
    np.save(tmp_path / "two.npy", np.zeros((4, 16), dtype=np.float32))
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {"path": "one.npy", "patient_id": "p1", "split": "train", "label": 0},
            {"path": "two.npy", "patient_id": "p1", "split": "test", "label": 0},
        ],
    )

    with pytest.raises(ValueError, match="patient"):
        SegmentDataset(manifest, "train")


@pytest.mark.parametrize(
    ("second_patient", "second_split", "second_label"),
    [
        ("p2", "train", 0),
        ("p2", "test", 0),
        ("p1", "train", 1),
        ("p1", "train", 0),
    ],
)
def test_segment_dataset_rejects_reused_resolved_path_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    second_patient: str,
    second_split: str,
    second_label: int,
) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "segment.npy").write_bytes(b"payload must not be read")
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {"path": "segment.npy", "patient_id": "p1", "split": "train", "label": 0},
            {
                "path": "nested/../segment.npy",
                "patient_id": second_patient,
                "split": second_split,
                "label": second_label,
            },
        ],
    )

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("segment payload was read during manifest validation")

    monkeypatch.setattr(np, "load", unexpected_load)
    with pytest.raises(ValueError, match="segment.*multiple"):
        SegmentDataset(manifest, "train")


def test_segment_dataset_accepts_unique_paths_without_loading_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "train.npy").write_bytes(b"train payload")
    (tmp_path / "test.npy").write_bytes(b"test payload")
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [
            {"path": "train.npy", "patient_id": "p1", "split": "train", "label": 0},
            {"path": "test.npy", "patient_id": "p2", "split": "test", "label": 1},
        ],
    )

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("segment payload was read while constructing a dataset")

    monkeypatch.setattr(np, "load", unexpected_load)

    assert len(SegmentDataset(manifest, "train")) == 1


def test_segment_dataset_rejects_path_escape(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "prepared"
    manifest_dir.mkdir()
    np.save(tmp_path / "escaped.npy", np.zeros((4, 16), dtype=np.float32))
    manifest = _write_manifest(
        manifest_dir / "manifest.csv",
        [{"path": "../escaped.npy", "patient_id": "p1", "split": "train", "label": 0}],
    )

    with pytest.raises(ValueError, match="path"):
        SegmentDataset(manifest, "train")


@pytest.mark.parametrize(
    "signal",
    [
        np.zeros((4, 15), dtype=np.float32),
        np.zeros((1, 4, 16), dtype=np.float32),
        np.full((4, 16), np.nan, dtype=np.float32),
    ],
)
def test_segment_dataset_rejects_invalid_segment_arrays(
    tmp_path: Path, signal: np.ndarray
) -> None:
    np.save(tmp_path / "segment.npy", signal)
    manifest = _write_manifest(
        tmp_path / "manifest.csv",
        [{"path": "segment.npy", "patient_id": "p1", "split": "train", "label": 0}],
    )

    dataset = SegmentDataset(manifest, "train")
    with pytest.raises(ValueError, match="segment"):
        dataset[0]


def test_derive_bipolar_uses_tcp_16_channel_order() -> None:
    channels = required_referential_channels()
    assert channels == [
        "Fp1",
        "F7",
        "T3",
        "T5",
        "O1",
        "Fp2",
        "F8",
        "T4",
        "T6",
        "O2",
        "F3",
        "C3",
        "P3",
        "F4",
        "C4",
        "P4",
    ]
    referential = np.tile(np.arange(16, dtype=np.float32), (3, 1))

    bipolar = derive_bipolar(referential, channels)

    expected_row = np.array(
        [-1, -1, -1, -1, -1, -1, -1, -1, -10, -1, -1, 8, -8, -1, -1, 6],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(bipolar, np.tile(expected_row, (3, 1)))


def test_robust_q95_scale_normalizes_each_channel_independently() -> None:
    base = np.arange(1, 21, dtype=np.float64)
    segment = np.stack([base, base * 2], axis=1)

    scaled = robust_q95_scale(segment)

    expected = np.stack(
        [base / (19.05 + 1e-8), (base * 2) / (38.1 + 1e-8)], axis=1
    ).astype(np.float32)
    np.testing.assert_allclose(scaled, expected, rtol=1e-6, atol=0.0)
    assert scaled.dtype == np.float32


def test_segment_bounds_omits_incomplete_trailing_crop() -> None:
    assert segment_bounds(25_700) == ((0, 12_800), (12_800, 25_600))
    assert segment_bounds(12_799) == ()


def test_patient_train_val_split_is_deterministic_and_patient_disjoint() -> None:
    patients = ["p3", "p1", "p2", "p4", "p1"]

    train_one, val_one = patient_train_val_split(patients, val_fraction=0.25, seed=0)
    train_two, val_two = patient_train_val_split(patients, val_fraction=0.25, seed=0)

    assert (train_one, val_one) == (train_two, val_two)
    assert len(train_one) == 3
    assert len(val_one) == 1
    assert set(train_one).isdisjoint(val_one)
    assert set(train_one + val_one) == {"p1", "p2", "p3", "p4"}


def _write_synthetic_edf(path: Path, *, seconds: float, sample_rate: int = 256) -> None:
    mne = pytest.importorskip("mne")
    pytest.importorskip("edfio")
    canonical = required_referential_channels()
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    names = [f"EEG {aliases.get(name, name)}-REF" for name in reversed(canonical)]
    times = np.arange(round(seconds * sample_rate), dtype=np.float64) / sample_rate
    common = 2e-6 * np.sin(2 * np.pi * times)
    data_by_name = {name: common.copy() for name in canonical}
    data_by_name["Fp1"] += 20e-6 * np.sin(2 * np.pi * 2 * times)
    data = np.stack([data_by_name[name] for name in reversed(canonical)])
    raw = mne.io.RawArray(
        data,
        mne.create_info(names, sample_rate, ch_types="eeg"),
        verbose="ERROR",
    )
    raw.export(path, fmt="edf", overwrite=True, verbose="ERROR")


def test_preprocess_manifest_resamples_projects_normalizes_and_drops_short_edf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MNE_DONTWRITE_HOME", "true")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    _write_synthetic_edf(tmp_path / "long.edf", seconds=101.0)
    _write_synthetic_edf(tmp_path / "short.edf", seconds=50.0)
    source = _write_manifest(
        tmp_path / "edf.csv",
        [
            {"path": "long.edf", "patient_id": "p1", "split": "train", "label": 1},
            {"path": "short.edf", "patient_id": "p2", "split": "val", "label": 0},
        ],
    )
    output = tmp_path / "prepared"

    assert main([str(source), str(output)]) == 0

    output_manifest = output / "manifest.csv"
    assert output_manifest == output / "manifest.csv"
    with output_manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["patient_id"] == "p1"
    assert rows[0]["split"] == "train"
    assert rows[0]["label"] == "1"
    segment = np.load(output / rows[0]["path"], allow_pickle=False)
    assert segment.shape == (12_800, 16)
    assert segment.dtype == np.float32
    assert np.isfinite(segment).all()
    channel_rms = np.sqrt(np.mean(np.square(segment), axis=0))
    assert set(np.flatnonzero(channel_rms > 0.1)) == {0, 8}
    np.testing.assert_allclose(segment[:, 0], segment[:, 8], atol=2e-4)
    np.testing.assert_allclose(
        np.quantile(np.abs(segment[:, [0, 8]]), 0.95, axis=0),
        np.full(2, 0.9995),
        atol=3e-4,
    )


def test_preprocess_manifest_rejects_existing_output(tmp_path: Path) -> None:
    source = _write_manifest(tmp_path / "edf.csv", [])
    output = tmp_path / "prepared"
    output.mkdir()

    with pytest.raises(FileExistsError):
        preprocess_manifest(source, output)


def test_preprocess_manifest_rejects_patient_split_overlap(tmp_path: Path) -> None:
    (tmp_path / "one.edf").write_bytes(b"not read")
    (tmp_path / "two.edf").write_bytes(b"not read")
    source = _write_manifest(
        tmp_path / "edf.csv",
        [
            {"path": "one.edf", "patient_id": "p1", "split": "train", "label": 0},
            {"path": "two.edf", "patient_id": "p1", "split": "test", "label": 0},
        ],
    )

    with pytest.raises(ValueError, match="patient"):
        preprocess_manifest(source, tmp_path / "prepared")


@pytest.mark.parametrize(
    ("second_patient", "second_split", "second_label"),
    [
        ("p2", "train", 0),
        ("p2", "test", 0),
        ("p1", "train", 1),
        ("p1", "train", 0),
    ],
)
def test_preprocess_manifest_rejects_reused_resolved_edf_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    second_patient: str,
    second_split: str,
    second_label: int,
) -> None:
    from rqf_ssl import data as data_module

    (tmp_path / "nested").mkdir()
    (tmp_path / "recording.edf").write_bytes(b"payload must not be read")
    source = _write_manifest(
        tmp_path / "edf.csv",
        [
            {"path": "recording.edf", "patient_id": "p1", "split": "train", "label": 0},
            {
                "path": "nested/../recording.edf",
                "patient_id": second_patient,
                "split": second_split,
                "label": second_label,
            },
        ],
    )

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("EDF payload was read during manifest validation")

    monkeypatch.setattr(data_module, "_load_edf", unexpected_load)
    with pytest.raises(ValueError, match="EDF.*multiple"):
        preprocess_manifest(source, tmp_path / "prepared")


def test_preprocess_manifest_accepts_unique_edf_paths_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rqf_ssl import data as data_module

    (tmp_path / "one.edf").write_bytes(b"one")
    (tmp_path / "two.edf").write_bytes(b"two")
    source = _write_manifest(
        tmp_path / "edf.csv",
        [
            {"path": "one.edf", "patient_id": "p1", "split": "train", "label": 0},
            {"path": "two.edf", "patient_id": "p2", "split": "test", "label": 1},
        ],
    )
    loaded: list[Path] = []

    def empty_recording(path: Path) -> np.ndarray:
        loaded.append(path)
        return np.empty((0, 16), dtype=np.float32)

    monkeypatch.setattr(data_module, "_load_edf", empty_recording)

    output = preprocess_manifest(source, tmp_path / "prepared")

    assert output.is_file()
    assert loaded == [
        (tmp_path / "one.edf").resolve(),
        (tmp_path / "two.edf").resolve(),
    ]


def test_label_budgets_are_nested_stratified_and_keep_whole_patients() -> None:
    """Catches budgets that drop a class, split a patient, or reshuffle per budget."""
    from rqf_ssl.data import label_budget_indices

    patient_ids, labels = [], []
    for patient in range(300):
        label = int(patient % 3 == 0)
        for _ in range(1 + patient % 4):
            patient_ids.append(f"p{patient:03d}")
            labels.append(label)
    selected = {}
    for fraction in (0.01, 0.05, 0.10, 0.25):
        indices = label_budget_indices(patient_ids, labels, fraction=fraction, seed=0)
        patients = {patient_ids[index] for index in indices}
        assert indices == sorted(indices)
        assert indices == [i for i, p in enumerate(patient_ids) if p in patients]
        classes = [labels[patient_ids.index(patient)] for patient in patients]
        per_class = [classes.count(0), classes.count(1)]
        assert per_class == [max(1, int(200 * fraction)), max(1, int(100 * fraction))]
        selected[fraction] = patients
    assert selected[0.01] <= selected[0.05] <= selected[0.10] <= selected[0.25]
    assert label_budget_indices(patient_ids, labels, fraction=0.05, seed=1) != (
        label_budget_indices(patient_ids, labels, fraction=0.05, seed=0)
    )
    with pytest.raises(ValueError, match="both labels"):
        label_budget_indices(["a", "a", "b"], [0, 1, 1], fraction=0.5)
