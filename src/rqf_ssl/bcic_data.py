"""BCI Competition IV dataset 2a: preprocessing and trial datasets.

All 5,184 four-class motor-imagery trials are kept. Subjects 1-5, 6-7 and 8-9
form the training, validation and test splits (2,880 / 1,152 / 1,152 trials).
Each trial uses the 22 EEG channels (EOG excluded), common-average reference,
a causal fifth-order Butterworth bandpass at 0.3-40 Hz applied to the six
seconds from trial onset, the four seconds from the cue, and Fourier
resampling from 250 to 200 Hz, in microvolts.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from rqf_ssl.alignment import align_sessions

CHANNELS = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
)  # fmt: skip
CLASSES = ("left_hand", "right_hand", "feet", "tongue")
SAMPLE_RATE = 200
SAMPLES = 800
# The models read microvolts divided by this factor, as in the paper's runs.
INPUT_SCALE = 100.0
_SOURCE_RATE = 250
_TRIAL_SOURCE_SAMPLES = 1500
_TRIALS_PER_SESSION = 288
_COLUMNS = ["subject", "session", "run", "trial", "label", "split"]
Split = Literal["train", "val", "test"]


def subject_split(subject: int) -> Split:
    if subject in range(1, 6):
        return "train"
    if subject in (6, 7):
        return "val"
    if subject in (8, 9):
        return "test"
    raise ValueError("BCI-IV-2a subjects are 1 through 9")


def transform_trial(source_microvolts: np.ndarray) -> np.ndarray:
    """Map a ``[22, >=1500]`` trial from its onset to ``[22, 800]`` float32."""
    from scipy import signal

    source = np.asarray(source_microvolts, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] != 22 or source.shape[1] < 1500:
        raise ValueError("a trial must be [22, >=1500] microvolts at 250 Hz")
    if not np.isfinite(source).all():
        raise ValueError("a trial contains non-finite values")
    referenced = source - source.mean(axis=0, keepdims=True)
    b, a = signal.butter(5, [0.3, 40.0], btype="bandpass", fs=_SOURCE_RATE)
    filtered = signal.lfilter(b, a, referenced, axis=-1)
    resampled = signal.resample(filtered[:, 500:1500], SAMPLES, axis=-1)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def run_assignments(record_onsets: list[int], trial_onsets: list[int]) -> list[int]:
    """Assign each of the 288 trials to one of the six 48-trial MI runs."""
    runs: list[int | None] = [None] * len(trial_onsets)
    run = 0
    for index, start in enumerate(record_onsets):
        stop = record_onsets[index + 1] if index + 1 < len(record_onsets) else None
        members = [
            i
            for i, onset in enumerate(trial_onsets)
            if onset >= start and (stop is None or onset < stop)
        ]
        if not members:
            continue
        if len(members) != 48:
            raise ValueError("a recording run does not hold 48 trials")
        for i in members:
            runs[i] = run
        run += 1
    if run != 6 or any(value is None for value in runs):
        raise ValueError("a session does not contain exactly six MI runs")
    return [int(value) for value in runs if value is not None]


def load_labels(path: Path) -> list[int]:
    from scipy.io import loadmat

    values = np.asarray(loadmat(path)["classlabel"])
    if values.shape != (_TRIALS_PER_SESSION, 1):
        raise ValueError(f"{path.name}: classlabel must have shape [288, 1]")
    labels = [int(value) - 1 for value in values[:, 0]]
    if any(label not in range(4) for label in labels):
        raise ValueError(f"{path.name}: labels must lie in 1..4")
    return labels


def read_session(
    gdf: Path, labels_path: Path, *, session: str
) -> tuple[np.ndarray, list[int], list[int]]:
    """Return ``[288, 22, 800]`` trials, their labels, and their run indices."""
    import mne

    raw = mne.io.read_raw_gdf(gdf, preload=False, verbose="ERROR")
    if float(raw.info["sfreq"]) != _SOURCE_RATE or len(raw.ch_names) != 25:
        raise ValueError(f"{gdf.name}: expected 25 channels at 250 Hz")
    events = [
        (round(float(onset) * _SOURCE_RATE), str(code))
        for onset, code in zip(raw.annotations.onset, raw.annotations.description)
    ]
    record_onsets = [onset for onset, code in events if code == "32766"]
    trial_onsets = [onset for onset, code in events if code == "768"]
    labels = load_labels(labels_path)
    if len(trial_onsets) != _TRIALS_PER_SESSION:
        raise ValueError(f"{gdf.name}: expected 288 trial onsets")
    cues = {
        onset: code
        for onset, code in events
        if code in {"769", "770", "771", "772", "783"}
    }
    for onset, label in zip(trial_onsets, labels, strict=True):
        expected = str(769 + label) if session == "T" else "783"
        if cues.get(onset + 500) != expected:
            raise ValueError(f"{gdf.name}: cue events disagree with the labels")
    runs = run_assignments(record_onsets, trial_onsets)
    trials = []
    for onset in trial_onsets:
        volts = raw.get_data(
            picks=list(range(22)), start=onset, stop=onset + _TRIAL_SOURCE_SAMPLES
        )
        trials.append(transform_trial(np.asarray(volts, dtype=np.float64) * 1e6))
    return np.stack(trials), labels, runs


def preprocess(
    gdf_dir: str | Path, labels_dir: str | Path, output_dir: str | Path
) -> Path:
    """Write ``signals.npy`` ``[5184, 22, 800]`` and ``trials.csv`` to a new directory.

    ``signals.npy`` holds every trial channel-first in microvolts.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    signals: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for subject in range(1, 10):
        for session in ("T", "E"):
            name = f"A{subject:02d}{session}"
            trials, labels, runs = read_session(
                Path(gdf_dir) / f"{name}.gdf",
                Path(labels_dir) / f"{name}.mat",
                session=session,
            )
            signals.append(trials)
            within_run = {run: 0 for run in range(6)}
            for label, run in zip(labels, runs, strict=True):
                rows.append(
                    {
                        "subject": subject,
                        "session": session,
                        "run": run,
                        "trial": within_run[run],
                        "label": label,
                        "split": subject_split(subject),
                    }
                )
                within_run[run] += 1
    output.mkdir(parents=True)
    np.save(output / "signals.npy", np.concatenate(signals), allow_pickle=False)
    with (output / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "format": "rqf_ssl.bcic_iv_2a.v1",
        "channels": list(CHANNELS),
        "classes": list(CLASSES),
        "sample_rate": SAMPLE_RATE,
        "samples": SAMPLES,
        "unit": "microvolt",
        "counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "val", "test")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


@dataclass(frozen=True)
class Trial:
    subject: int
    session: str
    run: int
    trial: int
    label: int
    split: str

    @property
    def session_id(self) -> str:
        return f"A{self.subject:02d}{self.session}"


def read_trials(root: str | Path) -> list[Trial]:
    with (Path(root) / "trials.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _COLUMNS:
            raise ValueError(f"trials.csv columns must be: {', '.join(_COLUMNS)}")
        return [
            Trial(
                int(row["subject"]),
                row["session"],
                int(row["run"]),
                int(row["trial"]),
                int(row["label"]),
                row["split"],
            )
            for row in reader
        ]


class BCICDataset:
    """Trials of one split as ``([800, 22] float32, label)`` pairs.

    Signals are returned in microvolts divided by ``INPUT_SCALE``.

    ``alignment="euclidean"`` aligns each session with statistics from all of
    its trials in this split. ``ssl=True`` selects the training trials used
    for self-supervised learning: every training-subject trial except the
    sixth run of each subject's second (``E``) session, which the paper held
    out to monitor pretraining.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split,
        *,
        alignment: str | None = None,
        ssl: bool = False,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError("split must be one of: train, val, test")
        if alignment not in (None, "euclidean"):
            raise ValueError("alignment must be None or 'euclidean'")
        if ssl and split != "train":
            raise ValueError("self-supervised trials come from the training split only")
        trials = read_trials(root)
        signals = np.load(Path(root) / "signals.npy", mmap_mode="r", allow_pickle=False)
        if signals.shape != (len(trials), len(CHANNELS), SAMPLES):
            raise ValueError("signals.npy disagrees with trials.csv")
        keep = [
            index
            for index, trial in enumerate(trials)
            if trial.split == split
            and not (ssl and trial.session == "E" and trial.run == 5)
        ]
        if not keep:
            raise ValueError(f"requested split {split!r} is empty")
        self.trials = [trials[index] for index in keep]
        selected = np.asarray(signals[keep], dtype=np.float32)
        if not np.isfinite(selected).all():
            raise ValueError("signals contain non-finite values")
        self.session_counts: dict[str, int] = {}
        if alignment == "euclidean":
            selected, self.session_counts = align_sessions(
                selected, [trial.session_id for trial in self.trials]
            )
        # Power-preserving alignment is linear in the signal, so scaling after
        # it equals aligning the scaled signals.
        self.signals = selected / np.float32(INPUT_SCALE)
        self.alignment = alignment

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        return np.ascontiguousarray(self.signals[index].T), self.trials[index].label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gdf_dir", type=Path, help="holds A01T.gdf ... A09E.gdf")
    parser.add_argument("labels_dir", type=Path, help="holds A01T.mat ... A09E.mat")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    print(preprocess(args.gdf_dir, args.labels_dir, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
