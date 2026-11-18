"""Per-session Euclidean alignment of multichannel signals (He and Wu, 2019).

Every trial of a recording session is multiplied by the inverse square root
of that session's mean spatial covariance, so the session's average
covariance becomes the identity. The matrix is then scaled by the square root
of the session's mean channel power, which keeps the input amplitude that the
downstream recipe was tuned on. Alignment is label-free and uses only the
signals of the session it is applied to.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


def euclidean_alignment_matrix(
    signals: Iterable[torch.Tensor], *, eps: float = 1e-8, preserve_power: bool = True
) -> torch.Tensor:
    """Return ``R^{-1/2}`` (times ``sqrt(trace(R) / C)``) for the mean covariance ``R``.

    Each signal has shape ``[channels, samples]``.
    """
    total: torch.Tensor | None = None
    count = 0
    for signal in signals:
        if signal.ndim != 2:
            raise ValueError("Euclidean alignment expects [channels, samples] signals")
        x = signal.to(torch.float64)
        covariance = x @ x.T / x.shape[1]
        total = covariance if total is None else total + covariance
        count += 1
    if total is None:
        raise ValueError("Euclidean alignment needs at least one signal per session")
    reference = total / count
    reference = 0.5 * (reference + reference.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(reference)
    floor = eps * float(eigenvalues.max().clamp_min(eps))
    inverse_sqrt = eigenvalues.clamp_min(floor).rsqrt()
    matrix = eigenvectors @ torch.diag(inverse_sqrt) @ eigenvectors.T
    if preserve_power:
        matrix = matrix * torch.sqrt(torch.trace(reference) / reference.shape[0])
    return matrix.to(torch.float32)


def align_sessions(
    signals: np.ndarray, sessions: Sequence[str]
) -> tuple[np.ndarray, Mapping[str, int]]:
    """Align ``[trials, channels, samples]`` signals session by session.

    Returns the aligned float32 signals and the trial count of each session.
    """
    if signals.ndim != 3 or len(signals) != len(sessions):
        raise ValueError("signals must be [trials, channels, samples] with session ids")
    members: dict[str, list[int]] = {}
    for index, session in enumerate(sessions):
        members.setdefault(session, []).append(index)
    aligned = np.empty(signals.shape, dtype=np.float32)
    for indices in members.values():
        trials = [
            torch.from_numpy(np.asarray(signals[i], dtype=np.float32))
            for i in indices
        ]
        matrix = euclidean_alignment_matrix(trials)
        for index, trial in zip(indices, trials, strict=True):
            aligned[index] = (matrix @ trial).numpy()
    return aligned, {session: len(indices) for session, indices in members.items()}
