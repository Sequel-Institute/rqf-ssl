"""Dense future log-power prediction, scored by InfoNCE over sampled negatives.

At each of a fixed set of anchor timesteps, the real and imaginary encoder
state is the source. The target is the log of the mean pre-activation power of
the final RQF layer over the ``window`` samples that start ``offset`` samples
after the anchor, standardized per coordinate with statistics refit on a fixed
training subset before the first epoch and after every epoch.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Iterator

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


_ROW_CHUNK = 1024


def select_anchors(
    time_size: int, *, offset: int, window: int, count: int, seed: int
) -> torch.Tensor:
    """Draw ``count`` distinct sorted anchors whose future window fits the sequence."""
    if min(time_size, offset, window, count) <= 0 or seed < 0:
        raise ValueError("Invalid dense anchor specification")
    valid = time_size - offset - window + 1
    if valid <= 0:
        raise ValueError("Dense anchor support is empty")
    if count > valid:
        raise ValueError("anchor count exceeds the available support")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    drawn = torch.randperm(valid, generator=generator)[:count]
    return drawn.sort().values.to(torch.int64)


def window_log_power(
    power: torch.Tensor,
    anchors: torch.Tensor,
    *,
    shift: int,
    window: int,
    epsilon: float,
) -> torch.Tensor:
    """``log(epsilon + mean power)`` over ``window`` samples from ``anchor + shift``."""
    if power.ndim != 3:
        raise ValueError("power must have shape [sequences, time, filters]")
    starts = anchors.to(power.device) + shift
    prefix = torch.cat((torch.zeros_like(power[:, :1]), power.cumsum(dim=1)), dim=1)
    mean = (prefix[:, starts + window] - prefix[:, starts]) / window
    return torch.log(epsilon + mean)


def temporal_negatives(
    anchors: torch.Tensor,
    *,
    sequences: int,
    negatives: int,
    min_separation: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample any other row; mask rows of the same sequence that are too close."""
    positions = anchors.to(device)
    per_sequence = positions.numel()
    rows = sequences * per_sequence
    if rows < 2:
        raise ValueError("InfoNCE needs at least two prediction rows")
    offsets = torch.randint(1, rows, (rows, negatives), device=device)
    indices = (torch.arange(rows, device=device)[:, None] + offsets) % rows
    sequence = torch.arange(rows, device=device) // per_sequence
    position = positions.repeat(sequences)
    same = sequence[indices] == sequence[:, None]
    close = (position[indices] - position[:, None]).abs() < min_separation
    return indices, ~(same & close)


def cross_example_negatives(
    anchors: torch.Tensor,
    *,
    sequences: int,
    channels: int,
    negatives: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the same channel of a different example, at any anchor.

    Sequences are flattened example-major, so sequence ``s`` is channel
    ``s % channels`` of example ``s // channels``.
    """
    if channels <= 0 or sequences % channels:
        raise ValueError("sequences must be a whole number of multichannel examples")
    examples = sequences // channels
    if examples < 2:
        raise ValueError("cross-example negatives need at least two examples")
    per_sequence = int(anchors.numel())
    rows = sequences * per_sequence
    positive_sequence = torch.arange(rows, device=device) // per_sequence
    positive_example = positive_sequence // channels
    positive_channel = positive_sequence % channels
    negative_example = torch.randint(examples - 1, (rows, negatives), device=device)
    negative_example += (negative_example >= positive_example[:, None]).long()
    negative_anchor = torch.randint(per_sequence, (rows, negatives), device=device)
    negative_sequence = negative_example * channels + positive_channel[:, None]
    indices = negative_sequence * per_sequence + negative_anchor
    return indices, torch.ones_like(indices, dtype=torch.bool)


class _BoundedSampledDot(torch.autograd.Function):
    """Sampled query-key dot products, recomputing gathered keys in backward."""

    @staticmethod
    def forward(ctx: Any, queries, keys, indices):
        output = queries.new_empty((queries.shape[0], indices.shape[1]))
        for start in range(0, queries.shape[0], _ROW_CHUNK):
            stop = min(start + _ROW_CHUNK, queries.shape[0])
            output[start:stop] = torch.einsum(
                "nd,nkd->nk", queries[start:stop], keys[indices[start:stop]]
            )
        ctx.save_for_backward(queries, keys, indices)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output):
        queries, keys, indices = ctx.saved_tensors
        grad_queries = torch.zeros_like(queries) if ctx.needs_input_grad[0] else None
        grad_keys = torch.zeros_like(keys) if ctx.needs_input_grad[1] else None
        for start in range(0, queries.shape[0], _ROW_CHUNK):
            stop = min(start + _ROW_CHUNK, queries.shape[0])
            chunk = indices[start:stop]
            grad = grad_output[start:stop]
            if grad_queries is not None:
                grad_queries[start:stop] = torch.einsum("nk,nkd->nd", grad, keys[chunk])
            if grad_keys is not None:
                contribution = grad.unsqueeze(-1) * queries[start:stop].unsqueeze(1)
                grad_keys.index_add_(
                    0, chunk.reshape(-1), contribution.reshape(-1, queries.shape[1])
                )
        return grad_queries, grad_keys, None


def masked_infonce(
    queries: torch.Tensor,
    keys: torch.Tensor,
    indices: torch.Tensor,
    admitted: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean InfoNCE, true key first; masked negatives never enter the softmax."""
    positive = (queries * keys).sum(-1, keepdim=True)
    negative = _BoundedSampledDot.apply(queries, keys, indices)
    negative = negative.masked_fill(~admitted, float("-inf"))
    logits = torch.cat((positive, negative), dim=1) / temperature
    labels = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    top1 = ((logits > logits[:, :1]).sum(dim=1) == 0).float().mean()
    return F.cross_entropy(logits, labels), top1


class DensePrediction(nn.Module):
    """Predict standardized future log-power from the state at fixed anchors."""

    def __init__(
        self,
        width: int,
        *,
        sequence_length: int,
        offset: int,
        window: int,
        anchor_count: int = 256,
        anchor_seed: int = 1701,
        negatives: int = 256,
        temperature: float = 0.1,
        epsilon: float = 1e-6,
        min_separation: int,
        channels: int | None = None,
        calibration_examples: int = 256,
    ) -> None:
        super().__init__()
        if min(width, negatives, calibration_examples, min_separation) < 1:
            raise ValueError("Invalid dense configuration")
        if temperature <= 0 or epsilon <= 0:
            raise ValueError("temperature and epsilon must be positive")
        if channels is not None and channels < 1:
            raise ValueError("channels must be positive")
        self.width, self.offset, self.window = width, offset, window
        self.negatives, self.temperature, self.epsilon = negatives, temperature, epsilon
        self.min_separation, self.channels = min_separation, channels
        self.calibration_examples = calibration_examples
        self.register_buffer(
            "anchors",
            select_anchors(
                sequence_length,
                offset=offset,
                window=window,
                count=anchor_count,
                seed=anchor_seed,
            ),
        )
        features = 2 * width
        self.predictor = nn.Sequential(
            nn.Linear(features, features),
            nn.GELU(),
            nn.Linear(features, features),
            nn.GELU(),
            nn.Linear(features, width),
        )
        self.register_buffer("target_mean", torch.zeros(width))
        self.register_buffer("target_std", torch.ones(width))
        self.register_buffer("calibrated", torch.tensor(False))

    def pairs(
        self, features: torch.Tensor, power: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Anchor sources ``[S, A, 2w]`` and detached raw targets ``[S, A, w]``."""
        if features.ndim != 3 or features.shape[-1] != 2 * self.width:
            raise ValueError("features must have shape [sequences, time, 2 * width]")
        if power.shape != (*features.shape[:2], self.width):
            raise ValueError("power must have shape [sequences, time, width]")
        source = features[:, self.anchors.to(features.device)]
        target = window_log_power(
            power.float(),
            self.anchors,
            shift=self.offset,
            window=self.window,
            epsilon=self.epsilon,
        )
        return source, target.detach()

    @torch.no_grad()
    def calibrate(self, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Refit the target statistics on the first ``calibration_examples`` sequences.

        Sequences are counted after channel flattening, so a multichannel batch
        contributes one sequence per channel of each example.
        """
        rows: list[torch.Tensor] = []
        taken = 0
        for features, power in batches:
            _, target = self.pairs(features, power)
            take = min(self.calibration_examples - taken, target.shape[0])
            rows.append(target[:take].reshape(-1, self.width).double().cpu())
            taken += take
            if taken == self.calibration_examples:
                break
        if taken != self.calibration_examples:
            raise ValueError("calibration data did not provide the declared examples")
        stacked = torch.cat(rows)
        self.target_mean.copy_(stacked.mean(dim=0))
        self.target_std.copy_(stacked.std(dim=0, unbiased=False).clamp_min(1e-6))
        self.calibrated.fill_(True)

    def forward(
        self, features: torch.Tensor, power: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not bool(self.calibrated):
            raise RuntimeError("dense prediction must be calibrated before use")
        source, raw = self.pairs(features, power)
        target = ((raw - self.target_mean) / self.target_std).detach()
        prediction = self.predictor(source)
        sequences = source.shape[0]
        queries = F.normalize(prediction.reshape(-1, self.width), dim=-1)
        keys = F.normalize(target.reshape(-1, self.width), dim=-1)
        if self.channels is None:
            indices, admitted = temporal_negatives(
                self.anchors,
                sequences=sequences,
                negatives=self.negatives,
                min_separation=self.min_separation,
                device=features.device,
            )
        else:
            indices, admitted = cross_example_negatives(
                self.anchors,
                sequences=sequences,
                channels=self.channels,
                negatives=self.negatives,
                device=features.device,
            )
        loss, top1 = masked_infonce(queries, keys, indices, admitted, self.temperature)
        return loss, dict(nce=float(loss.detach()), top1=float(top1))


def calibration_loader(
    dataset: Any, *, examples: int, seed: int, batch_size: int
) -> DataLoader[Any]:
    """A fixed, seeded subset of the pretraining data, read in draw order."""
    if examples > len(dataset):
        raise ValueError("calibration exceeds the pretraining dataset")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:examples].tolist()
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)


@contextmanager
def preserved_rng() -> Iterator[None]:
    """Calibration must not consume the random stream that training draws from."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def dense_recipe(name: str) -> dict[str, int]:
    """Pretext geometry of the paper's dense configurations (Table 10, Appendix B.3)."""
    recipes = {
        # 100 ms horizon over a 25 ms target window at 16 kHz.
        "sc10": dict(offset=1600, window=400, min_separation=400),
        # 2 s horizon over a 1 s target window at 128 Hz, 16 TCP channels.
        "tuab": dict(offset=256, window=128, min_separation=128, channels=16),
    }
    if name not in recipes:
        raise ValueError(f"unknown dense recipe: {name!r}")
    return dict(
        anchor_count=256,
        anchor_seed=1701,
        negatives=256,
        calibration_examples=256,
        calibration_seed=1702,
        **recipes[name],
    )
