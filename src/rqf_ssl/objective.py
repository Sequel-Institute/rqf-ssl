"""Future-window InfoNCE with detached targets and a variance floor."""

import torch
from torch import nn
from torch.nn import functional as F

from rqf_ssl.dense import _BoundedSampledDot, cross_example_negatives

NEGATIVE_POLICIES = ("uniform", "same_channel")
# Above this many candidate entries, distinct negatives come from oversampled
# draws with duplicates discarded instead of an enumerated candidate pool.
_EXACT_POOL_ELEMENTS = 1 << 26


def same_channel_negatives(batch, times, channels, negatives, device):
    """Distinct negatives from the same channel of other examples, at any window.

    Rows flatten ``[batch, channel, time]``. Each row draws ``negatives`` rows
    without replacement from the ``(batch - 1) * times`` rows of its channel in
    the other examples.
    """
    pool = (batch - 1) * times
    if batch < 2:
        raise ValueError("Same-channel negatives need at least two examples")
    if negatives > pool:
        raise ValueError(
            f"negatives={negatives} exceed the same-channel pool of {pool} rows"
        )
    rows = batch * channels * times
    anchors = torch.arange(times, device=device)
    if rows * pool > _EXACT_POOL_ELEMENTS:
        draws = negatives + max(negatives // 2, 8)
        for _ in range(6):
            sampled, _ = cross_example_negatives(
                anchors,
                sequences=batch * channels,
                channels=channels,
                negatives=draws,
                device=device,
            )
            ordered = sampled.sort(dim=1).values
            duplicate = torch.zeros_like(ordered, dtype=torch.bool)
            duplicate[:, 1:] = ordered[:, 1:] == ordered[:, :-1]
            if bool(((~duplicate).sum(dim=1) >= negatives).all()):
                keys = torch.rand(rows, draws, device=device)
                keys = keys.masked_fill(duplicate, 2.0)
                pick = keys.topk(negatives, dim=1, largest=False).indices
                return ordered.gather(1, pick)
            draws *= 2
    row = torch.arange(rows, device=device)
    example = row // (channels * times)
    channel = (row // times) % channels
    others = torch.arange(batch - 1, device=device)[None, :].expand(rows, -1)
    others = others + (others >= example[:, None]).long()
    candidates = (
        (others[:, :, None] * channels + channel[:, None, None]) * times
        + anchors[None, None, :]
    ).reshape(rows, pool)
    pick = torch.rand(rows, pool, device=device).topk(negatives, dim=1).indices
    return candidates.gather(1, pick)


class WindowedPrediction(nn.Module):
    """Predict the window summary ``horizon`` windows ahead.

    ``negative_policy="uniform"`` samples other target rows of the batch with
    replacement; ``"same_channel"`` samples distinct rows of the same channel
    in other examples.
    """

    def __init__(
        self,
        width,
        horizon=1,
        negatives=256,
        temperature=0.1,
        variance_weight=0.04,
        negative_policy="uniform",
    ):
        super().__init__()
        if (
            min(width, horizon, negatives) < 1
            or temperature <= 0
            or variance_weight < 0
            or negative_policy not in NEGATIVE_POLICIES
        ):
            raise ValueError("Invalid contrastive configuration")
        self.negative_policy = negative_policy
        self.horizon, self.negatives = horizon, negatives
        self.temperature, self.variance_weight = temperature, variance_weight
        self.predictor = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )

    def forward(self, tokens):
        if tokens.ndim != 4 or tokens.shape[1] <= self.horizon:
            raise ValueError(
                "Need [batch, windows, channels, width] with a future window"
            )
        source = tokens[:, : -self.horizon]
        target = tokens[:, self.horizon :].detach()
        prediction = self.predictor(source)
        if self.negative_policy == "same_channel":
            batch, times, channels, width = prediction.shape
            predicted = F.normalize(
                prediction.permute(0, 2, 1, 3).reshape(-1, width), dim=-1
            )
            targets = F.normalize(target.permute(0, 2, 1, 3).reshape(-1, width), dim=-1)
            rows = len(predicted)
            indices = same_channel_negatives(
                batch, times, channels, self.negatives, tokens.device
            )
            positive = (predicted * targets).sum(-1, keepdim=True)
            negative = _BoundedSampledDot.apply(predicted, targets, indices)
        else:
            predicted = F.normalize(
                prediction.reshape(-1, prediction.shape[-1]), dim=-1
            )
            targets = F.normalize(target.reshape(-1, target.shape[-1]), dim=-1)
            rows = len(predicted)
            if rows < 2:
                raise ValueError("InfoNCE needs at least two prediction pairs")
            # Each target row is a positive for one source. Sample all other
            # target rows with replacement, with no channel restriction.
            offsets = torch.randint(
                1, rows, (rows, self.negatives), device=tokens.device
            )
            row_ids = torch.arange(rows, device=tokens.device)[:, None]
            indices = (row_ids + offsets) % rows
            positive = (predicted * targets).sum(-1, keepdim=True)
            negative = torch.einsum("nd,nkd->nk", predicted, targets[indices])
        logits = torch.cat([positive, negative], 1) / self.temperature
        nce = F.cross_entropy(
            logits, torch.zeros(rows, device=tokens.device, dtype=torch.long)
        )
        variance = tokens.new_zeros(())
        for value in (source, prediction):
            flat = value.reshape(-1, value.shape[-1]).float()
            std = (flat.var(0, unbiased=False) + 1e-4).sqrt()
            variance = variance + F.relu(1 - std).mean()
        loss = nce + self.variance_weight * variance
        return loss, dict(nce=float(nce.detach()), variance=float(variance.detach()))
