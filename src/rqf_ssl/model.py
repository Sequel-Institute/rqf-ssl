"""Shared-channel RQF encoder, window pooler and coordinate-aware readout."""

import torch
from torch import nn

from rqf_ssl._core.embeddings.rqf.implementations.pure_rqf import PureRQFEncoder


TEMPORAL_SUPPORTS = ("windowed", "dense")
MONTAGES = ("tcp16", "bcic22")


def default_config():
    """Architecture and prediction geometry for the TUAB windowed example.

    ``temporal_support="dense"`` removes the window pooler: the readout then
    consumes every timestep's real and imaginary state, and ``window_length``
    and ``horizon`` are unused.
    """
    return dict(
        width=64,
        layers=4,
        sample_rate=128.0,
        sequence_length=12800,
        window_length=128,
        horizon=1,
        classes=2,
        fbt_dtype="bf16",
        temporal_support="windowed",
        montage="tcp16",
    )


def parameter_count(module):
    """Count complex-valued parameters as two real scalars."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in module.parameters())


def bcic_coordinate_queries():
    """Integer-mm coordinates of the 22 BCI-IV-2a electrodes, shifted by 100.

    The order is Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3,
    CP1, CPz, CP2, CP4, P1, Pz, P2, POz.
    """
    return torch.tensor(
        [
            [100, 147, 153],
            [52, 118, 144],
            [73, 121, 164],
            [100, 122, 171],
            [128, 121, 163],
            [150, 119, 145],
            [36, 89, 123],
            [48, 91, 151],
            [71, 92, 172],
            [100, 93, 180],
            [130, 92, 171],
            [154, 91, 151],
            [167, 90, 123],
            [49, 62, 152],
            [72, 62, 173],
            [100, 62, 180],
            [131, 62, 173],
            [153, 63, 152],
            [77, 36, 160],
            [100, 35, 166],
            [126, 36, 161],
            [100, 18, 140],
        ],
        dtype=torch.long,
    )


def montage_queries(montage):
    """Embedding queries for a named electrode montage."""
    if montage == "tcp16":
        return coordinate_queries()
    if montage == "bcic22":
        return bcic_coordinate_queries()
    raise ValueError(f"montage must be one of {MONTAGES}")


def coordinate_queries():
    """Integer-mm midpoints of the TCP-16 electrode pairs, shifted by 100."""
    positions = {
        "Fp1": (-21.2, 66.9, 12.1),
        "Fp2": (28.4, 65.6, 11.9),
        "F7": (-54.8, 33.9, -3.5),
        "F3": (-35.5, 49.4, 32.4),
        "F4": (40.2, 47.6, 32.1),
        "F8": (60.1, 33.3, -2.1),
        "T3": (-70.2, -0.1, -10.2),
        "C3": (-52.2, 1.7, 57.2),
        "C4": (54.4, 0.2, 57.4),
        "T4": (71.9, -2.5, -8.4),
        "T5": (-61.5, -36.0, -1.2),
        "P3": (-39.5, -37.3, 56.7),
        "P4": (41.1, -38.0, 56.1),
        "T6": (62.7, -37.7, -0.4),
        "O1": (-25.8, -70.1, 10.6),
        "O2": (27.1, -71.2, 11.2),
    }
    pairs = [
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
    ]
    return torch.tensor(
        [
            [int((positions[a][k] + positions[b][k]) / 2 + 100) for k in range(3)]
            for a, b in pairs
        ],
        dtype=torch.long,
    )


class SpatialReadout(nn.Module):
    """Sixteen coordinate-derived bases plus channel and window statistics.

    With ``chunk_size`` the temporal mean and standard deviation accumulate
    over chunks of steps, so per-timestep dense tokens never materialize their
    projected values for the whole sequence at once.
    """

    def __init__(self, width, classes, chunk_size=None, coordinates=None):
        super().__init__()
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self.embeddings = nn.ModuleList(
            [nn.Embedding(200, 32, padding_idx=0) for _ in range(3)]
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight)
            with torch.no_grad():
                embedding.weight[0].zero_()
        if coordinates is None:
            coordinates = coordinate_queries()
        self.register_buffer("coordinates", coordinates.to(torch.long))
        self.basis = nn.Sequential(
            nn.LayerNorm(32), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 16)
        )
        self.values = nn.Linear(width, 16 * 32)
        features = 2 * (16 * 32 + 3 * width)
        self.classifier = nn.Sequential(
            nn.LayerNorm(features), nn.Linear(features, classes)
        )

    def _spatial_basis(self):
        position = torch.stack(
            [e(self.coordinates[:, i]) for i, e in enumerate(self.embeddings)]
        ).sum(0)
        basis = self.basis(position)
        basis = basis - basis.mean(0, keepdim=True)
        return basis / (basis.square().sum(0, keepdim=True).sqrt() + 1e-6)

    def forward(self, tokens):
        """Map [batch, windows, channels, width] summaries to class logits."""
        if tokens.ndim != 4 or tokens.shape[2] != len(self.coordinates):
            raise ValueError(
                f"The readout requires {len(self.coordinates)} ordered channels"
            )
        basis = self._spatial_basis()
        if self.chunk_size is not None:
            return self._chunked(tokens, basis)
        b, w, c, _ = tokens.shape
        values = self.values(tokens).reshape(b, w, c, 16, 32)
        latents = torch.einsum("cm,btcmd->btmd", basis, values).reshape(b, w, -1)
        features = torch.cat(
            [
                latents,
                tokens.mean(2),
                tokens.std(2, unbiased=False),
                tokens.max(2).values,
            ],
            -1,
        )
        return self.classifier(
            torch.cat([features.mean(1), features.std(1, unbiased=False)], -1)
        )

    def _chunked(self, tokens, basis):
        b, steps, c, _ = tokens.shape
        if not steps:
            raise ValueError("The readout received an empty sequence")
        total = square_total = None
        for start in range(0, steps, self.chunk_size):
            chunk = tokens[:, start : start + self.chunk_size]
            values = self.values(chunk).reshape(b, chunk.shape[1], c, 16, 32)
            latents = torch.einsum("cm,btcmd->btmd", basis, values).flatten(2)
            spread = chunk.float().var(2, unbiased=False).clamp_min(1e-10).sqrt()
            local = torch.cat(
                [latents, chunk.mean(2), spread.to(chunk.dtype), chunk.max(2).values],
                -1,
            )
            total = local.sum(1) if total is None else total + local.sum(1)
            square = local.square().sum(1)
            square_total = square if square_total is None else square_total + square
        mean = total / steps
        variance = square_total.float() / steps - mean.float().square()
        variance = variance.clamp_min(1e-10)
        return self.classifier(torch.cat([mean, variance.sqrt().to(mean.dtype)], -1))


class RQFSystem(nn.Module):
    """The temporal encoder is shared across channels; spatial fusion is downstream."""

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        width, layers = config["width"], config["layers"]
        self.window_length = config["window_length"]
        self.sequence_length = config["sequence_length"]
        if min(width, layers, self.window_length, self.sequence_length) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.sequence_length % self.window_length:
            raise ValueError("Sequence length must be divisible by window length")
        if config["sample_rate"] <= 0:
            raise ValueError("Sample rate must be positive")
        self.temporal_support = config.get("temporal_support", "windowed")
        if self.temporal_support not in TEMPORAL_SUPPORTS:
            raise ValueError(f"temporal_support must be one of {TEMPORAL_SUPPORTS}")
        coordinates = montage_queries(config.get("montage", "tcp16"))
        self.encoder = PureRQFEncoder(
            input_dims=1,
            hidden_dims=width,
            num_layers=layers,
            samp_frequency=config["sample_rate"],
            min_freq=1.0,
            max_freq=50.0,
            dropout=0.0,
            activation="split_relu",
            discretization="zoh",
            tau=0.001,
            learn_tau=False,
            max_gamma=0.85,
            min_gamma=0.001,
            norm_type="none",
            clamp_gamma=False,
            complex_weights=True,
            skip_connection=False,
            prospective_input=True,
            tau_dt_factor=None,
            seq_duration=self.sequence_length / config["sample_rate"],
            filter_max_freq_factor=[0.5**layer for layer in range(layers)],
            w_init="forward",
            w_init_filter_gain=0.5,
            fbt_native=True,
            fbt_dtype=config["fbt_dtype"],
            layer0_real_input=True,
        )
        if self.temporal_support == "windowed":
            self.pooler = nn.Sequential(
                nn.Linear(4 * width, width), nn.LayerNorm(width)
            )
            self.readout = SpatialReadout(
                width, config["classes"], coordinates=coordinates
            )
        else:
            self.pooler = None
            self.readout = SpatialReadout(
                2 * width, config["classes"], chunk_size=128, coordinates=coordinates
            )

    def _states(self, signal):
        """Real/imaginary states of each channel, encoded alone: [B*C, T, 2w]."""
        if signal.ndim != 3 or signal.shape[1] != self.sequence_length:
            raise ValueError("Expected [batch, configured sequence length, channels]")
        b, t, c = signal.shape
        states = self.encoder(signal.permute(0, 2, 1).reshape(b * c, t, 1))
        return torch.cat([states.real, states.imag], -1)

    def summarize(self, signal):
        """Window summaries, or dense per-timestep states: [B, steps, C, features]."""
        real = self._states(signal)
        b, t, c = signal.shape
        if self.pooler is None:
            return real.reshape(b, c, t, -1).permute(0, 2, 1, 3).contiguous()
        windows = real.reshape(b, c, t // self.window_length, self.window_length, -1)
        pooled = torch.cat([windows.mean(3), windows[:, :, :, -1]], -1)
        return self.pooler(pooled).permute(0, 2, 1, 3).contiguous()

    def dense_features(self, signal):
        """Flattened per-channel states and final-layer prescan power.

        Row ``b * C + c`` holds channel ``c`` of example ``b``: states
        ``[B*C, T, 2w]`` and power ``[B*C, T, w]``.
        """
        layer = self.config["layers"] - 1
        with self.encoder.capture_prescan_power(layer) as capture:
            real = self._states(signal)
        return real, capture.power

    def representation_modules(self):
        """The modules a source checkpoint transfers downstream."""
        return [self.encoder] if self.pooler is None else [self.encoder, self.pooler]

    def freeze_representation(self):
        """Frozen evaluation keeps the transferred representation fixed."""
        for module in self.representation_modules():
            module.requires_grad_(False)

    def forward(self, signal):
        return self.readout(self.summarize(signal))
