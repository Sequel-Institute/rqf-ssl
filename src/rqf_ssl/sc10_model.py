"""SC10's deterministic windowed RQF representation and clip classifier."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn

from rqf_ssl._core.embeddings.rqf.implementations.pure_rqf import PureRQFEncoder


def sc10_config() -> dict[str, Any]:
    """Return the fixed SC10 architecture recipe, suitable for JSON overrides."""
    return {
        "width": 64,
        "layers": 4,
        "sample_rate": 16000,
        "sequence_length": 16000,
        "window_length": 400,
        "horizon": 4,
        "classes": 10,
        "readout_hidden": 256,
        "fbt_dtype": "bf16",
    }


def validate_sc10_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the public SC10 architecture surface before model allocation."""
    defaults = sc10_config()
    unknown = sorted(set(config).difference(defaults))
    if unknown:
        raise ValueError(f"unknown architecture option: {unknown[0]}")
    resolved = {**defaults, **config}
    for name in (
        "width",
        "layers",
        "sample_rate",
        "sequence_length",
        "window_length",
        "horizon",
        "classes",
        "readout_hidden",
    ):
        value = resolved[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"architecture option {name!r} must be a positive integer")
    if resolved["classes"] != 10:
        raise ValueError("SC10 requires exactly ten classes")
    if resolved["sequence_length"] % resolved["window_length"]:
        raise ValueError("sequence length must be divisible by window length")
    if resolved["sequence_length"] // resolved["window_length"] <= resolved["horizon"]:
        raise ValueError("prediction horizon must be smaller than the window count")
    if not isinstance(resolved["fbt_dtype"], str) or resolved["fbt_dtype"] not in {
        "bf16",
        "fp32",
    }:
        raise ValueError("fbt_dtype must be 'bf16' or 'fp32'")
    return resolved


def _encoder(config: Mapping[str, Any]) -> PureRQFEncoder:
    width, layers = int(config["width"]), int(config["layers"])
    sequence_length, sample_rate = (
        int(config["sequence_length"]),
        int(config["sample_rate"]),
    )
    return PureRQFEncoder(
        input_dims=1,
        hidden_dims=width,
        num_layers=layers,
        samp_frequency=sample_rate,
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
        seq_duration=sequence_length / sample_rate,
        filter_max_freq_factor=[
            max(0.5**layer, 2 / sequence_length * (1 + 1e-6)) for layer in range(layers)
        ],
        w_init="forward",
        w_init_filter_gain=0.5,
        fbt_native=True,
        fbt_dtype=config["fbt_dtype"],
        layer0_real_input=True,
    )


class SC10Encoder(nn.Module):
    """The mono full-BPTT temporal encoder used by both SC10 phases."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = validate_sc10_config(config)
        self.encoder = _encoder(self.config)

    def states(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1:] != (
            self.config["sequence_length"],
            1,
        ):
            raise ValueError("SC10 signal must have shape [batch, sequence_length, 1]")
        return self.encoder(signal)

    def clamp_parameters(self) -> None:
        """Project constrained temporal parameters after an optimizer step."""
        for module in self.encoder.modules():
            clamp = getattr(module, "clamp_parameters", None)
            if callable(clamp):
                clamp()


class SC10Representation(SC10Encoder):
    """SSL representation: encoder only, followed by fixed non-overlapping means."""

    def summarize(self, signal: torch.Tensor) -> torch.Tensor:
        states = self.states(signal)
        features = torch.cat([states.real, states.imag], dim=-1)
        batch, steps, width = features.shape
        windows = steps // self.config["window_length"]
        return features.reshape(
            batch, windows, self.config["window_length"], width
        ).mean(dim=2)

    def dense_features(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample states ``[B, T, 2w]`` and final-layer prescan power ``[B, T, w]``.

        The power is taken before the final activation, as in the paper.
        """
        with self.encoder.capture_prescan_power(self.config["layers"] - 1) as capture:
            states = self.states(signal)
        return torch.cat([states.real, states.imag], dim=-1), capture.power


class SC10Classifier(SC10Encoder):
    """Fresh per-step nonlinear readout whose logits are averaged at clip level."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__(config)
        # The downstream path consumes the encoder's native channel-first
        # layout.  On CUDA this also keeps the planar fused scan in its real
        # [real..., imag...] representation until the readout.
        self.encoder.output_channel_first = True
        self.encoder.output_planar = True
        feature_width = 2 * self.config["width"]
        self.readout = nn.Sequential(
            nn.Linear(feature_width, self.config["readout_hidden"]),
            nn.ReLU(),
            nn.Linear(self.config["readout_hidden"], self.config["classes"]),
        )
        hidden, _, output = self.readout
        assert isinstance(hidden, nn.Linear) and isinstance(output, nn.Linear)
        nn.init.normal_(hidden.weight, std=math.sqrt(2.0 / hidden.in_features))
        nn.init.zeros_(hidden.bias)
        nn.init.normal_(output.weight, std=1e-3)
        nn.init.zeros_(output.bias)

    def classify_states(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3 or states.shape[-1] != self.config["width"]:
            raise ValueError("SC10 states must have shape [batch, time, width]")
        return self._classify_channel_first(
            torch.cat([states.real, states.imag], dim=-1).permute(2, 0, 1)
        )

    def _classify_channel_first(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the pointwise MLP through the source channel-first GEMMs."""
        if features.ndim != 3 or features.shape[0] != 2 * self.config["width"]:
            raise ValueError("SC10 channel-first features have an invalid shape")
        hidden, _, output = self.readout
        assert isinstance(hidden, nn.Linear) and isinstance(output, nn.Linear)
        width, batch, steps = features.shape
        flat = features.reshape(width, batch * steps)
        # Explicit feature-by-column products retain the planar/CUDA handoff
        # order instead of materializing a time-last activation first.
        activation = torch.addmm(
            hidden.bias.to(flat.dtype)[:, None], hidden.weight.to(flat.dtype), flat
        ).relu()
        logits = torch.addmm(
            output.bias.to(activation.dtype)[:, None],
            output.weight.to(activation.dtype),
            activation,
        )
        return (
            logits.reshape(self.config["classes"], batch, steps)
            .float()
            .mean(dim=2)
            .transpose(0, 1)
        )

    def freeze_encoder(self) -> None:
        self.encoder.requires_grad_(False)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        states = self.states(signal)
        if torch.is_complex(states):
            features = torch.cat([states.real, states.imag], dim=0)
        else:
            features = states
        return self._classify_channel_first(features)
