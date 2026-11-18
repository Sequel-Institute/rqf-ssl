from __future__ import annotations

import math
import warnings

import torch

NOMINAL_FILTER_GAIN: float = 0.5

NEAR_ZERO_STD: float = 1e-3

GELU_GAIN: float = math.sqrt(1.0 / (1.0 / 3.0 + 1.0 / (2.0 * math.sqrt(3.0) * math.pi)))

RQF_ACTIVATION_GAINS: dict[str, tuple[float, float]] = {
    "identity": (1.0, 1.0),
    "split_tanh": (1.0, 1.0),
    "split_relu": (0.5, 0.5),
    "real_part": (0.5, 0.5),
    "real_relu": (0.25, 0.25),
    "abs": (1.0, 0.5),
}

REAL_MLP_GAINS: dict[str, float] = {
    "relu": math.sqrt(2.0),
    "gelu": GELU_GAIN,
    "tanh": 5.0 / 3.0,
    "sigmoid": 1.0,
    "identity": 1.0,
}


def rqf_activation_gains(name: str) -> tuple[float, float]:

    if name in {"tanh", "complex_tanh"}:
        raise ValueError(
            "complex tanh has no finite Gaussian second moment; use 'split_tanh' "
            "or an explicit float w_init_std"
        )
    if name == "split_sigmoid":
        warnings.warn(
            "split_sigmoid output power is mean-dominated; no power-preserving "
            "weight scale exists, falling back to identity gains (1, 1)",
            stacklevel=2,
        )
        return (1.0, 1.0)
    if name not in RQF_ACTIVATION_GAINS:
        raise ValueError(
            f"unknown activation {name!r} for the W init criterion; expected one "
            f"of {sorted(RQF_ACTIVATION_GAINS)} (or 'split_sigmoid')"
        )
    return RQF_ACTIVATION_GAINS[name]


def rqf_forward_std(
    n_in: int,
    in_activation: str = "identity",
    *,
    filter_gain: float = NOMINAL_FILTER_GAIN,
) -> float:

    g_rho, _ = rqf_activation_gains(in_activation)
    return 1.0 / math.sqrt(2.0 * n_in * filter_gain * g_rho)


def rqf_glorot_std(
    n_in: int,
    n_out: int,
    in_activation: str = "identity",
    *,
    filter_gain: float = NOMINAL_FILTER_GAIN,
) -> float:

    g_rho, chi_rho = rqf_activation_gains(in_activation)
    return math.sqrt(1.0 / (filter_gain * (n_in * g_rho + n_out * chi_rho)))


def bare_complex_std(fan_in: int) -> float:

    return 1.0 / math.sqrt(2.0 * fan_in)


def theta_min_from_sequence(
    tau: float,
    dt: float,
    seq_length: int,
    *,
    coverage_fraction: float = 1.0,
    settle_fraction: float = 0.73,
    seq_length_counts_samples: bool = True,
) -> float:

    if not 0.0 < settle_fraction < 1.0:
        raise ValueError(f"settle_fraction must be in (0, 1), got {settle_fraction}")
    if not 0.0 < coverage_fraction <= 1.0:
        raise ValueError(
            f"coverage_fraction must be in (0, 1], got {coverage_fraction}"
        )
    n_step = seq_length - 1 if seq_length_counts_samples else seq_length
    if n_step <= 0:
        raise ValueError(f"sequence length {seq_length} leaves no update intervals")
    t_cover = coverage_fraction * n_step * dt
    gamma_min = -(tau / t_cover) * math.log(1.0 - settle_fraction)
    if gamma_min >= 1.0:
        raise ValueError(
            f"requested settling coverage needs gamma_min = {gamma_min:.3f} >= 1, "
            "impossible under gamma = theta / (1 + theta); reduce settle_fraction, "
            "increase coverage_fraction, or use a longer sequence"
        )
    return gamma_min / (1.0 - gamma_min)


@torch.no_grad()
def init_complex_weight_(
    weight: torch.Tensor,
    std: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:

    if weight.is_complex():
        real = weight.real
        re = torch.randn(
            real.shape, generator=generator, dtype=real.dtype, device=real.device
        )
        im = torch.randn(
            real.shape, generator=generator, dtype=real.dtype, device=real.device
        )
        weight.copy_(torch.complex(re * std, im * std))
    else:
        weight.normal_(0.0, std, generator=generator)
    return weight
