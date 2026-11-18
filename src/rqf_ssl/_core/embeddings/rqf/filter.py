import math
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

DiscretizationType = Literal["euler", "zoh", "bilinear"]


def _expm1_div_x_complex(x: torch.Tensor, threshold: float = 1e-4) -> torch.Tensor:

    x_abs = torch.abs(x)
    small_mask = x_abs < threshold

    exp_x = torch.exp(x)

    x_safe = torch.where(small_mask, torch.ones_like(x), x)
    direct = (exp_x - 1.0) / x_safe

    taylor = 1.0 + x / 2 + x**2 / 6 + x**3 / 24 + x**4 / 120

    return torch.where(small_mask, taylor, direct)


class RQF(nn.Module):
    @staticmethod
    def _to_buffer(val: float | Sequence[float], size: int) -> torch.Tensor:

        if isinstance(val, Sequence):
            t = torch.tensor(val, dtype=torch.float32)
            if t.shape[0] != size:
                raise ValueError(
                    f"Expected sequence of length {size}, got {t.shape[0]}"
                )
            return t
        return torch.full((size,), val, dtype=torch.float32)

    @staticmethod
    def _validate_init_args(
        tau_tensor: torch.Tensor,
        freq: float | None,
        seq_duration: float | None,
        filter_max_freq_factor: float,
        num_filters: int,
        min_freq: float | Sequence[float],
        max_freq: float | Sequence[float],
        min_gamma: float | Sequence[float],
        max_gamma: float | Sequence[float],
    ) -> None:

        if (tau_tensor <= 0).any():
            raise ValueError("tau must be positive")
        if freq is not None and freq <= 0:
            raise ValueError(f"freq must be positive, got {freq}")
        if seq_duration is not None:
            if seq_duration <= 0:
                raise ValueError(f"seq_duration must be positive, got {seq_duration}")

            floor_factor = (tau_tensor / seq_duration).max().item()
            if not floor_factor <= filter_max_freq_factor <= 1.0:
                raise ValueError(
                    "filter_max_freq_factor must lie in [max(tau)/seq_duration, 1] = "
                    f"[{floor_factor:.3g}, 1], got {filter_max_freq_factor}"
                )
            return
        min_freq_t = RQF._to_buffer(min_freq, num_filters)
        max_freq_t = RQF._to_buffer(max_freq, num_filters)
        min_gamma_t = RQF._to_buffer(min_gamma, num_filters)
        max_gamma_t = RQF._to_buffer(max_gamma, num_filters)
        if (min_freq_t <= 0).any() or (min_freq_t > max_freq_t).any():
            raise ValueError(
                f"require 0 < min_freq <= max_freq, got {min_freq} and {max_freq}"
            )
        if (min_gamma_t <= 0).any() or (min_gamma_t > max_gamma_t).any():
            raise ValueError(
                f"require 0 < min_gamma <= max_gamma, got {min_gamma} and {max_gamma}"
            )

    def __init__(
        self,
        num_filters: int = 1,
        tau: float | Sequence[float] | None = None,
        dt: float = 0.0001,
        learn_tau: bool = False,
        learn_freq: bool = True,
        learn_gamma: bool = True,
        gamma: float | None = None,
        freq: float | None = None,
        min_gamma: float | Sequence[float] = 0.01,
        max_gamma: float | Sequence[float] = 0.2,
        min_freq: float | Sequence[float] = 1.0,
        max_freq: float | Sequence[float] = 50.0,
        seq_duration: float | None = None,
        filter_max_freq_factor: float = 1.0,
        discretization: DiscretizationType = "euler",
        use_b_matrix: bool = False,
        input_features: int = 1,
        clamp_gamma: bool = False,
    ) -> None:

        super().__init__()
        if discretization not in ("euler", "zoh", "bilinear"):
            raise ValueError(
                f"discretization must be 'euler', 'zoh', or 'bilinear', got {discretization!r}"
            )

        self.num_filters = num_filters
        self.dt = dt
        self.learn_tau = learn_tau
        self.learn_freq = learn_freq
        self.learn_gamma = learn_gamma
        self.discretization = discretization
        self.use_b_matrix = use_b_matrix
        self.input_features = input_features
        self.clamp_gamma = clamp_gamma

        if tau is None:
            tau = dt * 10
            warnings.warn(
                f"tau not specified, defaulting to dt*10 = {tau}",
                stacklevel=2,
            )

        tau_tensor = self._to_buffer(tau, num_filters)

        self._validate_init_args(
            tau_tensor,
            freq,
            seq_duration,
            filter_max_freq_factor,
            num_filters,
            min_freq,
            max_freq,
            min_gamma,
            max_gamma,
        )

        self.seq_duration = seq_duration
        self.filter_max_freq_factor = filter_max_freq_factor
        recipe = seq_duration is not None
        if recipe:
            min_freq_t = self._to_buffer(1.0 / seq_duration, num_filters)
            max_freq_t = 1.0 / tau_tensor
            min_gamma_t = self._to_buffer(1e-6, num_filters)
            max_gamma_t = self._to_buffer(1.0 - 1e-6, num_filters)
        else:
            min_freq_t = self._to_buffer(min_freq, num_filters)
            max_freq_t = self._to_buffer(max_freq, num_filters)
            min_gamma_t = self._to_buffer(min_gamma, num_filters)
            max_gamma_t = self._to_buffer(max_gamma, num_filters)

        self.register_buffer("_min_gamma", min_gamma_t)
        self.register_buffer("_max_gamma", max_gamma_t)
        self.register_buffer("_min_freq", min_freq_t)
        self.register_buffer("_max_freq", max_freq_t)

        self.log_tau = nn.Parameter(torch.log(tau_tensor), requires_grad=learn_tau)

        if freq is not None:
            log_theta_init = torch.log(
                2 * np.pi * torch.full((num_filters,), freq) * tau_tensor
            )
        elif recipe:
            log_theta_min = torch.log(2 * np.pi * tau_tensor / seq_duration)
            log_theta_max = torch.log(
                torch.full((num_filters,), 2 * np.pi * filter_max_freq_factor)
            )
            u = torch.rand(num_filters)
            log_theta_init = log_theta_min + u * (log_theta_max - log_theta_min)
        else:
            freq_init = self._min_freq + torch.rand(num_filters) * (
                self._max_freq - self._min_freq
            )
            log_theta_init = torch.log(2 * np.pi * freq_init * tau_tensor)

        self.log_theta = nn.Parameter(log_theta_init, requires_grad=learn_freq)

        if not recipe:
            with torch.no_grad():
                max_freq_tau = (self._max_freq * tau_tensor).max().item()
                if max_freq_tau >= 1.0:
                    warnings.warn(
                        f"freq * tau >= 1 (max_freq={self._max_freq.max().item():.4f}, "
                        f"max_tau={tau_tensor.max().item():.4f}, product={max_freq_tau:.4f}). "
                        f"Some filters may behave as low-pass. Consider reducing tau or max_freq.",
                        UserWarning,
                        stacklevel=2,
                    )

        if gamma is None:
            theta_init = torch.exp(log_theta_init)
            gamma_init_val = theta_init / (1.0 + theta_init)
        else:
            gamma_init_val = torch.full((num_filters,), gamma)

        gamma_init_clamped = torch.clamp(
            gamma_init_val, self._min_gamma, self._max_gamma
        )
        self.log_gamma = nn.Parameter(
            torch.log(gamma_init_clamped.clone().float()),
            requires_grad=learn_gamma,
        )

        if use_b_matrix:
            std = 1.0 / math.sqrt(input_features)
            self.B = nn.Parameter(torch.randn(num_filters, input_features) * std)
        else:
            self.B = None

    @property
    def tau(self) -> torch.Tensor:

        return torch.exp(self.log_tau)

    @property
    def theta(self) -> torch.Tensor:

        return torch.exp(self.log_theta)

    @property
    def gamma(self) -> torch.Tensor:

        return torch.exp(self.log_gamma)

    @property
    def omega(self) -> torch.Tensor:

        return self.theta / self.tau

    @property
    def freq(self) -> torch.Tensor:

        return self.theta / (2 * np.pi * self.tau)

    @torch.no_grad()
    def set_freq_(self, value: torch.Tensor | float) -> None:

        freq = torch.as_tensor(
            value, dtype=self.log_theta.dtype, device=self.log_theta.device
        )
        if (freq <= 0).any():
            raise ValueError(f"freq must be positive, got {value}")
        self.log_theta.copy_(torch.log(2 * np.pi * freq * self.tau))

    @torch.no_grad()
    def clamp_parameters(self) -> None:

        tau = self.tau
        self.log_theta.clamp_(
            min=torch.log(2 * np.pi * self._min_freq * tau),
            max=torch.log(2 * np.pi * self._max_freq * tau),
        )
        self.log_gamma.clamp_(
            min=torch.log(self._min_gamma), max=torch.log(self._max_gamma)
        )

    def _migration_tensor(
        self,
        state_dict: dict,
        prefix: str,
        name: str,
        fallback: torch.Tensor,
        like: torch.Tensor,
    ) -> torch.Tensor:

        tensor = state_dict.get(prefix + name, fallback)
        return tensor.to(device=like.device, dtype=like.dtype)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:

        legacy_freq = prefix + "freq_param"
        if legacy_freq in state_dict and (prefix + "log_theta") not in state_dict:
            logits = state_dict.pop(legacy_freq)
            min_freq = self._migration_tensor(
                state_dict, prefix, "_min_freq", self._min_freq, logits
            )
            max_freq = self._migration_tensor(
                state_dict, prefix, "_max_freq", self._max_freq, logits
            )
            log_tau = self._migration_tensor(
                state_dict, prefix, "log_tau", self.log_tau, logits
            )
            freq = min_freq + (max_freq - min_freq) * torch.sigmoid(logits)
            state_dict[prefix + "log_theta"] = torch.log(
                2 * np.pi * freq * torch.exp(log_tau)
            )
        legacy_theta = prefix + "theta"
        if legacy_theta in state_dict and (prefix + "log_theta") not in state_dict:
            theta = state_dict.pop(legacy_theta)
            state_dict[prefix + "log_theta"] = torch.log(
                theta.to(device=self.log_theta.device, dtype=self.log_theta.dtype)
            )
        legacy_gamma_param = prefix + "gamma_param"
        if (
            legacy_gamma_param in state_dict
            and (prefix + "log_gamma") not in state_dict
        ):
            logits = state_dict.pop(legacy_gamma_param)
            min_gamma = self._migration_tensor(
                state_dict, prefix, "_min_gamma", self._min_gamma, logits
            )
            max_gamma = self._migration_tensor(
                state_dict, prefix, "_max_gamma", self._max_gamma, logits
            )
            gamma = min_gamma + (max_gamma - min_gamma) * torch.sigmoid(logits)
            state_dict[prefix + "log_gamma"] = torch.log(gamma)
        legacy_gamma_linear = prefix + "gamma"
        if (
            legacy_gamma_linear in state_dict
            and (prefix + "log_gamma") not in state_dict
        ):
            gamma = state_dict.pop(legacy_gamma_linear)
            state_dict[prefix + "log_gamma"] = torch.log(
                gamma.to(device=self.log_gamma.device, dtype=self.log_gamma.dtype)
            )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _clamped_gamma(self) -> torch.Tensor:

        gamma = self.gamma
        if not self.clamp_gamma:
            return gamma
        omega_tau = 2 * np.pi * self.freq * self.tau
        gamma_min = 1 - 1 / torch.sqrt(1 + omega_tau**2)
        gamma_crit = omega_tau / (1 + omega_tau)
        return torch.clamp(gamma, min=gamma_min, max=gamma_crit)

    def compute_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:

        tau = self.tau
        gamma = self._clamped_gamma()
        freq = self.freq

        alpha = gamma / tau
        A = -alpha + 1j * 2 * np.pi * freq * (1.0 - gamma)
        B = alpha

        if self.discretization == "euler":
            dt_ratio = self.dt / tau
            w = 1.0 + 1j * 2 * np.pi * tau * freq
            C1 = 1.0 + dt_ratio * (-1.0 + (1.0 - gamma) * w)
            C2 = dt_ratio * gamma

        elif self.discretization == "zoh":
            A_dt = A * self.dt
            C1 = torch.exp(A_dt)
            C2 = self.dt * _expm1_div_x_complex(A_dt) * B

        else:
            A_dt_half = A * self.dt / 2
            denominator = 1.0 - A_dt_half
            C1 = (1.0 + A_dt_half) / denominator
            C2 = self.dt * B / denominator

        return C1, C2

    def compute_prospective_taps(
        self,
        prospective_tau: float | Sequence[float] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.discretization == "bilinear":
            raise NotImplementedError(
                "Two-tap prospective input is not defined for bilinear discretization. "
                "Use 'euler' or 'zoh'."
            )

        c1, c2 = self.compute_coefficients()
        tau = self.tau
        gamma = self._clamped_gamma()

        if prospective_tau is None:
            lookahead_ratio = torch.ones_like(gamma)
        else:
            lookahead = torch.as_tensor(
                prospective_tau, dtype=tau.dtype, device=tau.device
            )
            if lookahead.ndim != 0 and lookahead.shape != tau.shape:
                raise ValueError(
                    f"prospective_tau must be scalar or shape {tuple(tau.shape)}, "
                    f"got {tuple(lookahead.shape)}"
                )
            lookahead_ratio = lookahead / tau

        extra = gamma * lookahead_ratio
        if self.discretization != "euler":
            extra = extra * c1
        return c1, c2 + extra, -extra
