import math
from typing import ClassVar

import torch
import torch.nn as nn

from rqf_ssl._core.embeddings.base import EmbeddingBase
from rqf_ssl._core.embeddings.rqf.filter import RQF
from rqf_ssl._core.embeddings.rqf.utils.activations import SplitReLU, SplitTanh
from rqf_ssl._core.embeddings.rqf.utils.divisive_normalization import (
    DivisiveNormalization,
)
from rqf_ssl._core.ops import scan as _scan_module
from rqf_ssl._core.ops.scan import (
    parallel_scan,
    sequential_scan,
)

try:
    from rqf_ssl._core.ops.triton_scan_kernel import (
        _triton_parallel_scan_multi_osc,
        _triton_sequential_scan_multi_osc,
    )

    _HAS_FUSED_MULTI_OSC = True
except ImportError:
    _HAS_FUSED_MULTI_OSC = False

try:
    from rqf_ssl._core.ops.chunked_scan_kernel import _triton_chunked_scan_prospective

    _HAS_CHUNKED_PROSPECTIVE = True
except ImportError:
    _HAS_CHUNKED_PROSPECTIVE = False

try:
    from rqf_ssl._core.ops.fbt_multiosc_kernel import fused_multiosc_scan

    _HAS_FBT_MULTIOSC = True
except ImportError:
    _HAS_FBT_MULTIOSC = False


def _halfwave_rectify_4ch(z: torch.Tensor) -> torch.Tensor:

    re, im = z.real, z.imag
    return torch.cat(
        [
            torch.relu(re),
            torch.relu(-re),
            torch.relu(im),
            torch.relu(-im),
        ],
        dim=-1,
    )


def _halfwave_rectify_2ch(x: torch.Tensor) -> torch.Tensor:

    return torch.cat([torch.relu(x), torch.relu(-x)], dim=-1)


def _normalize_prospective_tau(
    prospective_tau: float | list[float] | torch.Tensor | None,
    num_filters: int,
    per_filter_osc: list[int],
) -> float | torch.Tensor | None:

    if prospective_tau is None or isinstance(prospective_tau, int | float):
        return prospective_tau
    lookahead = torch.as_tensor(prospective_tau, dtype=torch.float32)
    if lookahead.ndim == 0:
        return lookahead
    if lookahead.shape[0] != num_filters:
        raise ValueError(
            f"Per-filter prospective_tau must have one entry per logical "
            f"filter (num_filters={num_filters}), got {lookahead.shape[0]}."
        )
    if sum(per_filter_osc) > num_filters:
        lookahead = lookahead.repeat_interleave(
            torch.tensor(per_filter_osc, dtype=torch.long)
        )
    return lookahead


def _resolve_recipe_tau(
    tau: float,
    dt: float,
    tau_dt_factor: float | None,
    seq_duration: float | None,
) -> float:

    if seq_duration is not None and tau_dt_factor is None:
        tau_dt_factor = 2.0
    if tau_dt_factor is not None:
        return tau_dt_factor * dt
    return tau


def _validate_readout_config(
    readout_mode: str,
    num_oscillators: int,
    c_projection: bool,
    activation: str = "gelu",
) -> None:

    import warnings

    valid = (
        "magnitude",
        "4channel",
        "2channel",
        "real",
        "divnorm",
        "4channel_divnorm",
        "2channel_divnorm",
        "complex",
    )
    if readout_mode not in valid:
        raise ValueError(f"readout_mode must be one of {valid}, got {readout_mode!r}")

    _complex_acts = {"split_tanh", "split_relu", "identity"}
    _real_acts = {"silu", "gelu", "relu", "identity"}
    if readout_mode == "complex" and activation not in _complex_acts:
        raise ValueError(
            f"readout_mode='complex' requires a complex-compatible activation "
            f"{_complex_acts}, got {activation!r}"
        )
    if readout_mode != "complex" and activation not in _real_acts:
        raise ValueError(
            f"readout_mode={readout_mode!r} (real output) requires a real activation "
            f"{_real_acts}, got {activation!r}"
        )

    if readout_mode == "real" and num_oscillators == 1 and not c_projection:
        warnings.warn(
            "readout_mode='real' with num_oscillators=1 and no c_projection "
            "will produce carrier oscillations. Consider c_projection=True "
            "to add a learned complex weight: Re(c*z).",
            stacklevel=3,
        )
    if c_projection and readout_mode == "magnitude" and num_oscillators == 1:
        warnings.warn(
            "c_projection=True with readout_mode='magnitude' has no effect "
            "for N=1: |c*z| = |c|*|z|. Use readout_mode='real' to benefit "
            "from the learned complex projection: Re(c*z).",
            stacklevel=3,
        )


class BaseRQFBlock(EmbeddingBase):
    _activations: ClassVar[dict[str, type[nn.Module]]] = {
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "split_tanh": SplitTanh,
        "split_relu": SplitReLU,
        "identity": nn.Identity,
    }

    def __init__(
        self,
        d_model: int,
        num_filters: int,
        samp_frequency: float,
        min_freq: float,
        max_freq: float,
        dropout: float,
        activation: str,
        use_parallel_scan: bool,
        bidirectional: bool,
        discretization: str,
        tau: float,
        learn_tau: bool,
        max_gamma: float,
        min_gamma: float,
        sigma_init: float,
        norm_type: str,
        filter_scales: list[dict] | None,
        readout_mode: str,
        num_oscillators: int,
        c_projection: bool,
        clamp_gamma: bool = False,
        d_model_out: int | None = None,
        prospective_input: bool = False,
        prospective_tau: float | list[float] | torch.Tensor | None = None,
        tau_dt_factor: float | None = None,
        seq_duration: float | None = None,
        filter_max_freq_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if prospective_input and discretization == "bilinear":
            raise ValueError(
                "prospective_input=True is not supported with bilinear discretization. "
                "Use 'euler' or 'zoh'."
            )
        self.prospective_input = prospective_input

        self.d_model_out = d_model_out if d_model_out is not None else d_model

        self.residual_proj: nn.Module | None = (
            nn.Linear(d_model, self.d_model_out, bias=False)
            if self.d_model_out != d_model
            else None
        )

        self.num_filters = num_filters
        self.bidirectional = bidirectional
        self.use_parallel_scan = use_parallel_scan
        self.norm_type = norm_type
        self.num_oscillators = num_oscillators
        self.readout_mode = readout_mode

        _validate_readout_config(
            readout_mode, num_oscillators, c_projection, activation
        )

        self.total_filters = num_filters * 2 if bidirectional else num_filters

        if readout_mode in ("4channel", "4channel_divnorm"):
            self.readout_multiplier = 4
        elif readout_mode in ("2channel", "2channel_divnorm"):
            self.readout_multiplier = 2
        else:
            self.readout_multiplier = 1
        self.scan_output_width = self.total_filters * self.readout_multiplier

        if activation not in self._activations:
            raise ValueError(
                f"Unknown activation: {activation!r}. Must be one of {set(self._activations)}"
            )
        self.act = self._activations[activation]()

        self._use_batch_norm = norm_type == "batch"
        self.norm = self._create_prenorm(norm_type, readout_mode, d_model)

        dt = 1.0 / samp_frequency

        tau = _resolve_recipe_tau(tau, dt, tau_dt_factor, seq_duration)

        rqf_params, group_specs = self._parse_filter_config(
            filter_scales,
            num_filters,
            num_oscillators,
            tau,
            min_freq,
            max_freq,
            min_gamma,
            max_gamma,
        )
        rqf_tau, rqf_min_freq, rqf_max_freq, rqf_min_gamma, rqf_max_gamma = rqf_params

        per_filter_osc: list[int] = []
        for count, n_osc in group_specs:
            per_filter_osc.extend([n_osc] * count)
        self.scan_filters = sum(per_filter_osc)
        self._group_specs = group_specs
        self._uniform_osc = len(set(per_filter_osc)) == 1

        has_multi_osc = self.scan_filters > num_filters
        if has_multi_osc:
            self.register_buffer(
                "_input_repeats",
                torch.tensor(per_filter_osc, dtype=torch.long),
            )

        self.prospective_tau = _normalize_prospective_tau(
            prospective_tau, num_filters, per_filter_osc
        )

        self.rqf_fwd = RQF(
            num_filters=self.scan_filters,
            dt=dt,
            tau=rqf_tau,
            learn_tau=learn_tau,
            min_freq=rqf_min_freq,
            max_freq=rqf_max_freq,
            min_gamma=rqf_min_gamma,
            max_gamma=rqf_max_gamma,
            discretization=discretization,
            clamp_gamma=clamp_gamma,
            seq_duration=seq_duration,
            filter_max_freq_factor=filter_max_freq_factor,
        )

        if bidirectional:
            self.rqf_bwd = RQF(
                num_filters=self.scan_filters,
                dt=dt,
                tau=rqf_tau,
                learn_tau=learn_tau,
                min_freq=rqf_min_freq,
                max_freq=rqf_max_freq,
                min_gamma=rqf_min_gamma,
                max_gamma=rqf_max_gamma,
                discretization=discretization,
                clamp_gamma=clamp_gamma,
                seq_duration=seq_duration,
                filter_max_freq_factor=filter_max_freq_factor,
            )

        self._has_c_weights = has_multi_osc or c_projection
        if self._has_c_weights:
            self.c_weights_fwd = self._make_c_weights(group_specs)
            if bidirectional:
                self.c_weights_bwd = self._make_c_weights(group_specs)

        self.divisive_norm: DivisiveNormalization | None = None
        if readout_mode in ("divnorm", "4channel_divnorm", "2channel_divnorm"):
            dn_channels = (
                self.num_filters * self.readout_multiplier
                if readout_mode in ("4channel_divnorm", "2channel_divnorm")
                else self.total_filters
            )
            self.divisive_norm = DivisiveNormalization(
                num_channels=dn_channels,
                sigma_init=sigma_init,
            )

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _make_c_weights(group_specs: list[tuple[int, int]]) -> nn.Parameter:

        parts = []
        for count, n_osc in group_specs:
            phase = torch.rand(count * n_osc) * 2 * math.pi
            c = torch.complex(torch.cos(phase), torch.sin(phase))
            if n_osc > 1:
                c = c / (n_osc**0.5)
            parts.append(c)
        return nn.Parameter(torch.view_as_real(torch.cat(parts)))

    @staticmethod
    def _parse_filter_config(
        filter_scales: list[dict] | None,
        num_filters: int,
        num_oscillators: int,
        tau: float,
        min_freq: float,
        max_freq: float,
        min_gamma: float,
        max_gamma: float,
    ) -> tuple[tuple, list[tuple[int, int]]]:

        if filter_scales is not None:
            rqf_tau: float | list[float] = []
            rqf_min_freq: float | list[float] = []
            rqf_max_freq: float | list[float] = []
            rqf_min_gamma: float | list[float] = []
            rqf_max_gamma: float | list[float] = []
            group_specs: list[tuple[int, int]] = []
            total_count = 0
            for group in filter_scales:
                n = group["count"]
                n_osc = group.get("num_oscillators", num_oscillators)
                rqf_tau.extend([group.get("tau", tau)] * n)
                rqf_min_freq.extend([group.get("min_freq", min_freq)] * n)
                rqf_max_freq.extend([group.get("max_freq", max_freq)] * n)
                rqf_min_gamma.extend([group.get("min_gamma", min_gamma)] * n)
                rqf_max_gamma.extend([group.get("max_gamma", max_gamma)] * n)
                group_specs.append((n, n_osc))
                total_count += n
            if total_count != num_filters:
                raise ValueError(
                    f"filter_scales counts sum to {total_count}, expected {num_filters}"
                )
        else:
            rqf_tau = tau
            rqf_min_freq = min_freq
            rqf_max_freq = max_freq
            rqf_min_gamma = min_gamma
            rqf_max_gamma = max_gamma
            group_specs = [(num_filters, num_oscillators)]

        per_filter_osc: list[int] = []
        for count, n_osc in group_specs:
            per_filter_osc.extend([n_osc] * count)
        if sum(per_filter_osc) > num_filters and isinstance(rqf_tau, list):
            rqf_tau = [v for v, n in zip(rqf_tau, per_filter_osc) for _ in range(n)]
            rqf_min_freq = [
                v for v, n in zip(rqf_min_freq, per_filter_osc) for _ in range(n)
            ]
            rqf_max_freq = [
                v for v, n in zip(rqf_max_freq, per_filter_osc) for _ in range(n)
            ]
            rqf_min_gamma = [
                v for v, n in zip(rqf_min_gamma, per_filter_osc) for _ in range(n)
            ]
            rqf_max_gamma = [
                v for v, n in zip(rqf_max_gamma, per_filter_osc) for _ in range(n)
            ]

        return (
            rqf_tau,
            rqf_min_freq,
            rqf_max_freq,
            rqf_min_gamma,
            rqf_max_gamma,
        ), group_specs

    @staticmethod
    def _create_prenorm(
        norm_type: str, readout_mode: str, d_model: int
    ) -> nn.Module | None:

        _divnorm_readouts = ("divnorm", "4channel_divnorm", "2channel_divnorm")
        if norm_type != "none" and readout_mode in _divnorm_readouts:
            import warnings

            warnings.warn(
                f"norm_type='{norm_type}' with readout_mode='{readout_mode}': "
                "pre-normalization is discouraged when using divisive normalization "
                "readout, as the signal is already normalized by the readout. "
                "Consider norm_type='none'.",
                stacklevel=3,
            )
        if norm_type == "none":
            return None
        if norm_type == "batch":
            return nn.BatchNorm1d(d_model)
        return nn.LayerNorm(d_model)

    def _prenorm(self, x: torch.Tensor) -> torch.Tensor:

        if self.norm is None:
            return x
        if self._use_batch_norm:
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)

    def _apply_readout(
        self, y_fwd: torch.Tensor, y_bwd: torch.Tensor | None = None
    ) -> torch.Tensor:

        if self.readout_mode == "complex":
            if y_bwd is not None:
                return torch.cat([y_fwd, y_bwd], dim=-1)
            return y_fwd
        if self.readout_mode == "divnorm":
            return self.divisive_norm(y_fwd, y_bwd)
        if self.readout_mode == "4channel_divnorm":
            y_fwd_norm = self.divisive_norm(_halfwave_rectify_4ch(y_fwd))
            if y_bwd is not None:
                y_bwd_norm = self.divisive_norm(_halfwave_rectify_4ch(y_bwd))
                return torch.cat([y_fwd_norm, y_bwd_norm], dim=-1)
            return y_fwd_norm
        if self.readout_mode == "2channel_divnorm":
            y_fwd_norm = self.divisive_norm(_halfwave_rectify_2ch(y_fwd.real))
            if y_bwd is not None:
                y_bwd_norm = self.divisive_norm(_halfwave_rectify_2ch(y_bwd.real))
                return torch.cat([y_fwd_norm, y_bwd_norm], dim=-1)
            return y_fwd_norm
        if self.readout_mode == "4channel":
            y_fwd_out = _halfwave_rectify_4ch(y_fwd).square()
            if y_bwd is not None:
                return torch.cat(
                    [y_fwd_out, _halfwave_rectify_4ch(y_bwd).square()], dim=-1
                )
            return y_fwd_out
        if self.readout_mode == "2channel":
            y_fwd_out = _halfwave_rectify_2ch(y_fwd.real).square()
            if y_bwd is not None:
                return torch.cat(
                    [y_fwd_out, _halfwave_rectify_2ch(y_bwd.real).square()], dim=-1
                )
            return y_fwd_out
        if self.readout_mode == "real":
            if y_bwd is not None:
                return torch.cat([y_fwd.real, y_bwd.real], dim=-1)
            return y_fwd.real

        if y_bwd is not None:
            return torch.cat([y_fwd.abs(), y_bwd.abs()], dim=-1)
        return y_fwd.abs()

    def _c_weighted_sum(
        self,
        y: torch.Tensor,
        c_weights: nn.Parameter,
    ) -> torch.Tensor:

        b, t = y.shape[:2]
        c = torch.view_as_complex(c_weights)

        if self._uniform_osc:
            n_osc = self._group_specs[0][1]
            return (
                y.view(b, t, self.num_filters, n_osc) * c.view(self.num_filters, n_osc)
            ).sum(dim=-1)
        else:
            offset = 0
            c_offset = 0
            parts = []
            for count, n_osc in self._group_specs:
                w = count * n_osc
                y_g = y[:, :, offset : offset + w]
                c_g = c[c_offset : c_offset + w]
                if n_osc > 1:
                    parts.append(
                        (y_g.view(b, t, count, n_osc) * c_g.view(count, n_osc)).sum(
                            dim=-1
                        )
                    )
                else:
                    parts.append(y_g * c_g)
                offset += w
                c_offset += w
            return torch.cat(parts, dim=-1)

    def _compute_coefficients(self, rqf: RQF) -> tuple[torch.Tensor, torch.Tensor]:

        C1, C2 = rqf.compute_coefficients()
        return C1.to(torch.cfloat), C2.to(torch.cfloat)

    @staticmethod
    def _apply_prospective_taps(
        u: torch.Tensor,
        tap0: torch.Tensor,
        tap1: torch.Tensor,
    ) -> torch.Tensor:

        u_delayed = torch.cat([torch.zeros_like(u[:, :1, :]), u[:, :-1, :]], dim=1)
        return tap0 * u + tap1 * u_delayed

    def _scan_prospective(
        self,
        x: torch.Tensor,
        rqf_module: torch.nn.Module,
        c_weights: nn.Parameter | None,
    ) -> torch.Tensor:

        C1, tap0, tap1 = rqf_module.compute_prospective_taps(
            prospective_tau=self.prospective_tau
        )
        C1 = C1.to(torch.cfloat)
        tap0 = tap0.to(torch.cfloat)
        tap1 = tap1.to(torch.cfloat)

        if self.scan_filters > self.num_filters:
            x = x.repeat_interleave(self._input_repeats, dim=-1)
        if not x.is_complex():
            x = x.to(torch.cfloat)

        if (
            _HAS_CHUNKED_PROSPECTIVE
            and _scan_module._HAS_TRITON
            and x.is_cuda
            and not self._has_c_weights
            and _scan_module._SCAN_BACKEND == "chunked"
        ):
            return _triton_chunked_scan_prospective(C1, tap0, tap1, x)

        drive = self._apply_prospective_taps(x, tap0, tap1)
        y = self._scan(drive, C1, torch.ones_like(C1))
        if self._has_c_weights:
            y = self._c_weighted_sum(y, c_weights)
        return y

    def _scan_direction(
        self,
        x: torch.Tensor,
        rqf_module: torch.nn.Module,
        c_weights: nn.Parameter | None,
        flip: bool = False,
    ) -> torch.Tensor:

        if self.prospective_input:
            if flip:
                x = x.flip(1)
            y = self._scan_prospective(x, rqf_module, c_weights)
            if flip:
                y = y.flip(1)
            return y

        C1, C2 = self._compute_coefficients(rqf_module)

        if (
            self.scan_filters > self.num_filters
            and self._uniform_osc
            and _HAS_FUSED_MULTI_OSC
            and _scan_module._HAS_TRITON
            and x.is_cuda
        ):
            return self._scan_multiosc(x, C1, C2, c_weights, flip)

        if flip:
            x = x.flip(1)
        if self.scan_filters > self.num_filters:
            x = x.repeat_interleave(self._input_repeats, dim=-1)
        y = self._scan(x, C1, C2)
        if self._has_c_weights:
            y = self._c_weighted_sum(y, c_weights)
        if flip:
            y = y.flip(1)
        return y

    def _scan_multiosc(
        self,
        x: torch.Tensor,
        C1: torch.Tensor,
        C2: torch.Tensor,
        c_weights: nn.Parameter | None,
        flip: bool,
    ) -> torch.Tensor:

        n_osc = self._group_specs[0][1]
        u = x if x.is_complex() else x.to(torch.cfloat)
        if _HAS_FBT_MULTIOSC and _scan_module._SCAN_BACKEND == "chunked":
            return fused_multiosc_scan(C1, C2, u, c_weights, n_osc, reversed_time=flip)
        if flip:
            u = u.flip(1)
        if self.use_parallel_scan:
            y = _triton_parallel_scan_multi_osc(C1, C2, u, c_weights, n_osc)
        else:
            y = _triton_sequential_scan_multi_osc(C1, C2, u, c_weights, n_osc)
        if flip:
            y = y.flip(1)
        return y

    def _scan(
        self, u: torch.Tensor, C1: torch.Tensor, C2: torch.Tensor
    ) -> torch.Tensor:

        scan_fn = parallel_scan if self.use_parallel_scan else sequential_scan
        return scan_fn(C1, C2, u)
