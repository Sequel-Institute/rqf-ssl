import math
import os
import warnings
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

from rqf_ssl._core.embeddings.rqf.block import BaseRQFBlock
from rqf_ssl._core.embeddings.rqf.encoder import BaseRQFEncoder
from rqf_ssl._core.embeddings.rqf.utils.activations import SplitReLU
from rqf_ssl._core.embeddings.rqf.utils.init import (
    NOMINAL_FILTER_GAIN,
    init_complex_weight_,
    rqf_forward_std,
    rqf_glorot_std,
)
from rqf_ssl._core.ops import scan as _scan_module

try:
    from rqf_ssl._core.ops.fbt_planar_kernel import fbt_planar_scan
    from rqf_ssl._core.ops.fbt_scan_kernel import (
        fbt_layer0_real_scan,
        fbt_prospective_scan,
        fbt_recompute_scan,
    )

    _HAS_FBT_PROSPECTIVE = True
except ImportError:
    _HAS_FBT_PROSPECTIVE = False

_FBT_RECOMPUTE_MIN_GRID = 256


class PureRQFBlock(BaseRQFBlock):
    def __init__(
        self,
        d_model: int,
        complex_weights: bool = True,
        w_init: str | float | None = "forward",
        w_init_activation: str | None = None,
        w_init_filter_gain: float = NOMINAL_FILTER_GAIN,
        **kwargs: Any,
    ) -> None:
        d_out = kwargs.get("d_model_out") or d_model
        super().__init__(d_model=d_model, num_filters=d_out, **kwargs)

        if w_init is None:
            std = 1.0 / math.sqrt(d_model)
        else:
            in_activation = w_init_activation or "identity"
            if isinstance(w_init, int | float):
                std = float(w_init)
            elif w_init == "forward":
                std = rqf_forward_std(
                    d_model, in_activation, filter_gain=w_init_filter_gain
                )
            elif w_init == "glorot":
                std = rqf_glorot_std(
                    d_model, d_out, in_activation, filter_gain=w_init_filter_gain
                )
            else:
                raise ValueError(
                    f"w_init must be None, 'forward', 'glorot', or a float std; got {w_init!r}"
                )
            if not complex_weights:
                warnings.warn(
                    "the w_init variance rule is derived for a complex W; with "
                    "complex_weights=False the real W carries half the power "
                    "(use complex_weights=True for the recipe)",
                    stacklevel=2,
                )

        dtype = torch.complex64 if complex_weights else torch.float32
        self.W = nn.Parameter(torch.empty(d_out, d_model, dtype=dtype))
        init_complex_weight_(self.W, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        target_dtype = x.dtype if x.is_complex() else torch.complex64
        W_c = self.W.to(target_dtype)
        z = torch.matmul(x, W_c.t())

        y = self._scan_direction(z, self.rqf_fwd, None)

        return self.act(y)

    def _fbt_eligible(self) -> bool:

        return (
            self.prospective_input
            and not self.bidirectional
            and not self._has_c_weights
            and self.scan_filters == self.num_filters
        )

    def _forward_fbt(self, h: torch.Tensor) -> torch.Tensor:

        d_model, batch, time = h.shape
        W_c = self.W.to(torch.complex64)
        d_out = W_c.shape[0]
        z = torch.matmul(W_c, h.reshape(d_model, batch * time)).reshape(
            d_out, batch, time
        )

        C1, tap0, tap1 = self.rqf_fwd.compute_prospective_taps(
            prospective_tau=self.prospective_tau
        )
        fuse = isinstance(self.act, SplitReLU)
        if fuse and d_out * batch >= _FBT_RECOMPUTE_MIN_GRID:
            return fbt_recompute_scan(C1, tap0, tap1, z)
        y = fbt_prospective_scan(C1, tap0, tap1, z, fuse_act=fuse)

        return y if fuse else self.act(y)

    def _forward_fbt_planar(self, hp: torch.Tensor) -> torch.Tensor:

        two, d_model, batch, time = hp.shape
        Wr, Wi = self.W.real, self.W.imag
        w4 = torch.cat(
            [torch.cat([Wr, -Wi], dim=1), torch.cat([Wi, Wr], dim=1)], dim=0
        ).to(hp.dtype)
        z = w4 @ hp.reshape(2 * d_model, batch * time)
        zp = z.reshape(2, w4.shape[0] // 2, batch, time)
        C1, tap0, tap1 = self.rqf_fwd.compute_prospective_taps(
            prospective_tau=self.prospective_tau
        )
        return fbt_planar_scan(C1, tap0, tap1, zp)

    def _layer0_real_eligible(self) -> bool:

        return isinstance(self.act, SplitReLU) and self.W.is_complex()

    def _forward_fbt_layer0_real(self, h_real: torch.Tensor) -> torch.Tensor:

        d_model, batch, time = h_real.shape
        w_stacked = torch.cat([self.W.real, self.W.imag], dim=0).contiguous()
        C1, tap0, tap1 = self.rqf_fwd.compute_prospective_taps(
            prospective_tau=self.prospective_tau
        )
        return fbt_layer0_real_scan(
            w_stacked, h_real.reshape(d_model, batch * time), C1, tap0, tap1, batch
        )


class PrescanPowerCapture:
    def __init__(self) -> None:
        self._kind: str | None = None
        self._block = None
        self._stored = None

    def _store(self, kind: str, block, stored) -> None:
        self._kind = kind
        self._block = block
        self._stored = stored

    @property
    def power(self) -> torch.Tensor:

        if self._kind is None:
            raise RuntimeError(
                "no layer was captured: run the encoder forward inside the "
                "capture_prescan_power context first"
            )
        block, stored = self._block, self._stored
        if not block._fbt_eligible():
            raise NotImplementedError(
                "prescan power capture supports the prospective single-oscillator "
                "configuration (the fused FBT family)"
            )
        if self._kind == "planar_drive":
            z = torch.complex(stored[0].float(), stored[1].float())
        else:
            if self._kind == "planar_input":
                h_c = torch.complex(stored[0].float(), stored[1].float())
            elif self._kind == "real_input":
                h_c = stored.to(torch.complex64)
            elif self._kind == "btf_input":
                h_c = stored.permute(2, 0, 1)
            else:
                h_c = stored
            d_model, b, t = h_c.shape
            W_c = block.W.to(torch.complex64)
            z = (W_c @ h_c.reshape(d_model, b * t)).reshape(W_c.shape[0], b, t)
        C1, tap0, tap1 = block.rqf_fwd.compute_prospective_taps(
            prospective_tau=block.prospective_tau
        )
        if _HAS_FBT_PROSPECTIVE and z.is_cuda:
            y = fbt_prospective_scan(C1, tap0, tap1, z.contiguous(), fuse_act=False)
            return (y.real.square() + y.imag.square()).permute(1, 2, 0)
        # CPU or no-Triton path: the same two-tap drive and recurrence, unfused.
        z_delayed = torch.cat([torch.zeros_like(z[..., :1]), z[..., :-1]], dim=-1)
        drive = tap0[:, None, None] * z + tap1[:, None, None] * z_delayed
        y = _scan_module.sequential_scan(C1, torch.ones_like(C1), drive.permute(1, 2, 0))
        return y.real.square() + y.imag.square()


class PureRQFEncoder(BaseRQFEncoder):
    _block_class = PureRQFBlock

    def __init__(
        self,
        input_dims: int = 1,
        hidden_dims: int | list[int] = 32,
        num_layers: int = 5,
        samp_frequency: float = 10000.0,
        min_freq: float = 1.0,
        max_freq: float = 50.0,
        dropout: float = 0.0,
        activation: str = "split_relu",
        discretization: str = "zoh",
        tau: float = 0.001,
        learn_tau: bool = False,
        max_gamma: float = 0.85,
        min_gamma: float = 0.001,
        norm_type: str = "none",
        filter_scales: list[dict] | list[list[dict]] | None = None,
        clamp_gamma: bool = False,
        complex_weights: bool = True,
        skip_connection: bool = False,
        prospective_input: bool = False,
        prospective_tau: float | list[float] | torch.Tensor | None = None,
        tau_dt_factor: float | None = None,
        seq_duration: float | None = None,
        filter_max_freq_factor: float | list[float] = 1.0,
        w_init: str | float | None = "forward",
        w_init_filter_gain: float = NOMINAL_FILTER_GAIN,
        fbt_native: bool = True,
        fbt_dtype: str = "fp32",
        output_channel_first: bool = False,
        output_planar: bool = False,
        layer0_real_input: bool = True,
    ) -> None:
        if dropout > 0:
            warnings.warn(
                "pure RQF blocks do not apply dropout; the value is ignored",
                stacklevel=2,
            )
        self._block_kwargs = {
            "samp_frequency": samp_frequency,
            "min_freq": min_freq,
            "max_freq": max_freq,
            "dropout": dropout,
            "activation": activation,
            "use_parallel_scan": False,
            "bidirectional": False,
            "discretization": discretization,
            "tau": tau,
            "learn_tau": learn_tau,
            "max_gamma": max_gamma,
            "min_gamma": min_gamma,
            "sigma_init": 0.1,
            "norm_type": "none",
            "readout_mode": "complex",
            "num_oscillators": 1,
            "c_projection": False,
            "clamp_gamma": clamp_gamma,
            "complex_weights": complex_weights,
            "prospective_input": prospective_input,
            "prospective_tau": prospective_tau,
            "tau_dt_factor": tau_dt_factor,
            "seq_duration": seq_duration,
            "filter_max_freq_factor": filter_max_freq_factor,
            "w_init": w_init,
            "w_init_filter_gain": w_init_filter_gain,
        }

        super().__init__(
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            num_layers=num_layers,
            norm_type=norm_type,
            filter_scales=filter_scales,
        )
        self.skip_connection = skip_connection
        self._fbt_native = fbt_native

        if fbt_dtype not in ("fp32", "bf16"):
            raise ValueError(f"fbt_dtype must be 'fp32' or 'bf16'; got {fbt_dtype!r}")
        self._fbt_dtype = fbt_dtype

        self.output_channel_first = output_channel_first

        self.output_planar = output_planar

        self._prescan_capture: tuple | None = None

        self._layer0_real_input = layer0_real_input

    def _is_complex_output(self) -> bool:

        return True

    def _block_kwargs_for_layer(
        self, layer_idx: int, block_kwargs: dict[str, object]
    ) -> dict[str, object]:

        if self._block_kwargs.get("w_init") is not None:
            inter = self._block_kwargs.get("activation", "identity")
            block_kwargs["w_init_activation"] = "identity" if layer_idx == 0 else inter
        return block_kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._fbt_enabled(x):
            return self._forward_fbt(x)

        x = self.input_proj(x)
        h = x.to(torch.complex64)
        with torch.amp.autocast(x.device.type, enabled=False):
            for idx, block in enumerate(self.blocks):
                self._maybe_capture(idx, "btf_input", block, h)
                out = block(h)
                h = out + h if self.skip_connection and out.shape == h.shape else out

        return h.permute(2, 0, 1).contiguous() if self.output_channel_first else h

    def _fbt_enabled(self, x: torch.Tensor) -> bool:

        return (
            self._fbt_native
            and _HAS_FBT_PROSPECTIVE
            and _scan_module._HAS_TRITON
            and x.is_cuda
            and _scan_module._SCAN_BACKEND == "chunked"
            and os.environ.get("RQF_SSL_RQF_FBT", "1") != "0"
            and all(block._fbt_eligible() for block in self.blocks)
        )

    def _forward_fbt(self, x: torch.Tensor) -> torch.Tensor:

        planar_dt = self._planar_dtype()
        if planar_dt is not None:
            return self._forward_fbt_planar_pipeline(x, planar_dt)

        batch, time, in_dims = x.shape
        w_in = self.input_proj.weight
        b_in = self.input_proj.bias
        x_cf = x.permute(2, 0, 1).reshape(in_dims, batch * time)
        proj = w_in @ x_cf if b_in is None else torch.addmm(b_in[:, None], w_in, x_cf)
        proj_r = proj.reshape(w_in.shape[0], batch, time)
        use_layer0_real = (
            self._layer0_real_input and self.blocks[0]._layer0_real_eligible()
        )
        self._maybe_capture(0, "real_input", self.blocks[0], proj_r)
        with torch.amp.autocast(x.device.type, enabled=False):
            if use_layer0_real:
                h = self.blocks[0]._forward_fbt_layer0_real(proj_r)
                if self.skip_connection and h.shape == proj_r.shape:
                    h = h + proj_r
                remaining, start = self.blocks[1:], 1
            else:
                h = proj_r.to(torch.complex64)
                remaining, start = self.blocks, 0
            for idx, block in enumerate(remaining, start=start):
                if idx > 0:
                    self._maybe_capture(idx, "fbt_input", block, h)
                out = block._forward_fbt(h)
                h = out + h if self.skip_connection and out.shape == h.shape else out

            return h if self.output_channel_first else h.permute(1, 2, 0).contiguous()

    def _planar_dtype(self) -> torch.dtype | None:

        if os.environ.get("RQF_SSL_RQF_PLANAR", "1") == "0":
            return None
        if self.skip_connection or not all(
            isinstance(block.act, SplitReLU) for block in self.blocks
        ):
            return None
        name = os.environ.get("RQF_SSL_RQF_DTYPE", self._fbt_dtype)
        if name == "bf16":
            return torch.bfloat16
        if name == "fp32" and self.output_planar and self.output_channel_first:
            return torch.float32
        return None

    @contextmanager
    def capture_prescan_power(self, layer: int):

        if not 0 <= layer < len(self.blocks):
            raise ValueError(
                f"layer must be in [0, {len(self.blocks) - 1}], got {layer}"
            )
        holder = PrescanPowerCapture()
        self._prescan_capture = (layer, holder)
        try:
            yield holder
        finally:
            self._prescan_capture = None

    def _maybe_capture(self, idx: int, kind: str, block, stored) -> None:

        cap = self._prescan_capture
        if cap is not None and cap[0] == idx:
            cap[1]._store(kind, block, stored)

    def _forward_fbt_planar_pipeline(
        self, x: torch.Tensor, dt: torch.dtype
    ) -> torch.Tensor:

        batch, time, in_dims = x.shape
        w_in = self.input_proj.weight
        b_in = self.input_proj.bias
        x_cf = x.permute(2, 0, 1).reshape(in_dims, batch * time)
        blk0 = self.blocks[0]
        with torch.amp.autocast(x.device.type, enabled=False):
            w0 = torch.cat([blk0.W.real, blk0.W.imag], dim=0)
            w0_in = (w0 @ w_in).to(dt)
            xc = x_cf.to(dt)
            z0 = (
                w0_in @ xc
                if b_in is None
                else torch.addmm((w0 @ b_in).to(dt)[:, None], w0_in, xc)
            )
            d0 = blk0.W.shape[0]
            zp = z0.reshape(2, d0, batch, time)
            self._maybe_capture(0, "planar_drive", blk0, zp)
            C1, tap0, tap1 = blk0.rqf_fwd.compute_prospective_taps(
                prospective_tau=blk0.prospective_tau
            )
            h = fbt_planar_scan(C1, tap0, tap1, zp)
            for idx, block in enumerate(self.blocks[1:], start=1):
                self._maybe_capture(idx, "planar_input", block, h)
                h = block._forward_fbt_planar(h)
            if self.output_channel_first and self.output_planar:
                return h.reshape(2 * h.shape[1], batch, time)
            hc = torch.complex(h[0].float(), h[1].float())
            return hc if self.output_channel_first else hc.permute(1, 2, 0).contiguous()
