from __future__ import annotations

import json
import os
from pathlib import Path

import torch

_CACHE_ENV = "RQF_SSL_TUNING_CACHE"
_AUTOTUNE_ENV = "RQF_SSL_AUTOTUNE"
_CHUNKS = (256, 512, 1024, 2048)
_WARPS = (1, 2, 4)
_WIN_MARGIN = 0.95
_MIN_TIME = 4096

_TUNED_PLANAR: dict[tuple[str, str, int, int], tuple[int, int]] = {}
_CACHE_LOADED = False
_CALIBRATING = False


def _cache_path() -> Path:
    override = os.environ.get(_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    return Path("~/.cache/rqf_ssl._core/launch_tuning.json").expanduser()


_GPU_NAME: str | None = None


def _gpu_name() -> str:
    global _GPU_NAME
    if _GPU_NAME is None:
        _GPU_NAME = torch.cuda.get_device_name(torch.cuda.current_device())
    return _GPU_NAME


def _key(filters: int, batch: int, dtype: torch.dtype) -> tuple[str, str, int, int]:
    return (_gpu_name(), str(dtype).removeprefix("torch."), filters, batch)


def _load_cache() -> None:
    global _CACHE_LOADED
    _CACHE_LOADED = True
    path = _cache_path()
    if not path.exists():
        return
    try:
        blob = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    for key_str, val in blob.items():
        parts = key_str.split("|")
        if len(parts) != 4:
            continue
        gpu, dtype_str, filters, batch = parts
        _TUNED_PLANAR[(gpu, dtype_str, int(filters), int(batch))] = (
            int(val["chunk"]),
            int(val["warps"]),
        )


def _save_cache() -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "|".join(map(str, key)): {"chunk": chunk, "warps": warps}
            for key, (chunk, warps) in _TUNED_PLANAR.items()
        }
        path.write_text(json.dumps(blob, indent=1, sort_keys=True))
    except OSError:
        pass


def tuned_planar_launch(
    filters: int, batch: int, dtype: torch.dtype
) -> tuple[int, int] | None:

    if _CALIBRATING or not torch.cuda.is_available():
        return None
    if not _CACHE_LOADED:
        _load_cache()
    key = _key(filters, batch, dtype)
    tuned = _TUNED_PLANAR.get(key)
    if tuned is not None:
        return tuned
    if os.environ.get(_AUTOTUNE_ENV) == "1":
        return calibrate_planar(filters, batch, dtype=dtype)
    return None


def _time_config(
    zp: torch.Tensor, coeffs, chunk: int, warps: int, iters: int = 3
) -> float:

    from rqf_ssl._core.ops import fbt_planar_kernel

    C1, tap0, tap1 = coeffs
    fbt_planar_kernel._FORCED_LAUNCH = (chunk, warps)
    try:
        x = zp.detach().requires_grad_(True)
        out = fbt_planar_kernel.fbt_planar_scan(C1, tap0, tap1, x)
        grad = torch.ones_like(out)
        out.backward(grad)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            x.grad = None
            out = fbt_planar_kernel.fbt_planar_scan(C1, tap0, tap1, x)
            out.backward(grad)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters
    finally:
        fbt_planar_kernel._FORCED_LAUNCH = None


def calibrate_planar(
    filters: int,
    batch: int,
    *,
    time_size: int = 16000,
    dtype: torch.dtype = torch.bfloat16,
    force: bool = False,
) -> tuple[int, int] | None:

    global _CALIBRATING
    if not torch.cuda.is_available() or time_size < _MIN_TIME:
        return None
    if not _CACHE_LOADED:
        _load_cache()
    key = _key(filters, batch, dtype)
    if not force and key in _TUNED_PLANAR:
        return _TUNED_PLANAR[key]

    from rqf_ssl._core.ops.fbt_planar_kernel import _pick_planar_launch

    _CALIBRATING = True
    try:
        heuristic = _pick_planar_launch(filters, batch, dtype)
        gen = torch.Generator(device="cuda").manual_seed(0)
        device = torch.device("cuda")
        zp = torch.randn(2, filters, batch, time_size, generator=gen, device=device).to(
            dtype
        )
        radius = torch.rand(filters, generator=gen, device=device) * 0.099 + 0.9
        theta = torch.rand(filters, generator=gen, device=device) * torch.pi
        C1 = (radius * torch.cos(theta) + 1j * radius * torch.sin(theta)).to(
            torch.complex64
        )
        tap0 = torch.full((filters,), 0.5 + 0.1j, device=device, dtype=torch.complex64)
        tap1 = torch.full((filters,), 0.1 - 0.2j, device=device, dtype=torch.complex64)
        coeffs = (C1, tap0, tap1)

        base_ms = _time_config(zp, coeffs, *heuristic)
        best, best_ms = heuristic, base_ms
        for chunk in _CHUNKS:
            for warps in _WARPS:
                if (chunk, warps) == heuristic:
                    continue
                ms = _time_config(zp, coeffs, chunk, warps)
                if ms < best_ms:
                    best, best_ms = (chunk, warps), ms
        if best != heuristic and best_ms >= _WIN_MARGIN * base_ms:
            best = heuristic
        _TUNED_PLANAR[key] = best
        _save_cache()
        return best
    except (RuntimeError, ImportError):
        return None
    finally:
        _CALIBRATING = False
