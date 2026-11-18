from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from rqf_ssl._core.ops import launch_tuning
from rqf_ssl._core.ops.chunked_scan_kernel import _affine_combine, _complex_mul


@triton.jit
def _planar_forward_kernel(
    c1_ptr,
    tap0_ptr,
    tap1_ptr,
    zr_ptr,
    zi_ptr,
    ar_ptr,
    ai_ptr,
    carry_ptr,
    seqlen,
    batch,
    nchunks,
    CHUNK: tl.constexpr,
):

    row = tl.program_id(axis=0)
    chan = row // batch
    idx = tl.arange(0, CHUNK)

    c1_r = tl.load(c1_ptr + chan * 2)
    c1_i = tl.load(c1_ptr + chan * 2 + 1)
    t0_r = tl.load(tap0_ptr + chan * 2)
    t0_i = tl.load(tap0_ptr + chan * 2 + 1)
    t1_r = tl.load(tap1_ptr + chan * 2)
    t1_i = tl.load(tap1_ptr + chan * 2 + 1)

    carry_r = tl.zeros((), dtype=tl.float32)
    carry_i = tl.zeros((), dtype=tl.float32)

    for chunk in range(0, nchunks):
        tl.store(carry_ptr + (row * nchunks + chunk) * 2, carry_r)
        tl.store(carry_ptr + (row * nchunks + chunk) * 2 + 1, carry_i)

        t = chunk * CHUNK + idx
        mask = t < seqlen
        base = row * seqlen + t
        x_r = tl.load(zr_ptr + base, mask=mask, other=0.0).to(tl.float32)
        x_i = tl.load(zi_ptr + base, mask=mask, other=0.0).to(tl.float32)
        tprev = t - 1
        pmask = (tprev >= 0) & mask
        pbase = row * seqlen + tprev
        xp_r = tl.load(zr_ptr + pbase, mask=pmask, other=0.0).to(tl.float32)
        xp_i = tl.load(zi_ptr + pbase, mask=pmask, other=0.0).to(tl.float32)

        b_r = t0_r * x_r - t0_i * x_i + t1_r * xp_r - t1_i * xp_i
        b_i = t0_r * x_i + t0_i * x_r + t1_r * xp_i + t1_i * xp_r
        f_r0 = tl.where(mask, c1_r, 1.0)
        f_i0 = tl.where(mask, c1_i, 0.0)

        s_r, s_i, g_r, g_i = tl.associative_scan(
            (b_r, b_i, f_r0, f_i0), axis=0, combine_fn=_affine_combine
        )
        last = idx == CHUNK - 1
        block_s_r = tl.sum(tl.where(last, s_r, 0.0), axis=0)
        block_s_i = tl.sum(tl.where(last, s_i, 0.0), axis=0)
        block_g_r = tl.sum(tl.where(last, g_r, 0.0), axis=0)
        block_g_i = tl.sum(tl.where(last, g_i, 0.0), axis=0)

        car_r, car_i = _complex_mul(g_r, g_i, carry_r, carry_i)
        out_r = s_r + car_r
        out_i = s_i + car_i
        a_r = tl.where(out_r > 0.0, out_r, 0.0)
        a_i = tl.where(out_i > 0.0, out_i, 0.0)
        tl.store(ar_ptr + base, a_r, mask=mask)
        tl.store(ai_ptr + base, a_i, mask=mask)

        nc_r, nc_i = _complex_mul(block_g_r, block_g_i, carry_r, carry_i)
        carry_r = block_s_r + nc_r
        carry_i = block_s_i + nc_i


@triton.jit
def _planar_backward_kernel(
    c1_ptr,
    tap0_ptr,
    tap1_ptr,
    zr_ptr,
    zi_ptr,
    carry_ptr,
    gor_ptr,
    goi_ptr,
    gzr_ptr,
    gzi_ptr,
    gc1_ptr,
    gt0_ptr,
    gt1_ptr,
    seqlen,
    batch,
    nchunks,
    CHUNK: tl.constexpr,
):

    row = tl.program_id(axis=0)
    chan = row // batch
    idx = tl.arange(0, CHUNK)

    c1_r = tl.load(c1_ptr + chan * 2)
    c1_i = tl.load(c1_ptr + chan * 2 + 1)
    t0_r = tl.load(tap0_ptr + chan * 2)
    t0_i = tl.load(tap0_ptr + chan * 2 + 1)
    t1_r = tl.load(tap1_ptr + chan * 2)
    t1_i = tl.load(tap1_ptr + chan * 2 + 1)
    den = c1_r * c1_r + c1_i * c1_i
    inv_r = c1_r / den
    inv_i = c1_i / den
    invc_r = c1_r / den
    invc_i = -c1_i / den

    carry_r = tl.zeros((), dtype=tl.float32)
    carry_i = tl.zeros((), dtype=tl.float32)
    gc1_r = tl.zeros((), dtype=tl.float32)
    gc1_i = tl.zeros((), dtype=tl.float32)
    gt0_r = tl.zeros((), dtype=tl.float32)
    gt0_i = tl.zeros((), dtype=tl.float32)
    gt1_r = tl.zeros((), dtype=tl.float32)
    gt1_i = tl.zeros((), dtype=tl.float32)

    for chunk in range(0, nchunks):
        rchunk = nchunks - 1 - chunk
        t = rchunk * CHUNK + idx
        mask = t < seqlen
        base = row * seqlen + t

        ec_r = tl.load(carry_ptr + (row * nchunks + rchunk) * 2)
        ec_i = tl.load(carry_ptr + (row * nchunks + rchunk) * 2 + 1)
        z_r = tl.load(zr_ptr + base, mask=mask, other=0.0).to(tl.float32)
        z_i = tl.load(zi_ptr + base, mask=mask, other=0.0).to(tl.float32)
        tprev = t - 1
        pmask = (tprev >= 0) & mask
        pbase = row * seqlen + tprev
        zp_r = tl.load(zr_ptr + pbase, mask=pmask, other=0.0).to(tl.float32)
        zp_i = tl.load(zi_ptr + pbase, mask=pmask, other=0.0).to(tl.float32)

        b_r = t0_r * z_r - t0_i * z_i + t1_r * zp_r - t1_i * zp_i
        b_i = t0_r * z_i + t0_i * z_r + t1_r * zp_i + t1_i * zp_r
        f_r0 = tl.where(mask, c1_r, 1.0)
        f_i0 = tl.where(mask, c1_i, 0.0)
        rs_r, rs_i, rg_r, rg_i = tl.associative_scan(
            (b_r, b_i, f_r0, f_i0), axis=0, combine_fn=_affine_combine
        )
        pc_r, pc_i = _complex_mul(rg_r, rg_i, ec_r, ec_i)
        raw_r = rs_r + pc_r
        raw_i = rs_i + pc_i
        d_r = raw_r - b_r
        d_i = raw_i - b_i
        rawp_r, rawp_i = _complex_mul(d_r, d_i, invc_r, invc_i)

        go_r = tl.load(gor_ptr + base, mask=mask, other=0.0).to(tl.float32)
        go_i = tl.load(goi_ptr + base, mask=mask, other=0.0).to(tl.float32)
        go_r = tl.where(raw_r > 0.0, go_r, 0.0)
        go_i = tl.where(raw_i > 0.0, go_i, 0.0)
        sf_r = tl.where(t < seqlen - 1, c1_r, 1.0)
        sf_i = tl.where(t < seqlen - 1, -c1_i, 0.0)

        g_r, g_i, gg_r, gg_i = tl.associative_scan(
            (go_r, go_i, sf_r, sf_i), axis=0, combine_fn=_affine_combine, reverse=True
        )
        first = idx == 0
        block_g_r = tl.sum(tl.where(first, g_r, 0.0), axis=0)
        block_g_i = tl.sum(tl.where(first, g_i, 0.0), axis=0)
        block_gg_r = tl.sum(tl.where(first, gg_r, 0.0), axis=0)
        block_gg_i = tl.sum(tl.where(first, gg_i, 0.0), axis=0)
        car_r, car_i = _complex_mul(gg_r, gg_i, carry_r, carry_i)
        g_r = g_r + car_r
        g_i = g_i + car_i

        d2_r = g_r - go_r
        d2_i = g_i - go_i
        gp_r, gp_i = _complex_mul(d2_r, d2_i, inv_r, inv_i)
        gz_r = t0_r * g_r + t0_i * g_i + t1_r * gp_r + t1_i * gp_i
        gz_i = t0_r * g_i - t0_i * g_r + t1_r * gp_i - t1_i * gp_r
        tl.store(gzr_ptr + base, gz_r, mask=mask)
        tl.store(gzi_ptr + base, gz_i, mask=mask)

        gc1_r += tl.sum(tl.where(mask, rawp_r * g_r + rawp_i * g_i, 0.0), axis=0)
        gc1_i += tl.sum(tl.where(mask, rawp_r * g_i - rawp_i * g_r, 0.0), axis=0)
        gt0_r += tl.sum(tl.where(mask, z_r * g_r + z_i * g_i, 0.0), axis=0)
        gt0_i += tl.sum(tl.where(mask, z_r * g_i - z_i * g_r, 0.0), axis=0)
        gt1_r += tl.sum(tl.where(pmask, zp_r * g_r + zp_i * g_i, 0.0), axis=0)
        gt1_i += tl.sum(tl.where(pmask, zp_r * g_i - zp_i * g_r, 0.0), axis=0)

        nc_r, nc_i = _complex_mul(block_gg_r, block_gg_i, carry_r, carry_i)
        carry_r = block_g_r + nc_r
        carry_i = block_g_i + nc_i

    tl.store(gc1_ptr + row * 2, gc1_r)
    tl.store(gc1_ptr + row * 2 + 1, gc1_i)
    tl.store(gt0_ptr + row * 2, gt0_r)
    tl.store(gt0_ptr + row * 2 + 1, gt0_i)
    tl.store(gt1_ptr + row * 2, gt1_r)
    tl.store(gt1_ptr + row * 2 + 1, gt1_i)


_FORCED_LAUNCH: tuple[int, int] | None = None


def _pick_planar_launch(
    filters: int, batch: int, dtype: torch.dtype
) -> tuple[int, int]:

    if _FORCED_LAUNCH is not None:
        return _FORCED_LAUNCH
    env_chunk = os.environ.get("RQF_SSL_PLANAR_CHUNK")
    env_warps = os.environ.get("RQF_SSL_PLANAR_WARPS")
    if env_chunk and env_warps:
        return int(env_chunk), int(env_warps)

    tuned = launch_tuning.tuned_planar_launch(filters, batch, dtype)
    programs = filters * batch
    if tuned is not None:
        chunk, warps = tuned
    elif programs < 256:
        chunk, warps = 2048, 4
    elif dtype == torch.bfloat16:
        chunk, warps = 256, 1
    else:
        chunk, warps = 512, 4
    if env_chunk:
        chunk = int(env_chunk)
    if env_warps:
        warps = int(env_warps)
    return chunk, warps


class _PlanarScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, tap0, tap1, zp):

        two, filters, batch, time = zp.shape
        zp = zp.contiguous()
        chunk, warps = _pick_planar_launch(filters, batch, zp.dtype)
        nchunks = (time + chunk - 1) // chunk

        c1_real = torch.view_as_real(C1.to(torch.cfloat).contiguous()).contiguous()
        t0_real = torch.view_as_real(tap0.to(torch.cfloat).contiguous()).contiguous()
        t1_real = torch.view_as_real(tap1.to(torch.cfloat).contiguous()).contiguous()

        act = torch.empty_like(zp)
        carry = torch.empty(
            filters * batch, nchunks, 2, device=zp.device, dtype=torch.float32
        )

        rows = zp.reshape(2, filters * batch, time)
        act_rows = act.reshape(2, filters * batch, time)
        grid = (filters * batch,)
        _planar_forward_kernel[grid](
            c1_real,
            t0_real,
            t1_real,
            rows[0],
            rows[1],
            act_rows[0],
            act_rows[1],
            carry,
            time,
            batch,
            nchunks,
            CHUNK=chunk,
            num_warps=warps,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(c1_real, t0_real, t1_real, zp, carry)
        ctx.launch = (chunk, warps, nchunks)
        return act

    @staticmethod
    def backward(ctx, grad_output):

        c1_real, t0_real, t1_real, zp, carry = ctx.saved_tensors
        two, filters, batch, time = zp.shape
        chunk, warps, nchunks = ctx.launch

        go = grad_output.contiguous().reshape(2, filters * batch, time)
        grad_z = torch.empty_like(zp)
        gz_rows = grad_z.reshape(2, filters * batch, time)
        rows_n = filters * batch
        grad_c1 = torch.empty(rows_n, 2, device=zp.device, dtype=torch.float32)
        grad_t0 = torch.empty(rows_n, 2, device=zp.device, dtype=torch.float32)
        grad_t1 = torch.empty(rows_n, 2, device=zp.device, dtype=torch.float32)

        rows = zp.reshape(2, filters * batch, time)
        grid = (filters * batch,)
        _planar_backward_kernel[grid](
            c1_real,
            t0_real,
            t1_real,
            rows[0],
            rows[1],
            carry,
            go[0],
            go[1],
            gz_rows[0],
            gz_rows[1],
            grad_c1,
            grad_t0,
            grad_t1,
            time,
            batch,
            nchunks,
            CHUNK=chunk,
            num_warps=warps,
            enable_fp_fusion=False,
        )

        def reduce_rows(part: torch.Tensor) -> torch.Tensor:

            summed = part.reshape(filters, batch, 2).sum(dim=1, dtype=torch.float64)
            return torch.view_as_complex(summed.to(torch.float32).contiguous())

        return reduce_rows(grad_c1), reduce_rows(grad_t0), reduce_rows(grad_t1), grad_z


def fbt_planar_scan(
    C1: torch.Tensor,
    tap0: torch.Tensor,
    tap1: torch.Tensor,
    zp: torch.Tensor,
) -> torch.Tensor:

    return _PlanarScan.apply(C1, tap0, tap1, zp)
