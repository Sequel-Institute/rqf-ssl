from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from rqf_ssl._core.ops.chunked_scan_kernel import _affine_combine, _complex_mul

_BWD_OSC_GROUP = 2


def _pick_multiosc_chunk(n_pow2: int) -> int:

    env = os.environ.get("RQF_SSL_MULTIOSC_CHUNK")
    if env:
        return int(env)
    return 256 if n_pow2 <= 8 else 128


@triton.jit
def _multiosc_forward_kernel(
    c1_ptr,
    c2_ptr,
    cw_ptr,
    u_ptr,
    y_ptr,
    carry_ptr,
    seqlen,
    batch,
    N: tl.constexpr,
    NP2: tl.constexpr,
    CHUNK: tl.constexpr,
    REV: tl.constexpr,
):

    row = tl.program_id(axis=0)
    f = row // batch
    idx = tl.arange(0, CHUNK)
    osc = tl.arange(0, NP2)
    omask = osc < N
    nchunks = tl.cdiv(seqlen, CHUNK)

    sf = f * N + osc
    c1_r = tl.load(c1_ptr + sf * 2, mask=omask, other=1.0)
    c1_i = tl.load(c1_ptr + sf * 2 + 1, mask=omask, other=0.0)
    c2_r = tl.load(c2_ptr + sf * 2, mask=omask, other=0.0)
    c2_i = tl.load(c2_ptr + sf * 2 + 1, mask=omask, other=0.0)
    cw_r = tl.load(cw_ptr + sf * 2, mask=omask, other=0.0)
    cw_i = tl.load(cw_ptr + sf * 2 + 1, mask=omask, other=0.0)

    carry_r = tl.zeros((NP2,), dtype=tl.float32)
    carry_i = tl.zeros((NP2,), dtype=tl.float32)

    for chunk in range(0, nchunks):
        cbase = (row * nchunks + chunk) * NP2 + osc
        tl.store(carry_ptr + cbase * 2, carry_r)
        tl.store(carry_ptr + cbase * 2 + 1, carry_i)

        t = chunk * CHUNK + idx
        mask = t < seqlen

        t_mem = seqlen - 1 - t if REV else t
        off = (row * seqlen + t_mem) * 2
        u_r = tl.load(u_ptr + off, mask=mask, other=0.0)
        u_i = tl.load(u_ptr + off + 1, mask=mask, other=0.0)

        b_r = c2_r[:, None] * u_r[None, :] - c2_i[:, None] * u_i[None, :]
        b_i = c2_r[:, None] * u_i[None, :] + c2_i[:, None] * u_r[None, :]
        f_r0 = tl.where(mask[None, :], c1_r[:, None], 1.0)
        f_i0 = tl.where(mask[None, :], c1_i[:, None], 0.0)

        s_r, s_i, g_r, g_i = tl.associative_scan(
            (b_r, b_i, f_r0, f_i0), axis=1, combine_fn=_affine_combine
        )
        last = (idx == CHUNK - 1)[None, :]
        block_s_r = tl.sum(tl.where(last, s_r, 0.0), axis=1)
        block_s_i = tl.sum(tl.where(last, s_i, 0.0), axis=1)
        block_g_r = tl.sum(tl.where(last, g_r, 0.0), axis=1)
        block_g_i = tl.sum(tl.where(last, g_i, 0.0), axis=1)

        car_r, car_i = _complex_mul(g_r, g_i, carry_r[:, None], carry_i[:, None])
        x_r = s_r + car_r
        x_i = s_i + car_i

        y_r = tl.sum(cw_r[:, None] * x_r - cw_i[:, None] * x_i, axis=0)
        y_i = tl.sum(cw_r[:, None] * x_i + cw_i[:, None] * x_r, axis=0)
        tl.store(y_ptr + off, y_r, mask=mask)
        tl.store(y_ptr + off + 1, y_i, mask=mask)

        nc_r, nc_i = _complex_mul(block_g_r, block_g_i, carry_r, carry_i)
        carry_r = block_s_r + nc_r
        carry_i = block_s_i + nc_i


@triton.jit
def _multiosc_backward_kernel(
    c1_ptr,
    c2_ptr,
    cw_ptr,
    u_ptr,
    carry_ptr,
    grad_out_ptr,
    grad_u_ptr,
    grad_c1_ptr,
    grad_c2_ptr,
    grad_cw_ptr,
    seqlen,
    batch,
    nchunks,
    N: tl.constexpr,
    NP2: tl.constexpr,
    OG: tl.constexpr,
    CHUNK: tl.constexpr,
    REV: tl.constexpr,
):

    row = tl.program_id(axis=0)
    group = tl.program_id(axis=1)
    f = row // batch
    idx = tl.arange(0, CHUNK)
    osc = group * OG + tl.arange(0, OG)
    omask = osc < N

    sf = f * N + osc
    c1_r = tl.load(c1_ptr + sf * 2, mask=omask, other=1.0)
    c1_i = tl.load(c1_ptr + sf * 2 + 1, mask=omask, other=0.0)
    c2_r = tl.load(c2_ptr + sf * 2, mask=omask, other=0.0)
    c2_i = tl.load(c2_ptr + sf * 2 + 1, mask=omask, other=0.0)
    cw_r = tl.load(cw_ptr + sf * 2, mask=omask, other=0.0)
    cw_i = tl.load(cw_ptr + sf * 2 + 1, mask=omask, other=0.0)
    den = c1_r * c1_r + c1_i * c1_i
    invc_r = c1_r / den
    invc_i = -c1_i / den

    acar_r = tl.zeros((OG,), dtype=tl.float32)
    acar_i = tl.zeros((OG,), dtype=tl.float32)
    gc1_r = tl.zeros((OG,), dtype=tl.float32)
    gc1_i = tl.zeros((OG,), dtype=tl.float32)
    gc2_r = tl.zeros((OG,), dtype=tl.float32)
    gc2_i = tl.zeros((OG,), dtype=tl.float32)
    gcw_r = tl.zeros((OG,), dtype=tl.float32)
    gcw_i = tl.zeros((OG,), dtype=tl.float32)

    for chunk in range(0, nchunks):
        rchunk = nchunks - 1 - chunk
        t = rchunk * CHUNK + idx
        mask = t < seqlen
        t_mem = seqlen - 1 - t if REV else t
        off = (row * seqlen + t_mem) * 2

        cbase = (row * nchunks + rchunk) * NP2 + osc
        ec_r = tl.load(carry_ptr + cbase * 2)
        ec_i = tl.load(carry_ptr + cbase * 2 + 1)

        u_r = tl.load(u_ptr + off, mask=mask, other=0.0)
        u_i = tl.load(u_ptr + off + 1, mask=mask, other=0.0)
        b_r = c2_r[:, None] * u_r[None, :] - c2_i[:, None] * u_i[None, :]
        b_i = c2_r[:, None] * u_i[None, :] + c2_i[:, None] * u_r[None, :]
        f_r0 = tl.where(mask[None, :], c1_r[:, None], 1.0)
        f_i0 = tl.where(mask[None, :], c1_i[:, None], 0.0)
        rs_r, rs_i, rg_r, rg_i = tl.associative_scan(
            (b_r, b_i, f_r0, f_i0), axis=1, combine_fn=_affine_combine
        )
        pc_r, pc_i = _complex_mul(rg_r, rg_i, ec_r[:, None], ec_i[:, None])
        x_r = rs_r + pc_r
        x_i = rs_i + pc_i
        d_r = x_r - b_r
        d_i = x_i - b_i
        xp_r, xp_i = _complex_mul(d_r, d_i, invc_r[:, None], invc_i[:, None])

        go_r = tl.load(grad_out_ptr + off, mask=mask, other=0.0)
        go_i = tl.load(grad_out_ptr + off + 1, mask=mask, other=0.0)

        bar_r = cw_r[:, None] * go_r[None, :] + cw_i[:, None] * go_i[None, :]
        bar_i = cw_r[:, None] * go_i[None, :] - cw_i[:, None] * go_r[None, :]
        sf_r = tl.where((t < seqlen - 1)[None, :], c1_r[:, None], 1.0)
        sf_i = tl.where((t < seqlen - 1)[None, :], -c1_i[:, None], 0.0)

        g_r, g_i, gg_r, gg_i = tl.associative_scan(
            (bar_r, bar_i, sf_r, sf_i), axis=1, combine_fn=_affine_combine, reverse=True
        )
        first = (idx == 0)[None, :]
        block_g_r = tl.sum(tl.where(first, g_r, 0.0), axis=1)
        block_g_i = tl.sum(tl.where(first, g_i, 0.0), axis=1)
        block_gg_r = tl.sum(tl.where(first, gg_r, 0.0), axis=1)
        block_gg_i = tl.sum(tl.where(first, gg_i, 0.0), axis=1)
        car_r, car_i = _complex_mul(gg_r, gg_i, acar_r[:, None], acar_i[:, None])
        g_r = g_r + car_r
        g_i = g_i + car_i

        gu_r = tl.sum(c2_r[:, None] * g_r + c2_i[:, None] * g_i, axis=0)
        gu_i = tl.sum(c2_r[:, None] * g_i - c2_i[:, None] * g_r, axis=0)
        if OG == NP2:
            tl.store(grad_u_ptr + off, gu_r, mask=mask)
            tl.store(grad_u_ptr + off + 1, gu_i, mask=mask)
        else:
            tl.atomic_add(grad_u_ptr + off, gu_r, mask=mask)
            tl.atomic_add(grad_u_ptr + off + 1, gu_i, mask=mask)

        mm = mask[None, :]
        gc2_r += tl.sum(
            tl.where(mm, u_r[None, :] * g_r + u_i[None, :] * g_i, 0.0), axis=1
        )
        gc2_i += tl.sum(
            tl.where(mm, u_r[None, :] * g_i - u_i[None, :] * g_r, 0.0), axis=1
        )
        gcw_r += tl.sum(
            tl.where(mm, x_r * go_r[None, :] + x_i * go_i[None, :], 0.0), axis=1
        )
        gcw_i += tl.sum(
            tl.where(mm, x_r * go_i[None, :] - x_i * go_r[None, :], 0.0), axis=1
        )
        gc1_r += tl.sum(tl.where(mm, xp_r * g_r + xp_i * g_i, 0.0), axis=1)
        gc1_i += tl.sum(tl.where(mm, xp_r * g_i - xp_i * g_r, 0.0), axis=1)

        nc_r, nc_i = _complex_mul(block_gg_r, block_gg_i, acar_r, acar_i)
        acar_r = block_g_r + nc_r
        acar_i = block_g_i + nc_i

    pbase = (row * NP2 + osc) * 2
    tl.store(grad_c1_ptr + pbase, gc1_r)
    tl.store(grad_c1_ptr + pbase + 1, gc1_i)
    tl.store(grad_c2_ptr + pbase, gc2_r)
    tl.store(grad_c2_ptr + pbase + 1, gc2_i)
    tl.store(grad_cw_ptr + pbase, gcw_r)
    tl.store(grad_cw_ptr + pbase + 1, gcw_i)


@triton.jit
def _transpose_c64_kernel(
    in_ptr,
    out_ptr,
    rows,
    cols,
    TR: tl.constexpr,
    TC: tl.constexpr,
):

    pid_r = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    rr = pid_r * TR + tl.arange(0, TR)
    cc = pid_c * TC + tl.arange(0, TC)
    m_in = (rr[:, None] < rows) & (cc[None, :] < cols)
    v = tl.load(in_ptr + rr[:, None] * cols + cc[None, :], mask=m_in, other=0.0)
    m_out = (cc[:, None] < cols) & (rr[None, :] < rows)
    tl.store(out_ptr + cc[:, None] * rows + rr[None, :], tl.trans(v), mask=m_out)


def _transpose_c64(u2d: torch.Tensor, rows: int, cols: int) -> torch.Tensor:

    out = torch.empty(cols, rows, dtype=u2d.dtype, device=u2d.device)

    src64 = torch.view_as_real(u2d).view(torch.float64).squeeze(-1)
    dst64 = torch.view_as_real(out).view(torch.float64).squeeze(-1)
    grid = (triton.cdiv(rows, 64), triton.cdiv(cols, 64))
    _transpose_c64_kernel[grid](
        src64,
        dst64,
        rows,
        cols,
        TR=64,
        TC=64,
        num_warps=4,
    )
    return out


class _ChannelFirstTranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u):

        batch, time, filters = u.shape
        ctx.dims = (batch, time, filters)
        return _transpose_c64(
            u.contiguous().reshape(batch * time, filters), batch * time, filters
        ).reshape(filters, batch, time)

    @staticmethod
    def backward(ctx, grad):

        batch, time, filters = ctx.dims
        return _transpose_c64(
            grad.contiguous().reshape(filters, batch * time), filters, batch * time
        ).reshape(batch, time, filters)


class _FusedMultiOscScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, c_weights, u_fbt, num_osc, reversed_time):

        C1 = C1.to(torch.cfloat).contiguous()
        C2 = C2.to(torch.cfloat).contiguous()
        c_weights = c_weights.contiguous()
        u = u_fbt.contiguous()
        filters, batch, time = u.shape
        np2 = triton.next_power_of_2(num_osc)
        chunk = _pick_multiosc_chunk(np2)

        nchunks = (time + chunk - 1) // chunk
        c1_real = torch.view_as_real(C1)
        c2_real = torch.view_as_real(C2)
        u_real = torch.view_as_real(u)
        y_real = torch.empty_like(u_real)
        entry_carries = torch.empty(
            filters * batch, nchunks, np2, 2, device=u.device, dtype=torch.float32
        )
        _multiosc_forward_kernel[(filters * batch,)](
            c1_real,
            c2_real,
            c_weights,
            u_real,
            y_real,
            entry_carries,
            time,
            batch,
            N=num_osc,
            NP2=np2,
            CHUNK=chunk,
            REV=reversed_time,
            num_warps=2,
            enable_fp_fusion=False,
        )
        ctx.save_for_backward(C1, C2, c_weights, u, entry_carries)
        ctx.num_osc = num_osc
        ctx.reversed_time = reversed_time
        return torch.view_as_complex(y_real)

    @staticmethod
    def backward(ctx, grad_y):

        C1, C2, c_weights, u, entry_carries = ctx.saved_tensors
        num_osc = ctx.num_osc
        filters, batch, time = u.shape
        np2 = triton.next_power_of_2(num_osc)
        og = min(np2, _BWD_OSC_GROUP)
        chunk = _pick_multiosc_chunk(np2)
        nchunks = (time + chunk - 1) // chunk

        c1_real = torch.view_as_real(C1.contiguous())
        c2_real = torch.view_as_real(C2.contiguous())
        cw_real = c_weights.contiguous()
        u_real = torch.view_as_real(u.contiguous())

        if not grad_y.is_contiguous() and grad_y.permute(1, 2, 0).is_contiguous():
            grad_y = _transpose_c64(
                grad_y.permute(1, 2, 0).reshape(batch * time, filters),
                batch * time,
                filters,
            ).reshape(filters, batch, time)
        grad_out_real = torch.view_as_real(grad_y.contiguous())

        grad_u = torch.empty_like(u_real) if og == np2 else torch.zeros_like(u_real)
        rows_n = filters * batch
        grad_c1 = torch.empty(rows_n, np2, 2, device=u.device, dtype=torch.float32)
        grad_c2 = torch.empty(rows_n, np2, 2, device=u.device, dtype=torch.float32)
        grad_cw = torch.empty(rows_n, np2, 2, device=u.device, dtype=torch.float32)

        _multiosc_backward_kernel[(filters * batch, np2 // og)](
            c1_real,
            c2_real,
            cw_real,
            u_real,
            entry_carries,
            grad_out_real,
            grad_u,
            grad_c1,
            grad_c2,
            grad_cw,
            time,
            batch,
            nchunks,
            N=num_osc,
            NP2=np2,
            OG=og,
            CHUNK=chunk,
            REV=ctx.reversed_time,
            num_warps=2,
            enable_fp_fusion=False,
        )

        def reduce_partials(part: torch.Tensor) -> torch.Tensor:

            summed = part.reshape(filters, batch, np2, 2).sum(
                dim=1, dtype=torch.float64
            )
            return summed[:, :num_osc].reshape(filters * num_osc, 2).to(torch.float32)

        return (
            torch.view_as_complex(reduce_partials(grad_c1).contiguous()),
            torch.view_as_complex(reduce_partials(grad_c2).contiguous()),
            reduce_partials(grad_cw).contiguous(),
            torch.view_as_complex(grad_u),
            None,
            None,
        )


def fused_multiosc_scan(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    c_weights: torch.Tensor,
    num_osc: int,
    reversed_time: bool = False,
) -> torch.Tensor:

    u_fbt = _ChannelFirstTranspose.apply(u)
    y_fbt = _FusedMultiOscScan.apply(C1, C2, c_weights, u_fbt, num_osc, reversed_time)
    return y_fbt.permute(1, 2, 0)
