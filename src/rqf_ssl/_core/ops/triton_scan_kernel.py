from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class ScanInputs:
    c1_real: torch.Tensor
    c1_imag: torch.Tensor
    c2_real: torch.Tensor
    c2_imag: torch.Tensor
    u: torch.Tensor
    u_real: torch.Tensor
    u_imag: torch.Tensor | None
    batch_size: int
    time_size: int
    filter_size: int


def _prepare_scan_inputs(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    require_complex_u: bool = True,
) -> ScanInputs:

    batch_size, time_size, filter_size = u.shape

    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)

    if require_complex_u:
        if not u.is_complex():
            u = u.to(torch.cfloat)
    else:
        if u.is_complex():
            u = u.real

    u = u.contiguous()
    C1 = C1.contiguous()
    C2 = C2.contiguous()

    target_dtype = u.dtype if not u.is_complex() else u.real.dtype
    c1_real = C1.real.to(target_dtype).contiguous()
    c1_imag = C1.imag.to(target_dtype).contiguous()
    c2_real = C2.real.to(target_dtype).contiguous()
    c2_imag = C2.imag.to(target_dtype).contiguous()

    if u.is_complex():
        u_real = u.real.contiguous()
        u_imag = u.imag.contiguous()
    else:
        u_real = u
        u_imag = None

    return ScanInputs(
        c1_real=c1_real,
        c1_imag=c1_imag,
        c2_real=c2_real,
        c2_imag=c2_imag,
        u=u,
        u_real=u_real,
        u_imag=u_imag,
        batch_size=batch_size,
        time_size=time_size,
        filter_size=filter_size,
    )


@triton.jit
def _sequential_scan_kernel(
    u_real_ptr,
    u_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    batch_size,
    time_size,
    filter_size,
    stride_batch,
    stride_time,
    stride_filter,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)

    batch_idx = pid // filter_size
    filter_idx = pid % filter_size

    if batch_idx >= batch_size:
        return

    c1_real = tl.load(c1_real_ptr + filter_idx)
    c1_imag = tl.load(c1_imag_ptr + filter_idx)
    c2_real = tl.load(c2_real_ptr + filter_idx)
    c2_imag = tl.load(c2_imag_ptr + filter_idx)

    first_offset = (
        batch_idx * stride_batch + 0 * stride_time + filter_idx * stride_filter
    )
    first_u = tl.load(u_real_ptr + first_offset)
    x_real = first_u - first_u
    x_imag = first_u - first_u

    for t in range(time_size):
        offset = batch_idx * stride_batch + t * stride_time + filter_idx * stride_filter

        u_r = tl.load(u_real_ptr + offset)
        u_i = tl.load(u_imag_ptr + offset)

        c1x_real = c1_real * x_real - c1_imag * x_imag
        c1x_imag = c1_real * x_imag + c1_imag * x_real

        c2u_real = c2_real * u_r - c2_imag * u_i
        c2u_imag = c2_real * u_i + c2_imag * u_r

        x_real = c1x_real + c2u_real
        x_imag = c1x_imag + c2u_imag

        tl.store(out_real_ptr + offset, x_real)
        tl.store(out_imag_ptr + offset, x_imag)


def _sequential_scan_forward_raw(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:

    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=True)

    out_real = torch.zeros_like(inputs.u_real)
    out_imag = torch.zeros_like(inputs.u_imag)

    stride_batch = inputs.u_real.stride(0)
    stride_time = inputs.u_real.stride(1)
    stride_filter = inputs.u_real.stride(2)

    grid = (inputs.batch_size * inputs.filter_size,)

    _sequential_scan_kernel[grid](
        inputs.u_real,
        inputs.u_imag,
        out_real,
        out_imag,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        inputs.batch_size,
        inputs.time_size,
        inputs.filter_size,
        stride_batch,
        stride_time,
        stride_filter,
        BLOCK_SIZE=256,
    )

    return torch.complex(out_real, out_imag)


@triton.jit
def _sequential_scan_backward_kernel(
    u_real_ptr,
    u_imag_ptr,
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    grad_out_real_ptr,
    grad_out_imag_ptr,
    scratch_xr_ptr,
    scratch_xi_ptr,
    grad_u_real_ptr,
    grad_u_imag_ptr,
    grad_c1r_ptr,
    grad_c1i_ptr,
    grad_c2r_ptr,
    grad_c2i_ptr,
    batch_size,
    time_size,
    filter_size,
    stride_batch,
    stride_time,
    stride_filter,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    batch_idx = pid // filter_size
    filter_idx = pid % filter_size

    if batch_idx >= batch_size:
        return

    c1r = tl.load(c1_real_ptr + filter_idx)
    c1i = tl.load(c1_imag_ptr + filter_idx)
    c2r = tl.load(c2_real_ptr + filter_idx)
    c2i = tl.load(c2_imag_ptr + filter_idx)

    first_offset = (
        batch_idx * stride_batch + 0 * stride_time + filter_idx * stride_filter
    )
    first_u = tl.load(u_real_ptr + first_offset)
    zero = first_u - first_u
    xr = zero
    xi = zero

    scratch_pid_off = pid.to(tl.int64) * time_size

    for t in range(time_size):
        offset = batch_idx * stride_batch + t * stride_time + filter_idx * stride_filter

        ur = tl.load(u_real_ptr + offset)
        ui = tl.load(u_imag_ptr + offset)

        xr_new = c1r * xr - c1i * xi + c2r * ur - c2i * ui
        xi_new = c1r * xi + c1i * xr + c2r * ui + c2i * ur
        xr = xr_new
        xi = xi_new

        scratch_off = scratch_pid_off + t
        tl.store(scratch_xr_ptr + scratch_off, xr)
        tl.store(scratch_xi_ptr + scratch_off, xi)

    adj_r = zero
    adj_i = zero

    gc1r = zero
    gc1i = zero
    gc2r = zero
    gc2i = zero

    for t_rev in range(time_size):
        t = time_size - 1 - t_rev
        offset = batch_idx * stride_batch + t * stride_time + filter_idx * stride_filter

        go_r = tl.load(grad_out_real_ptr + offset)
        go_i = tl.load(grad_out_imag_ptr + offset)

        adj_r_new = c1r * adj_r + c1i * adj_i + go_r
        adj_i_new = c1r * adj_i - c1i * adj_r + go_i

        gu_r = c2r * adj_r_new + c2i * adj_i_new
        gu_i = c2r * adj_i_new - c2i * adj_r_new
        tl.store(grad_u_real_ptr + offset, gu_r)
        tl.store(grad_u_imag_ptr + offset, gu_i)

        ur = tl.load(u_real_ptr + offset)
        ui = tl.load(u_imag_ptr + offset)
        gc2r += adj_r_new * ur + adj_i_new * ui
        gc2i += adj_i_new * ur - adj_r_new * ui

        if t > 0:
            scratch_off_prev = scratch_pid_off + (t - 1)
            xr_prev = tl.load(scratch_xr_ptr + scratch_off_prev)
            xi_prev = tl.load(scratch_xi_ptr + scratch_off_prev)
            gc1r += adj_r_new * xr_prev + adj_i_new * xi_prev
            gc1i += adj_i_new * xr_prev - adj_r_new * xi_prev

        adj_r = adj_r_new
        adj_i = adj_i_new

    tl.atomic_add(grad_c1r_ptr + filter_idx, gc1r)
    tl.atomic_add(grad_c1i_ptr + filter_idx, gc1i)
    tl.atomic_add(grad_c2r_ptr + filter_idx, gc2r)
    tl.atomic_add(grad_c2i_ptr + filter_idx, gc2i)


def _sequential_scan_backward_raw(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=True)

    if not grad_output.is_complex():
        grad_output = grad_output.to(torch.cfloat)
    grad_out_real = grad_output.real.contiguous()
    grad_out_imag = grad_output.imag.contiguous()

    batch_size = inputs.batch_size
    time_size = inputs.time_size
    filter_size = inputs.filter_size
    target_dtype = inputs.u_real.dtype

    num_programs = batch_size * filter_size
    scratch_xr = torch.empty(
        num_programs, time_size, device=u.device, dtype=target_dtype
    )
    scratch_xi = torch.empty(
        num_programs, time_size, device=u.device, dtype=target_dtype
    )

    grad_u_real = torch.empty(
        batch_size, time_size, filter_size, device=u.device, dtype=target_dtype
    )
    grad_u_imag = torch.empty_like(grad_u_real)
    grad_c1r = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c1i = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c2r = torch.zeros(filter_size, device=u.device, dtype=target_dtype)
    grad_c2i = torch.zeros(filter_size, device=u.device, dtype=target_dtype)

    stride_batch = inputs.u_real.stride(0)
    stride_time = inputs.u_real.stride(1)
    stride_filter = inputs.u_real.stride(2)

    grid = (num_programs,)

    _sequential_scan_backward_kernel[grid](
        inputs.u_real,
        inputs.u_imag,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        grad_out_real,
        grad_out_imag,
        scratch_xr,
        scratch_xi,
        grad_u_real,
        grad_u_imag,
        grad_c1r,
        grad_c1i,
        grad_c2r,
        grad_c2i,
        batch_size,
        time_size,
        filter_size,
        stride_batch,
        stride_time,
        stride_filter,
        BLOCK_SIZE=256,
    )

    grad_C1 = torch.complex(grad_c1r, grad_c1i)
    grad_C2 = torch.complex(grad_c2r, grad_c2i)
    grad_u = torch.complex(grad_u_real, grad_u_imag)

    return grad_C1, grad_C2, grad_u


class _SequentialScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        C1: torch.Tensor,
        C2: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:

        if not C1.is_complex():
            C1 = C1.to(torch.cfloat)
        if not C2.is_complex():
            C2 = C2.to(torch.cfloat)
        if not u.is_complex():
            u = u.to(torch.cfloat)

        output = _sequential_scan_forward_raw(C1, C2, u)

        ctx.save_for_backward(C1, C2, u)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):

        C1, C2, u = ctx.saved_tensors
        grad_C1, grad_C2, grad_u = _sequential_scan_backward_raw(C1, C2, u, grad_output)
        return grad_C1, grad_C2, grad_u


def _triton_sequential_scan(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:

    return _SequentialScanFunction.apply(C1, C2, u)


@triton.jit
def _sequential_scan_multi_osc_kernel(
    u_real_ptr,
    u_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    cw_real_ptr,
    cw_imag_ptr,
    batch_size,
    time_size,
    num_filters,
    num_osc,
    u_stride_batch,
    u_stride_time,
    u_stride_filter,
    out_stride_batch,
    out_stride_time,
    out_stride_filter,
    BLOCK_N: tl.constexpr,
):

    pid = tl.program_id(0)
    batch_idx = pid // num_filters
    filter_idx = pid % num_filters

    if batch_idx >= batch_size:
        return

    osc_start = filter_idx * num_osc
    osc_range = tl.arange(0, BLOCK_N)
    osc_mask = osc_range < num_osc
    coeff_idx = osc_start + osc_range

    c1r = tl.load(c1_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c1i = tl.load(c1_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2r = tl.load(c2_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2i = tl.load(c2_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)

    cw_r = tl.load(cw_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    cw_i = tl.load(cw_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)

    x_r = tl.zeros([BLOCK_N], dtype=tl.float32)
    x_i = tl.zeros([BLOCK_N], dtype=tl.float32)

    for t in range(time_size):
        u_offset = (
            batch_idx * u_stride_batch
            + t * u_stride_time
            + filter_idx * u_stride_filter
        )
        u_r = tl.load(u_real_ptr + u_offset)
        u_i = tl.load(u_imag_ptr + u_offset)

        x_r_new = c1r * x_r - c1i * x_i + c2r * u_r - c2i * u_i
        x_i_new = c1i * x_r + c1r * x_i + c2i * u_r + c2r * u_i
        x_r = x_r_new
        x_i = x_i_new

        out_r = tl.sum(cw_r * x_r - cw_i * x_i, axis=0)
        out_i = tl.sum(cw_r * x_i + cw_i * x_r, axis=0)

        out_offset = (
            batch_idx * out_stride_batch
            + t * out_stride_time
            + filter_idx * out_stride_filter
        )
        tl.store(out_real_ptr + out_offset, out_r)
        tl.store(out_imag_ptr + out_offset, out_i)


def _sequential_scan_multi_osc_forward_raw(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    cw_real: torch.Tensor,
    cw_imag: torch.Tensor,
    num_osc: int,
) -> torch.Tensor:

    if not u.is_complex():
        u = u.to(torch.cfloat)

    batch_size, time_size, num_filters = u.shape

    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)

    u = u.contiguous()

    target_dtype = u.real.dtype
    c1_real = C1.real.to(target_dtype).contiguous()
    c1_imag = C1.imag.to(target_dtype).contiguous()
    c2_real = C2.real.to(target_dtype).contiguous()
    c2_imag = C2.imag.to(target_dtype).contiguous()
    u_real = u.real.contiguous()
    u_imag = u.imag.contiguous()
    cw_real = cw_real.to(target_dtype).contiguous()
    cw_imag = cw_imag.to(target_dtype).contiguous()

    out_real = torch.zeros(
        batch_size, time_size, num_filters, device=u.device, dtype=target_dtype
    )
    out_imag = torch.zeros_like(out_real)

    BLOCK_N = triton.next_power_of_2(num_osc)

    grid = (batch_size * num_filters,)

    _sequential_scan_multi_osc_kernel[grid](
        u_real,
        u_imag,
        out_real,
        out_imag,
        c1_real,
        c1_imag,
        c2_real,
        c2_imag,
        cw_real,
        cw_imag,
        batch_size,
        time_size,
        num_filters,
        num_osc,
        u_real.stride(0),
        u_real.stride(1),
        u_real.stride(2),
        out_real.stride(0),
        out_real.stride(1),
        out_real.stride(2),
        BLOCK_N=BLOCK_N,
    )

    return torch.complex(out_real, out_imag)


@triton.jit
def _sequential_scan_multi_osc_backward_kernel(
    u_real_ptr,
    u_imag_ptr,
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    cw_real_ptr,
    cw_imag_ptr,
    grad_out_real_ptr,
    grad_out_imag_ptr,
    scratch_r_ptr,
    scratch_i_ptr,
    grad_u_real_ptr,
    grad_u_imag_ptr,
    grad_c1r_ptr,
    grad_c1i_ptr,
    grad_c2r_ptr,
    grad_c2i_ptr,
    grad_cwr_ptr,
    grad_cwi_ptr,
    batch_size,
    time_size,
    num_filters,
    num_osc,
    stride_batch,
    stride_time,
    stride_filter,
    BLOCK_N: tl.constexpr,
):

    pid = tl.program_id(0)
    batch_idx = pid // num_filters
    filter_idx = pid % num_filters

    if batch_idx >= batch_size:
        return

    osc_start = filter_idx * num_osc
    osc_range = tl.arange(0, BLOCK_N)
    osc_mask = osc_range < num_osc
    coeff_idx = osc_start + osc_range

    c1r = tl.load(c1_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c1i = tl.load(c1_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2r = tl.load(c2_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2i = tl.load(c2_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)
    cw_r = tl.load(cw_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    cw_i = tl.load(cw_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)

    x_r = tl.zeros([BLOCK_N], dtype=tl.float32)
    x_i = tl.zeros([BLOCK_N], dtype=tl.float32)

    gcw_r = tl.zeros([BLOCK_N], dtype=tl.float32)
    gcw_i = tl.zeros([BLOCK_N], dtype=tl.float32)

    scratch_pid_off = pid.to(tl.int64) * (time_size * BLOCK_N)

    for t in range(time_size):
        btf_off = (
            batch_idx * stride_batch + t * stride_time + filter_idx * stride_filter
        )
        u_r = tl.load(u_real_ptr + btf_off)
        u_i = tl.load(u_imag_ptr + btf_off)

        x_r_new = c1r * x_r - c1i * x_i + c2r * u_r - c2i * u_i
        x_i_new = c1i * x_r + c1r * x_i + c2i * u_r + c2r * u_i
        x_r = x_r_new
        x_i = x_i_new

        scratch_off = scratch_pid_off + t * BLOCK_N + osc_range
        tl.store(scratch_r_ptr + scratch_off, x_r, mask=osc_mask)
        tl.store(scratch_i_ptr + scratch_off, x_i, mask=osc_mask)

        go_r = tl.load(grad_out_real_ptr + btf_off)
        go_i = tl.load(grad_out_imag_ptr + btf_off)
        gcw_r += go_r * x_r + go_i * x_i
        gcw_i += go_i * x_r - go_r * x_i

    adj_r = tl.zeros([BLOCK_N], dtype=tl.float32)
    adj_i = tl.zeros([BLOCK_N], dtype=tl.float32)

    gc1r = tl.zeros([BLOCK_N], dtype=tl.float32)
    gc1i = tl.zeros([BLOCK_N], dtype=tl.float32)
    gc2r = tl.zeros([BLOCK_N], dtype=tl.float32)
    gc2i = tl.zeros([BLOCK_N], dtype=tl.float32)

    for t_rev in range(time_size):
        t = time_size - 1 - t_rev

        btf_off = (
            batch_idx * stride_batch + t * stride_time + filter_idx * stride_filter
        )
        go_r = tl.load(grad_out_real_ptr + btf_off)
        go_i = tl.load(grad_out_imag_ptr + btf_off)

        gs_r = cw_r * go_r + cw_i * go_i
        gs_i = cw_r * go_i - cw_i * go_r

        adj_r_new = c1r * adj_r + c1i * adj_i + gs_r
        adj_i_new = -c1i * adj_r + c1r * adj_i + gs_i

        grad_u_r_val = tl.sum(c2r * adj_r_new + c2i * adj_i_new, axis=0)
        grad_u_i_val = tl.sum(-c2i * adj_r_new + c2r * adj_i_new, axis=0)
        tl.store(grad_u_real_ptr + btf_off, grad_u_r_val)
        tl.store(grad_u_imag_ptr + btf_off, grad_u_i_val)

        u_r = tl.load(u_real_ptr + btf_off)
        u_i = tl.load(u_imag_ptr + btf_off)
        gc2r += adj_r_new * u_r + adj_i_new * u_i
        gc2i += adj_i_new * u_r - adj_r_new * u_i

        if t > 0:
            scratch_off_prev = scratch_pid_off + (t - 1) * BLOCK_N + osc_range
            x_r_prev = tl.load(
                scratch_r_ptr + scratch_off_prev, mask=osc_mask, other=0.0
            )
            x_i_prev = tl.load(
                scratch_i_ptr + scratch_off_prev, mask=osc_mask, other=0.0
            )
            gc1r += adj_r_new * x_r_prev + adj_i_new * x_i_prev
            gc1i += adj_i_new * x_r_prev - adj_r_new * x_i_prev

        adj_r = adj_r_new
        adj_i = adj_i_new

    tl.atomic_add(grad_c1r_ptr + coeff_idx, gc1r, mask=osc_mask)
    tl.atomic_add(grad_c1i_ptr + coeff_idx, gc1i, mask=osc_mask)
    tl.atomic_add(grad_c2r_ptr + coeff_idx, gc2r, mask=osc_mask)
    tl.atomic_add(grad_c2i_ptr + coeff_idx, gc2i, mask=osc_mask)
    tl.atomic_add(grad_cwr_ptr + coeff_idx, gcw_r, mask=osc_mask)
    tl.atomic_add(grad_cwi_ptr + coeff_idx, gcw_i, mask=osc_mask)


def _sequential_scan_multi_osc_backward_raw(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    c_weights: torch.Tensor,
    grad_output: torch.Tensor,
    num_osc: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    if not u.is_complex():
        u = u.to(torch.cfloat)

    batch_size, time_size, num_filters = u.shape
    scan_filters = C1.shape[0]

    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)
    u = u.contiguous()

    if not grad_output.is_complex():
        grad_output = grad_output.to(torch.cfloat)

    target_dtype = u.real.dtype
    c1_real = C1.real.to(target_dtype).contiguous()
    c1_imag = C1.imag.to(target_dtype).contiguous()
    c2_real = C2.real.to(target_dtype).contiguous()
    c2_imag = C2.imag.to(target_dtype).contiguous()
    cw_real = c_weights[:, 0].to(target_dtype).contiguous()
    cw_imag = c_weights[:, 1].to(target_dtype).contiguous()
    u_real = u.real.contiguous()
    u_imag = u.imag.contiguous()
    grad_out_real = grad_output.real.to(target_dtype).contiguous()
    grad_out_imag = grad_output.imag.to(target_dtype).contiguous()

    BLOCK_N = triton.next_power_of_2(num_osc)
    num_programs = batch_size * num_filters
    scratch_r = torch.empty(
        num_programs, time_size, BLOCK_N, device=u.device, dtype=target_dtype
    )
    scratch_i = torch.empty(
        num_programs, time_size, BLOCK_N, device=u.device, dtype=target_dtype
    )

    grad_u_real = torch.empty(
        batch_size, time_size, num_filters, device=u.device, dtype=target_dtype
    )
    grad_u_imag = torch.empty_like(grad_u_real)
    grad_c1r = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_c1i = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_c2r = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_c2i = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_cwr = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)
    grad_cwi = torch.zeros(scan_filters, device=u.device, dtype=target_dtype)

    grid = (num_programs,)

    _sequential_scan_multi_osc_backward_kernel[grid](
        u_real,
        u_imag,
        c1_real,
        c1_imag,
        c2_real,
        c2_imag,
        cw_real,
        cw_imag,
        grad_out_real,
        grad_out_imag,
        scratch_r,
        scratch_i,
        grad_u_real,
        grad_u_imag,
        grad_c1r,
        grad_c1i,
        grad_c2r,
        grad_c2i,
        grad_cwr,
        grad_cwi,
        batch_size,
        time_size,
        num_filters,
        num_osc,
        u_real.stride(0),
        u_real.stride(1),
        u_real.stride(2),
        BLOCK_N=BLOCK_N,
    )

    grad_C1 = torch.complex(grad_c1r, grad_c1i)
    grad_C2 = torch.complex(grad_c2r, grad_c2i)
    grad_u = torch.complex(grad_u_real, grad_u_imag)
    grad_c_weights = torch.stack([grad_cwr, grad_cwi], dim=-1)

    return grad_C1, grad_C2, grad_u, grad_c_weights


class _SequentialScanMultiOscFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        C1: torch.Tensor,
        C2: torch.Tensor,
        u: torch.Tensor,
        c_weights: torch.Tensor,
        num_osc: int,
    ) -> torch.Tensor:

        if not C1.is_complex():
            C1 = C1.to(torch.cfloat)
        if not C2.is_complex():
            C2 = C2.to(torch.cfloat)
        if not u.is_complex():
            u = u.to(torch.cfloat)

        cw_real = c_weights[:, 0].contiguous()
        cw_imag = c_weights[:, 1].contiguous()

        output = _sequential_scan_multi_osc_forward_raw(
            C1, C2, u, cw_real, cw_imag, num_osc
        )

        ctx.save_for_backward(C1, C2, u, c_weights)
        ctx.num_osc = num_osc

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):

        C1, C2, u, c_weights = ctx.saved_tensors
        num_osc = ctx.num_osc

        grad_C1, grad_C2, grad_u, grad_c_weights = (
            _sequential_scan_multi_osc_backward_raw(
                C1, C2, u, c_weights, grad_output, num_osc
            )
        )

        return grad_C1, grad_C2, grad_u, grad_c_weights, None


def _triton_sequential_scan_multi_osc(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    c_weights: torch.Tensor,
    num_osc: int,
) -> torch.Tensor:

    return _SequentialScanMultiOscFunction.apply(C1, C2, u, c_weights, num_osc)


@triton.jit
def _sequential_scan_b_matrix_kernel(
    u_real_ptr,
    u_imag_ptr,
    b_real_ptr,
    b_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    c1_real_ptr,
    c1_imag_ptr,
    batch_size,
    time_size,
    num_filters,
    input_features,
    u_stride_batch,
    u_stride_time,
    u_stride_feature,
    out_stride_batch,
    out_stride_time,
    out_stride_filter,
    b_stride_filter,
    b_stride_feature,
    BLOCK_FEATURES: tl.constexpr,
):

    pid = tl.program_id(0)

    batch_idx = pid // num_filters
    filter_idx = pid % num_filters

    if batch_idx >= batch_size:
        return

    c1r = tl.load(c1_real_ptr + filter_idx)
    c1i = tl.load(c1_imag_ptr + filter_idx)

    feature_offsets = tl.arange(0, BLOCK_FEATURES)
    feature_mask = feature_offsets < input_features
    b_offset = filter_idx * b_stride_filter + feature_offsets * b_stride_feature
    b_row_real = tl.load(b_real_ptr + b_offset, mask=feature_mask, other=0.0)
    b_row_imag = tl.load(b_imag_ptr + b_offset, mask=feature_mask, other=0.0)

    first_u_offset = batch_idx * u_stride_batch + 0 * u_stride_time
    first_u = tl.load(u_real_ptr + first_u_offset)
    x_r = first_u - first_u
    x_i = first_u - first_u

    for t in range(time_size):
        u_base = batch_idx * u_stride_batch + t * u_stride_time
        u_offset = u_base + feature_offsets * u_stride_feature
        u_r = tl.load(u_real_ptr + u_offset, mask=feature_mask, other=0.0)
        u_i = tl.load(u_imag_ptr + u_offset, mask=feature_mask, other=0.0)

        proj_real = tl.sum(b_row_real * u_r - b_row_imag * u_i, axis=0)
        proj_imag = tl.sum(b_row_real * u_i + b_row_imag * u_r, axis=0)

        x_r_new = c1r * x_r - c1i * x_i + proj_real
        x_i_new = c1i * x_r + c1r * x_i + proj_imag

        x_r = x_r_new
        x_i = x_i_new

        out_offset = (
            batch_idx * out_stride_batch
            + t * out_stride_time
            + filter_idx * out_stride_filter
        )
        tl.store(out_real_ptr + out_offset, x_r)
        tl.store(out_imag_ptr + out_offset, x_i)


def _sequential_scan_b_matrix_forward_raw(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:

    batch_size, time_size, input_features = u.shape
    num_filters = C1.shape[0]

    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)

    if not u.is_complex():
        u = u.to(torch.cfloat)

    B_complex = C2.unsqueeze(-1) * B.to(torch.cfloat)

    u = u.contiguous()
    C1 = C1.contiguous()
    B_complex = B_complex.contiguous()

    c1_real = C1.real.contiguous()
    c1_imag = C1.imag.contiguous()
    u_real = u.real.contiguous()
    u_imag = u.imag.contiguous()
    b_real = B_complex.real.contiguous()
    b_imag = B_complex.imag.contiguous()

    out_real = torch.zeros(
        batch_size, time_size, num_filters, device=u.device, dtype=u_real.dtype
    )
    out_imag = torch.zeros(
        batch_size, time_size, num_filters, device=u.device, dtype=u_real.dtype
    )

    BLOCK_FEATURES = triton.next_power_of_2(input_features)
    BLOCK_FEATURES = min(BLOCK_FEATURES, 1024)

    if input_features > BLOCK_FEATURES:
        raise ValueError(
            f"input_features ({input_features}) exceeds maximum supported ({BLOCK_FEATURES}). "
            "Use PyTorch fallback for large input dimensions."
        )

    grid = (batch_size * num_filters,)

    _sequential_scan_b_matrix_kernel[grid](
        u_real,
        u_imag,
        b_real,
        b_imag,
        out_real,
        out_imag,
        c1_real,
        c1_imag,
        batch_size,
        time_size,
        num_filters,
        input_features,
        u_real.stride(0),
        u_real.stride(1),
        u_real.stride(2),
        out_real.stride(0),
        out_real.stride(1),
        out_real.stride(2),
        b_real.stride(0),
        b_real.stride(1),
        BLOCK_FEATURES=BLOCK_FEATURES,
    )

    return torch.complex(out_real, out_imag)


class _SequentialScanBMatrixFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        C1: torch.Tensor,
        C2: torch.Tensor,
        u: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:

        with torch.no_grad():
            B_complex = C2.unsqueeze(-1) * B.to(C2.dtype)

        output = _sequential_scan_b_matrix_forward_raw(C1, C2, u, B)

        ctx.save_for_backward(C1, C2, u, B, B_complex, output)

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):

        C1, C2, u, B, B_complex, x = ctx.saved_tensors

        if not grad_output.is_complex():
            complex_dtype = (
                torch.complex128
                if grad_output.dtype == torch.float64
                else torch.complex64
            )
            grad_output = grad_output.to(complex_dtype)

        C1_conj = C1.conj()
        num_filters = C1.shape[0]
        ones = torch.ones(num_filters, dtype=C1.dtype, device=C1.device)

        grad_flipped = grad_output.flip(1)
        adj_flipped = _sequential_scan_forward_raw(C1_conj, ones, grad_flipped)
        adj = adj_flipped.flip(1)

        x_conj = x.conj()
        grad_C1 = (adj[:, 1:, :] * x_conj[:, :-1, :]).sum(dim=(0, 1))

        grad_u = torch.einsum("fi,btf->bti", B_complex.conj(), adj)

        u_conj = u.conj()
        grad_B_complex = torch.einsum("btf,bti->fi", adj, u_conj)

        grad_C2 = (grad_B_complex * B.to(adj.dtype)).sum(dim=-1)

        grad_B = (grad_B_complex * C2.conj().unsqueeze(-1)).real

        return grad_C1, grad_C2, grad_u, grad_B


def _triton_sequential_scan_b_matrix(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:

    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)
    if not u.is_complex():
        u = u.to(torch.cfloat)

    return _SequentialScanBMatrixFunction.apply(C1, C2, u, B)


@triton.jit
def _associative_combine(
    A_l_r,
    A_l_i,
    B_l_r,
    B_l_i,
    A_r_r,
    A_r_i,
    B_r_r,
    B_r_i,
):

    A_new_r = A_r_r * A_l_r - A_r_i * A_l_i
    A_new_i = A_r_r * A_l_i + A_r_i * A_l_r

    AB_r = A_r_r * B_l_r - A_r_i * B_l_i
    AB_i = A_r_r * B_l_i + A_r_i * B_l_r
    B_new_r = AB_r + B_r_r
    B_new_i = AB_i + B_r_i

    return A_new_r, A_new_i, B_new_r, B_new_i


@triton.jit
def _parallel_scan_kernel(
    u_real_ptr,
    u_imag_ptr,
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    batch_size,
    time_size,
    filter_size,
    stride_batch,
    stride_time,
    stride_filter,
    BLOCK_T: tl.constexpr,
):

    pid = tl.program_id(0)
    batch_idx = pid // filter_size
    filter_idx = pid % filter_size

    if batch_idx >= batch_size:
        return

    c1_r = tl.load(c1_real_ptr + filter_idx)
    c1_i = tl.load(c1_imag_ptr + filter_idx)
    c2_r = tl.load(c2_real_ptr + filter_idx)
    c2_i = tl.load(c2_imag_ptr + filter_idx)

    t_offsets = tl.arange(0, BLOCK_T)
    mask = t_offsets < time_size

    offsets = (
        batch_idx * stride_batch + t_offsets * stride_time + filter_idx * stride_filter
    )

    u_r = tl.load(u_real_ptr + offsets, mask=mask, other=0.0)
    u_i = tl.load(u_imag_ptr + offsets, mask=mask, other=0.0)

    B_r = c2_r * u_r - c2_i * u_i
    B_i = c2_r * u_i + c2_i * u_r

    A_r = tl.where(mask, c1_r, 1.0)
    A_i = tl.where(mask, c1_i, 0.0)

    _, _, B_scan_r, B_scan_i = tl.associative_scan(
        (A_r, A_i, B_r, B_i),
        axis=0,
        combine_fn=_associative_combine,
    )

    tl.store(out_real_ptr + offsets, B_scan_r, mask=mask)
    tl.store(out_imag_ptr + offsets, B_scan_i, mask=mask)


def _parallel_scan_forward_raw(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:

    inputs = _prepare_scan_inputs(C1, C2, u, require_complex_u=True)

    out_real = torch.empty_like(inputs.u_real)
    out_imag = torch.empty_like(inputs.u_imag)

    BLOCK_T = triton.next_power_of_2(inputs.time_size)
    BLOCK_T = min(BLOCK_T, 16384)

    grid = (inputs.batch_size * inputs.filter_size,)

    _parallel_scan_kernel[grid](
        inputs.u_real,
        inputs.u_imag,
        inputs.c1_real,
        inputs.c1_imag,
        inputs.c2_real,
        inputs.c2_imag,
        out_real,
        out_imag,
        inputs.batch_size,
        inputs.time_size,
        inputs.filter_size,
        inputs.u_real.stride(0),
        inputs.u_real.stride(1),
        inputs.u_real.stride(2),
        BLOCK_T=BLOCK_T,
    )

    return torch.complex(out_real, out_imag)


class _ParallelScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        C1: torch.Tensor,
        C2: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:

        if not C1.is_complex():
            C1 = C1.to(torch.cfloat)
        if not C2.is_complex():
            C2 = C2.to(torch.cfloat)
        if not u.is_complex():
            u = u.to(torch.cfloat)

        output = _parallel_scan_forward_raw(C1, C2, u)
        ctx.save_for_backward(C1, C2, u)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):

        C1, C2, u = ctx.saved_tensors
        grad_C1, grad_C2, grad_u = _sequential_scan_backward_raw(C1, C2, u, grad_output)
        return grad_C1, grad_C2, grad_u


def _triton_parallel_scan(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:

    _, _, in_filters = u.shape
    num_filters = C1.shape[0]
    if in_filters == 1 and num_filters > 1:
        u = u.expand(-1, -1, num_filters)

    time_size = u.shape[1]
    BLOCK_T = triton.next_power_of_2(time_size)
    if BLOCK_T > 16384:
        return _triton_sequential_scan(C1, C2, u)

    return _ParallelScanFunction.apply(C1, C2, u)


@triton.jit
def _parallel_scan_multi_osc_kernel(
    u_real_ptr,
    u_imag_ptr,
    c1_real_ptr,
    c1_imag_ptr,
    c2_real_ptr,
    c2_imag_ptr,
    cw_real_ptr,
    cw_imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    batch_size,
    time_size,
    num_filters,
    num_osc,
    stride_batch,
    stride_time,
    stride_filter,
    out_stride_batch,
    out_stride_time,
    out_stride_filter,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    pid = tl.program_id(0)
    batch_idx = pid // num_filters
    filter_idx = pid % num_filters

    if batch_idx >= batch_size:
        return

    osc_start = filter_idx * num_osc
    osc_range = tl.arange(0, BLOCK_N)
    osc_mask = osc_range < num_osc
    coeff_idx = osc_start + osc_range

    c1_r = tl.load(c1_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c1_i = tl.load(c1_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2_r = tl.load(c2_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    c2_i = tl.load(c2_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)
    cw_r = tl.load(cw_real_ptr + coeff_idx, mask=osc_mask, other=0.0)
    cw_i = tl.load(cw_imag_ptr + coeff_idx, mask=osc_mask, other=0.0)

    t_offsets = tl.arange(0, BLOCK_T)
    t_mask = t_offsets < time_size
    u_offsets = (
        batch_idx * stride_batch + t_offsets * stride_time + filter_idx * stride_filter
    )
    u_r = tl.load(u_real_ptr + u_offsets, mask=t_mask, other=0.0)
    u_i = tl.load(u_imag_ptr + u_offsets, mask=t_mask, other=0.0)

    u_r_2d = u_r[:, None]
    u_i_2d = u_i[:, None]

    c2_r_2d = c2_r[None, :]
    c2_i_2d = c2_i[None, :]

    B_r = c2_r_2d * u_r_2d - c2_i_2d * u_i_2d
    B_i = c2_r_2d * u_i_2d + c2_i_2d * u_r_2d

    c1_r_2d = c1_r[None, :]
    c1_i_2d = c1_i[None, :]
    A_r = tl.where(t_mask[:, None], c1_r_2d, 1.0)
    A_i = tl.where(t_mask[:, None], c1_i_2d, 0.0)

    _, _, X_r, X_i = tl.associative_scan(
        (A_r, A_i, B_r, B_i),
        axis=0,
        combine_fn=_associative_combine,
    )

    cw_r_2d = cw_r[None, :]
    cw_i_2d = cw_i[None, :]

    out_r = tl.sum(cw_r_2d * X_r - cw_i_2d * X_i, axis=1)
    out_i = tl.sum(cw_r_2d * X_i + cw_i_2d * X_r, axis=1)

    out_offsets = (
        batch_idx * out_stride_batch
        + t_offsets * out_stride_time
        + filter_idx * out_stride_filter
    )
    tl.store(out_real_ptr + out_offsets, out_r, mask=t_mask)
    tl.store(out_imag_ptr + out_offsets, out_i, mask=t_mask)


def _parallel_scan_multi_osc_forward_raw(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    cw_real: torch.Tensor,
    cw_imag: torch.Tensor,
    num_osc: int,
) -> torch.Tensor:

    if not u.is_complex():
        u = u.to(torch.cfloat)
    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)

    batch_size, time_size, num_filters = u.shape
    u = u.contiguous()

    target_dtype = u.real.dtype
    c1_real = C1.real.to(target_dtype).contiguous()
    c1_imag = C1.imag.to(target_dtype).contiguous()
    c2_real = C2.real.to(target_dtype).contiguous()
    c2_imag = C2.imag.to(target_dtype).contiguous()
    u_real = u.real.contiguous()
    u_imag = u.imag.contiguous()
    cw_real = cw_real.to(target_dtype).contiguous()
    cw_imag = cw_imag.to(target_dtype).contiguous()

    out_real = torch.zeros(
        batch_size, time_size, num_filters, device=u.device, dtype=target_dtype
    )
    out_imag = torch.zeros_like(out_real)

    BLOCK_T = min(triton.next_power_of_2(time_size), 16384)
    BLOCK_N = triton.next_power_of_2(num_osc)

    grid = (batch_size * num_filters,)

    _parallel_scan_multi_osc_kernel[grid](
        u_real,
        u_imag,
        c1_real,
        c1_imag,
        c2_real,
        c2_imag,
        cw_real,
        cw_imag,
        out_real,
        out_imag,
        batch_size,
        time_size,
        num_filters,
        num_osc,
        u_real.stride(0),
        u_real.stride(1),
        u_real.stride(2),
        out_real.stride(0),
        out_real.stride(1),
        out_real.stride(2),
        BLOCK_T=BLOCK_T,
        BLOCK_N=BLOCK_N,
    )

    return torch.complex(out_real, out_imag)


class _ParallelScanMultiOscFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        C1: torch.Tensor,
        C2: torch.Tensor,
        u: torch.Tensor,
        c_weights: torch.Tensor,
        num_osc: int,
    ) -> torch.Tensor:

        if not C1.is_complex():
            C1 = C1.to(torch.cfloat)
        if not C2.is_complex():
            C2 = C2.to(torch.cfloat)
        if not u.is_complex():
            u = u.to(torch.cfloat)

        cw_real = c_weights[:, 0].contiguous()
        cw_imag = c_weights[:, 1].contiguous()

        output = _parallel_scan_multi_osc_forward_raw(
            C1, C2, u, cw_real, cw_imag, num_osc
        )

        ctx.save_for_backward(C1, C2, u, c_weights)
        ctx.num_osc = num_osc
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):

        C1, C2, u, c_weights = ctx.saved_tensors
        num_osc = ctx.num_osc

        grad_C1, grad_C2, grad_u, grad_c_weights = (
            _sequential_scan_multi_osc_backward_raw(
                C1, C2, u, c_weights, grad_output, num_osc
            )
        )

        return grad_C1, grad_C2, grad_u, grad_c_weights, None


def _triton_parallel_scan_multi_osc(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    c_weights: torch.Tensor,
    num_osc: int,
) -> torch.Tensor:

    time_size = u.shape[1]
    BLOCK_T = triton.next_power_of_2(time_size)
    if BLOCK_T > 16384:
        return _triton_sequential_scan_multi_osc(C1, C2, u, c_weights, num_osc)

    return _ParallelScanMultiOscFunction.apply(C1, C2, u, c_weights, num_osc)
