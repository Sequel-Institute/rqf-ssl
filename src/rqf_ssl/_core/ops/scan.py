import math
import os

import torch

_TRITON_DISABLED = os.environ.get("RQF_SSL_DISABLE_TRITON", "0") == "1"

_SCAN_BACKEND = os.environ.get("RQF_SSL_SCAN_BACKEND", "chunked").lower()

_TRITON_MAX_INPUT_FEATURES = 1024

try:
    from rqf_ssl._core.ops.chunked_scan_kernel import _triton_chunked_scan
    from rqf_ssl._core.ops.triton_scan_kernel import (
        _triton_parallel_scan,
        _triton_sequential_scan,
        _triton_sequential_scan_b_matrix,
    )

    _HAS_TRITON = True and not _TRITON_DISABLED
except ImportError:
    _HAS_TRITON = False


def _pad_to_power_of_two(
    tensors: list[torch.Tensor],
    pad_values: list[float],
) -> tuple[list[torch.Tensor], int, int]:

    batch, time, filters = tensors[0].shape

    log_t = int(math.ceil(math.log2(time))) if time > 0 else 0
    padded_time = 2**log_t if log_t > 0 else 0

    if padded_time <= time:
        return tensors, time, log_t

    pad_size = padded_time - time
    padded = [
        torch.nn.functional.pad(t, (0, 0, 0, pad_size), value=v)
        for t, v in zip(tensors, pad_values, strict=True)
    ]
    return padded, time, log_t


def _complex_mul(
    a_real: torch.Tensor,
    a_imag: torch.Tensor,
    b_real: torch.Tensor,
    b_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    return (
        a_real * b_real - a_imag * b_imag,
        a_real * b_imag + a_imag * b_real,
    )


def _blelloch_scan(
    A_real: torch.Tensor,
    A_imag: torch.Tensor,
    B_real: torch.Tensor,
    B_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    batch, time, filters = A_real.shape
    device = A_real.device

    if time <= 1:
        return A_real, A_imag, B_real, B_imag

    [A_r, A_i, B_r, B_i], orig_time, log_t = _pad_to_power_of_two(
        [A_real.clone(), A_imag.clone(), B_real.clone(), B_imag.clone()],
        [1.0, 0.0, 0.0, 0.0],
    )
    padded_time = 2**log_t

    for d in range(log_t):
        stride = 2 ** (d + 1)
        half_stride = 2**d

        right_indices = torch.arange(stride - 1, padded_time, stride, device=device)
        left_indices = right_indices - half_stride

        if len(right_indices) == 0:
            continue

        A_left_r = A_r[:, left_indices, :]
        A_left_i = A_i[:, left_indices, :]
        B_left_r = B_r[:, left_indices, :]
        B_left_i = B_i[:, left_indices, :]

        A_right_r = A_r[:, right_indices, :]
        A_right_i = A_i[:, right_indices, :]
        B_right_r = B_r[:, right_indices, :]
        B_right_i = B_i[:, right_indices, :]

        A_new_r, A_new_i = _complex_mul(A_right_r, A_right_i, A_left_r, A_left_i)

        AB_r, AB_i = _complex_mul(A_right_r, A_right_i, B_left_r, B_left_i)
        B_new_r = AB_r + B_right_r
        B_new_i = AB_i + B_right_i

        A_r[:, right_indices, :] = A_new_r
        A_i[:, right_indices, :] = A_new_i
        B_r[:, right_indices, :] = B_new_r
        B_i[:, right_indices, :] = B_new_i

    for d in range(log_t - 2, -1, -1):
        stride = 2 ** (d + 1)
        half_stride = 2**d

        left_indices = torch.arange(
            stride - 1, padded_time - half_stride, stride, device=device
        )
        right_indices = left_indices + half_stride

        if len(left_indices) == 0:
            continue

        A_left_r = A_r[:, left_indices, :]
        A_left_i = A_i[:, left_indices, :]
        B_left_r = B_r[:, left_indices, :]
        B_left_i = B_i[:, left_indices, :]

        A_right_r = A_r[:, right_indices, :]
        A_right_i = A_i[:, right_indices, :]
        B_right_r = B_r[:, right_indices, :]
        B_right_i = B_i[:, right_indices, :]

        A_new_r, A_new_i = _complex_mul(A_right_r, A_right_i, A_left_r, A_left_i)
        AB_r, AB_i = _complex_mul(A_right_r, A_right_i, B_left_r, B_left_i)
        B_new_r = AB_r + B_right_r
        B_new_i = AB_i + B_right_i

        A_r[:, right_indices, :] = A_new_r
        A_i[:, right_indices, :] = A_new_i
        B_r[:, right_indices, :] = B_new_r
        B_i[:, right_indices, :] = B_new_i

    A_r = A_r[:, :time, :]
    A_i = A_i[:, :time, :]
    B_r = B_r[:, :time, :]
    B_i = B_i[:, :time, :]

    return A_r, A_i, B_r, B_i


def _sequential_scan_cpu(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    B_complex: torch.Tensor | None,
) -> torch.Tensor:

    batch, time, _ = u.shape
    num_filters = C1.shape[0]
    device = u.device
    use_matrix = B_complex is not None

    x_prev = torch.zeros(batch, num_filters, dtype=torch.cfloat, device=device)
    output = torch.empty(batch, time, num_filters, dtype=torch.cfloat, device=device)

    for t in range(time):
        input_proj = (
            torch.matmul(u[:, t, :], B_complex.T) if use_matrix else C2 * u[:, t, :]
        )

        x_new = C1 * x_prev + input_proj

        output[:, t, :] = x_new
        x_prev = x_new

    return output


def sequential_scan(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    B: torch.Tensor | None = None,
) -> torch.Tensor:

    batch, time, in_filters = u.shape
    num_filters = C1.shape[0]

    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)
    if not u.is_complex():
        u = u.to(torch.cfloat)

    use_matrix = B is not None
    if use_matrix:
        B_complex = C2.unsqueeze(-1) * B.to(torch.cfloat)
    else:
        if in_filters == 1 and num_filters > 1:
            u = u.expand(-1, -1, num_filters).contiguous()

    if u.is_cuda and _HAS_TRITON:
        if use_matrix and in_filters <= _TRITON_MAX_INPUT_FEATURES:
            return _triton_sequential_scan_b_matrix(C1, C2, u, B)
        elif not use_matrix:
            if _SCAN_BACKEND == "chunked":
                return _triton_chunked_scan(C1, C2, u)
            return _triton_sequential_scan(C1, C2, u)

    B_complex_arg = B_complex if use_matrix else None
    return _sequential_scan_cpu(C1, C2, u, B_complex_arg)


def parallel_scan(
    C1: torch.Tensor,
    C2: torch.Tensor,
    u: torch.Tensor,
    B: torch.Tensor | None = None,
) -> torch.Tensor:

    batch, time, in_filters = u.shape
    num_filters = C1.shape[0]

    if not C1.is_complex():
        C1 = C1.to(torch.cfloat)
    if not C2.is_complex():
        C2 = C2.to(torch.cfloat)
    if not u.is_complex():
        u = u.to(torch.cfloat)

    use_matrix = B is not None
    if use_matrix:
        B_complex = C2.unsqueeze(-1) * B.to(torch.cfloat)

        u_projected = torch.einsum("nf,btf->btn", B_complex, u)
    else:
        if in_filters == 1 and num_filters > 1:
            u = u.expand(-1, -1, num_filters).contiguous()
        u_projected = u

    if u_projected.is_cuda and _HAS_TRITON and not use_matrix:
        if _SCAN_BACKEND == "chunked":
            return _triton_chunked_scan(C1, C2, u_projected)
        return _triton_parallel_scan(C1, C2, u_projected)

    scan_input = (
        u_projected if use_matrix else C2.unsqueeze(0).unsqueeze(0) * u_projected
    )

    A_real = C1.real.unsqueeze(0).unsqueeze(0).expand(batch, time, -1).contiguous()
    A_imag = C1.imag.unsqueeze(0).unsqueeze(0).expand(batch, time, -1).contiguous()

    scan_input_real = scan_input.real.contiguous()
    scan_input_imag = scan_input.imag.contiguous()

    _, _, x_real, x_imag = _blelloch_scan(
        A_real, A_imag, scan_input_real, scan_input_imag
    )

    return torch.complex(x_real, x_imag)
