import pytest
import torch

from rqf_ssl._core.embeddings.rqf import block as block_module
from rqf_ssl._core.embeddings.rqf.block import BaseRQFBlock
from rqf_ssl._core.embeddings.rqf.implementations import pure_rqf
from rqf_ssl._core.embeddings.rqf.implementations.pure_rqf import PureRQFEncoder
from rqf_ssl._core.ops import scan as scan_module
from rqf_ssl._core.ops.scan import parallel_scan, sequential_scan


def recurrence(pole, gain, signal):
    state = torch.zeros_like(signal[:, 0])
    result = []
    for step in signal.unbind(1):
        state = pole * state + gain * step
        result.append(state)
    return torch.stack(result, 1)


class _CudaLikeInput:
    is_cuda = True

    def is_complex(self):
        return False

    def to(self, _dtype):
        return self

    def repeat_interleave(self, _repeats, dim):
        assert dim == -1
        return self

    def flip(self, _dim):
        return self


def test_global_triton_disable_makes_native_fbt_ineligible(monkeypatch):
    monkeypatch.setattr(pure_rqf, "_HAS_FBT_PROSPECTIVE", True)
    monkeypatch.setattr(scan_module, "_HAS_TRITON", False)
    monkeypatch.setattr(scan_module, "_SCAN_BACKEND", "chunked")
    encoder = type(
        "EligibleEncoder",
        (),
        {
            "_fbt_native": True,
            "blocks": [
                type("EligibleBlock", (), {"_fbt_eligible": lambda self: True})()
            ],
        },
    )()

    assert not PureRQFEncoder._fbt_enabled(encoder, _CudaLikeInput())


def test_global_triton_disable_skips_prospective_kernel(monkeypatch):
    monkeypatch.setattr(block_module, "_HAS_CHUNKED_PROSPECTIVE", True)
    monkeypatch.setattr(scan_module, "_HAS_TRITON", False)
    monkeypatch.setattr(scan_module, "_SCAN_BACKEND", "chunked")

    def unexpected_kernel(*_args, **_kwargs):
        pytest.fail("prospective kernel was called while Triton was disabled")

    monkeypatch.setattr(
        block_module,
        "_triton_chunked_scan_prospective",
        unexpected_kernel,
        raising=False,
    )
    fallback = object()
    block = type(
        "ProspectiveBlock",
        (),
        {
            "prospective_tau": None,
            "scan_filters": 1,
            "num_filters": 1,
            "_has_c_weights": False,
            "_apply_prospective_taps": lambda self, x, tap0, tap1: x,
            "_scan": lambda self, drive, pole, gain: fallback,
        },
    )()
    rqf = type(
        "ProspectiveRQF",
        (),
        {
            "compute_prospective_taps": lambda self, prospective_tau: (
                torch.ones(1),
                torch.ones(1),
                torch.ones(1),
            )
        },
    )()

    result = BaseRQFBlock._scan_prospective(block, _CudaLikeInput(), rqf, None)

    assert result is fallback


def test_global_triton_disable_skips_fused_multiosc_dispatch(monkeypatch):
    monkeypatch.setattr(block_module, "_HAS_FUSED_MULTI_OSC", True)
    monkeypatch.setattr(scan_module, "_HAS_TRITON", False)
    fallback = object()

    def unexpected_dispatch(*_args, **_kwargs):
        pytest.fail("multi-oscillator kernel was called while Triton was disabled")

    block = type(
        "MultiOscBlock",
        (),
        {
            "prospective_input": False,
            "scan_filters": 2,
            "num_filters": 1,
            "_uniform_osc": True,
            "_input_repeats": torch.tensor([2]),
            "_has_c_weights": False,
            "_compute_coefficients": lambda self, rqf: (
                torch.ones(2),
                torch.ones(2),
            ),
            "_scan_multiosc": unexpected_dispatch,
            "_scan": lambda self, drive, pole, gain: fallback,
        },
    )()

    result = BaseRQFBlock._scan_direction(block, _CudaLikeInput(), object(), None)

    assert result is fallback


@pytest.mark.parametrize("length", [1, 17, 65])
@pytest.mark.parametrize("scan", [parallel_scan, sequential_scan])
def test_scan_values_and_full_gradients_match_recurrence(length, scan):
    torch.manual_seed(23)
    original = [
        torch.full((4,), 0.8 + 0.1j, dtype=torch.complex64),
        torch.randn(4, dtype=torch.complex64),
        torch.randn(2, length, 4, dtype=torch.complex64),
    ]
    actual_inputs = [x.clone().requires_grad_() for x in original]
    oracle_inputs = [x.clone().requires_grad_() for x in original]
    actual = scan(*actual_inputs)
    expected = recurrence(*oracle_inputs)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    weights = torch.randn_like(expected)
    (actual.conj() * weights).real.sum().backward()
    (expected.conj() * weights).real.sum().backward()
    for actual_input, oracle_input in zip(actual_inputs, oracle_inputs):
        # For a single step the pole is unused: no gradient and zero gradient
        # are equivalent, since the initial recurrent state is zero.
        actual_grad = actual_input.grad
        if actual_grad is None:
            assert length == 1 and actual_input is actual_inputs[0]
            actual_grad = torch.zeros_like(actual_input)
        torch.testing.assert_close(actual_grad, oracle_input.grad, atol=2e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_planar_prospective_scan_matches_full_bptt(dtype):
    pytest.importorskip("triton")
    from rqf_ssl._core.ops.fbt_planar_kernel import fbt_planar_scan

    torch.manual_seed(31)
    originals = [
        torch.full((4,), 0.8 + 0.1j, dtype=torch.complex64, device="cuda"),
        torch.randn(4, dtype=torch.complex64, device="cuda") * 0.2,
        torch.randn(4, dtype=torch.complex64, device="cuda") * 0.1,
        torch.randn(2, 4, 2, 273, dtype=dtype, device="cuda"),
    ]
    actual_inputs = [x.clone().requires_grad_() for x in originals]
    oracle_inputs = [x.clone().requires_grad_() for x in originals]
    pole, tap0, tap1, planar = oracle_inputs
    signal = torch.complex(planar[0].float(), planar[1].float())
    state = torch.zeros_like(signal[..., 0])
    previous = torch.zeros_like(state)
    states = []
    for step in signal.unbind(-1):
        state = pole[:, None] * state + tap0[:, None] * step + tap1[:, None] * previous
        states.append(torch.stack((state.real.relu(), state.imag.relu())).to(dtype))
        previous = step
    expected = torch.stack(states, -1)
    actual = fbt_planar_scan(*actual_inputs)
    tolerance = 0.03 if dtype == torch.bfloat16 else 3e-4
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    weights = torch.randn_like(expected)
    (actual * weights).float().sum().backward()
    (expected * weights).float().sum().backward()
    for actual_input, oracle_input in zip(actual_inputs, oracle_inputs):
        torch.testing.assert_close(
            actual_input.grad, oracle_input.grad, atol=tolerance, rtol=tolerance
        )
