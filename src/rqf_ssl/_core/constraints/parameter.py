import math
import typing

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.utils.parametrizations import _SpectralNorm


class Unconstrained(nn.Module):
    def __init__(self) -> None:

        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        return X

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:

        return X


class Symmetric(nn.Module):
    def __init__(self) -> None:

        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        return X.triu() + X.triu(1).transpose(-1, -2)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:

        return X


class Diagonal(nn.Module):
    def __init__(self) -> None:

        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        return torch.diag(torch.diag(X))

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:

        return X


class Clamp(nn.Module):
    def __init__(self, min: float = 0.0, max: float = 1.0) -> None:

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        if torch.is_complex(X):
            real = torch.clamp(X.real, self.min, self.max)
            imag = torch.clamp(X.imag, self.min, self.max)
            return torch.complex(real, imag)
        return torch.clamp(X, self.min, self.max)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:

        return X


class AbsoluteValue(nn.Module):
    def __init__(self) -> None:

        super().__init__()

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        if torch.is_complex(X):
            mag = torch.abs(X)
            return torch.complex(mag, torch.zeros_like(mag))
        return torch.abs(X)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:

        return X


class SpectralNorm(nn.Module):
    def __init__(
        self,
        weight: nn.Parameter,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:

        super().__init__()
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps
        self.spectral_norm_object = _SpectralNorm(weight, n_power_iterations, dim, eps)

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        return self.spectral_norm_object(X)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:

        return X


class IdentityPlusSpectralNorm(nn.Module):
    def __init__(
        self,
        weight: nn.Parameter,
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
        **kwargs,
    ) -> None:

        super().__init__()
        self.n_power_iterations = n_power_iterations
        self.dim = dim
        self.eps = eps
        self.spectral_norm_object = _SpectralNorm(weight, n_power_iterations, dim, eps)

        device = kwargs.get("device", weight.device)
        dtype = kwargs.get("dtype", weight.dtype)
        self.register_buffer(
            "identity",
            torch.eye(weight.shape[-1], device=device, dtype=dtype),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        return self.identity + self.spectral_norm_object(X)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:

        return X


class LowRankSVD(nn.Module):
    def __init__(self, rank: int) -> None:

        super().__init__()
        if rank < 1:
            raise ValueError(f"Rank must be a positive integer, but got {rank}")
        self.rank = rank

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        if X.dim() < 2:
            raise ValueError(
                f"LowRank parametrization can only be applied to matrices, "
                f"but got a tensor of dimension {X.dim()}"
            )

        U, S, Vh = torch.linalg.svd(X, full_matrices=False)

        U_k = U[:, : self.rank]
        S_k = S[: self.rank]
        Vh_k = Vh[: self.rank, :]

        X_low_rank = (U_k * S_k) @ Vh_k

        return X_low_rank

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:

        return X


class LowRankFactorized(nn.Module):
    def __init__(self, weight: nn.Parameter, rank: int, **kwargs) -> None:

        super().__init__()
        original_shape = weight.shape
        if len(original_shape) != 2:
            raise ValueError(
                f"LowRankFactorized can only be applied to 2D matrices, "
                f"but got shape {original_shape}"
            )

        n, m = original_shape
        if rank < 1 or rank > min(n, m):
            raise ValueError(f"Rank must be between 1 and min(n, m), but got {rank}")

        self.factor1 = nn.Parameter(torch.empty(n, rank, **kwargs))
        self.factor2 = nn.Parameter(torch.empty(rank, m, **kwargs))

        nn.init.kaiming_uniform_(self.factor1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.factor2, a=math.sqrt(5))

    @typing.override
    def forward(self, X: torch.Tensor) -> torch.Tensor:

        return self.factor1 @ self.factor2

    def right_inverse(self, Y: torch.Tensor) -> torch.Tensor:

        U, S, Vh = torch.linalg.svd(Y, full_matrices=False)

        rank = self.factor1.shape[1]
        U_k = U[:, :rank]
        S_k = S[:rank]
        Vh_k = Vh[:rank, :]

        sqrt_S_k = torch.sqrt(S_k)
        self.factor1.data.copy_(U_k * sqrt_S_k)
        self.factor2.data.copy_(sqrt_S_k.unsqueeze(-1) * Vh_k)

        return Y


class Orthogonal(nn.Module):
    def __init__(self, weight: nn.Parameter, use_trivialization: bool = True) -> None:

        super().__init__()
        self.use_trivialization = use_trivialization

        from torch.nn.utils.parametrizations import _Orthogonal

        self._orthogonal = _Orthogonal(
            weight, orthogonal_map=None, use_trivialization=use_trivialization
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        return self._orthogonal(X)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:

        return self._orthogonal.right_inverse(X)


class BoundedPositiveSigmoid(nn.Module):
    def __init__(self, max_val: float = 1.0) -> None:

        super().__init__()
        self.max_val = max_val

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        return self.max_val * torch.sigmoid(X)

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:

        return X


class WMatrix(nn.Module):
    def __init__(
        self,
        dim: int,
        norm_matrix_constraint: str = "full_rank",
        init_log_W: torch.Tensor | None = None,
        init_log_p: torch.Tensor | None = None,
        init_log_q: torch.Tensor | None = None,
        init_W_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> None:

        super().__init__()
        if not isinstance(dim, int):
            raise TypeError(f"WMatrix requires dim to be an integer, got {type(dim)}")
        if dim <= 0:
            raise ValueError(
                f"WMatrix requires dim to be a positive integer, got {dim}"
            )
        self.dim = dim
        self.norm_matrix_constraint = norm_matrix_constraint

        match self.norm_matrix_constraint:
            case "full_rank":
                if init_log_W is not None:
                    if init_log_W.shape != (self.dim, self.dim):
                        raise ValueError(
                            f"init_log_W must have shape ({self.dim}, {self.dim}), got {tuple(init_log_W.shape)}"
                        )
                    self.log_W = nn.Parameter(init_log_W.clone().detach())
                else:
                    self.log_W = nn.Parameter(
                        2.5 * torch.randn(self.dim, self.dim, **kwargs)
                    )

            case "one_rank":
                if init_log_p is not None:
                    if init_log_p.shape != (self.dim,):
                        raise ValueError(
                            f"init_log_p must have shape ({self.dim},), got {tuple(init_log_p.shape)}"
                        )
                    self.log_p = nn.Parameter(init_log_p.clone().detach())
                else:
                    self.log_p = nn.Parameter(torch.randn(self.dim, **kwargs))

                if init_log_q is not None:
                    if init_log_q.shape != (self.dim,):
                        raise ValueError(
                            f"init_log_q must have shape ({self.dim},), got {tuple(init_log_q.shape)}"
                        )
                    self.log_q = nn.Parameter(init_log_q.clone().detach())
                else:
                    self.log_q = nn.Parameter(0.1 * torch.randn(self.dim, **kwargs))

            case "ones":
                self.register_buffer("log_p", torch.zeros(self.dim, **kwargs))
                self.register_buffer("log_q", torch.zeros(self.dim, **kwargs))

            case "fixed_custom":
                if init_W_weight is not None:
                    if init_W_weight.shape != (self.dim, self.dim):
                        raise ValueError(
                            f"init_W_weight must have shape ({self.dim}, {self.dim}), got {tuple(init_W_weight.shape)}"
                        )
                    self.register_buffer("W_weight", init_W_weight.clone().detach())
                else:
                    self.register_buffer(
                        "W_weight", torch.ones(self.dim, self.dim, **kwargs)
                    )

            case _:
                raise ValueError(
                    f"Unsupported norm matrix constraint '{self.norm_matrix_constraint}'. "
                    f"Supported constraints: {{'full_rank', 'one_rank', 'ones', 'fixed_custom'}}"
                )

    @property
    def p(self) -> torch.Tensor:

        return torch.exp(self.log_p)

    @property
    def q(self) -> torch.Tensor:

        return torch.exp(self.log_q)

    @property
    def W(self) -> torch.Tensor:

        return torch.exp(self.log_W)

    @typing.override
    def forward(self, X: torch.Tensor | None = None) -> torch.Tensor:

        match self.norm_matrix_constraint:
            case "full_rank":
                return self.W

            case "one_rank" | "ones":
                return torch.outer(self.p, self.q)

            case "fixed_custom":
                return self.W_weight

            case _:
                raise ValueError(
                    f"Unsupported norm matrix constraint '{self.norm_matrix_constraint}'"
                )

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:

        return X


class LDSMatrix(nn.Module):
    def __init__(
        self,
        dim: int,
        norm_matrix_constraint: str = "full_rank",
        init_raw_t: torch.Tensor | None = None,
        init_raw_u: torch.Tensor | None = None,
        init_raw_v: torch.Tensor | None = None,
        init_log_W: torch.Tensor | None = None,
        init_log_p: torch.Tensor | None = None,
        init_log_q: torch.Tensor | None = None,
        init_W_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> None:

        super().__init__()
        if not isinstance(dim, int):
            raise ValueError(
                f"LDSMatrix requires dim to be an integer, got {type(dim)}"
            )
        if dim <= 0:
            raise ValueError(
                f"LDSMatrix requires dim to be a positive integer, got {dim}"
            )
        self.dim = dim

        self.norm_matrix_constraint = norm_matrix_constraint
        assert self.norm_matrix_constraint in [
            "full_rank",
            "one_rank",
            "ones",
            "fixed_custom",
        ]

        if init_raw_t is not None:
            if init_raw_t.shape != (self.dim,):
                raise ValueError(
                    f"init_raw_t must have shape ({self.dim},), got {tuple(init_raw_t.shape)}"
                )
            self.raw_t = nn.Parameter(init_raw_t.clone().detach())
        else:
            self.raw_t = nn.Parameter(0.5 * torch.rand(self.dim, **kwargs))

        if init_raw_u is not None:
            if init_raw_u.shape != (self.dim,):
                raise ValueError(
                    f"init_raw_u must have shape ({self.dim},), got {tuple(init_raw_u.shape)}"
                )
            self.raw_u = nn.Parameter(init_raw_u.clone().detach())
        else:
            self.raw_u = nn.Parameter(torch.rand(self.dim, **kwargs))

        if init_raw_v is not None:
            if init_raw_v.shape != (self.dim,):
                raise ValueError(
                    f"init_raw_v must have shape ({self.dim},), got {tuple(init_raw_v.shape)}"
                )
            self.raw_v = nn.Parameter(init_raw_v.clone().detach())
        else:
            self.raw_v = nn.Parameter(torch.rand(self.dim, **kwargs))

        self._W_base = WMatrix(
            dim=self.dim,
            norm_matrix_constraint=norm_matrix_constraint,
            init_log_W=init_log_W,
            init_log_p=init_log_p,
            init_log_q=init_log_q,
            init_W_weight=init_W_weight,
            **kwargs,
        )

    @property
    def t(self) -> torch.Tensor:

        return torch.exp(-torch.abs(self.raw_t))

    @property
    def u(self) -> torch.Tensor:

        return self.raw_u**2

    @property
    def v(self) -> torch.Tensor:

        return self.raw_v**2

    @property
    def W(self) -> torch.Tensor:

        return self._W_base()

    @property
    def z(self) -> torch.Tensor:

        return torch.sqrt(self.u)

    @property
    def sigma(self) -> torch.Tensor:

        return torch.sqrt(self.v)

    @typing.override
    def forward(self, X: torch.Tensor | None = None) -> torch.Tensor:

        eps = 1e-8

        t = self.t
        u = self.u
        v = self.v

        match self.norm_matrix_constraint:
            case "full_rank" | "fixed_custom":
                W = self.W
                W_u = W @ u
                scaling_vector = u / (v + W_u + eps)
                second_term = W * scaling_vector
                lds_mat = -second_term
                lds_mat.diagonal().add_(1.0 / t)
            case "one_rank" | "ones":
                p = self._W_base.p
                q = self._W_base.q
                q_dot_u = torch.dot(q, u)
                W_u = p * q_dot_u
                scaling_vector = u / (v + W_u + eps)
                second_term = torch.outer(p, q * scaling_vector)
                lds_mat = -second_term
                lds_mat.diagonal().add_(1.0 / t)
            case _:
                raise ValueError(
                    f"Unsupported norm matrix constraint '{self.norm_matrix_constraint}'"
                )
        return lds_mat

    def right_inverse(self, X: torch.Tensor) -> torch.Tensor:

        return X


def register_unconstrained(module: nn.Module, parameter_name: str) -> None:

    constraint = Unconstrained()
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_symmetric(module: nn.Module, parameter_name: str) -> None:

    constraint = Symmetric()
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_diagonal(module: nn.Module, parameter_name: str) -> None:

    constraint = Diagonal()
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_clamp(
    module: nn.Module, parameter_name: str, min: float = 0.0, max: float = 1.0
) -> None:

    constraint = Clamp(min=min, max=max)
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_absolute_value(module: nn.Module, parameter_name: str) -> None:

    constraint = AbsoluteValue()
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_spectral_norm(
    module: nn.Module,
    parameter_name: str,
    n_power_iterations: int = 1,
    dim: int = 0,
    eps: float = 1e-12,
) -> None:

    param = getattr(module, parameter_name)
    constraint = SpectralNorm(param, n_power_iterations, dim, eps)
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_identity_plus_spectral_norm(
    module: nn.Module,
    parameter_name: str,
    n_power_iterations: int = 1,
    dim: int = 0,
    eps: float = 1e-12,
    **kwargs,
) -> None:

    param = getattr(module, parameter_name)
    constraint = IdentityPlusSpectralNorm(param, n_power_iterations, dim, eps, **kwargs)
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_low_rank_svd(module: nn.Module, parameter_name: str, rank: int) -> None:

    constraint = LowRankSVD(rank)
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_low_rank_factorized(
    module: nn.Module, parameter_name: str, rank: int, **kwargs
) -> None:

    param = getattr(module, parameter_name)
    constraint = LowRankFactorized(param, rank, **kwargs)
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_w_matrix(
    module: nn.Module,
    parameter_name: str,
    dim: int,
    norm_matrix_constraint: str = "full_rank",
    init_log_W: torch.Tensor | None = None,
    init_log_p: torch.Tensor | None = None,
    init_log_q: torch.Tensor | None = None,
    init_W_weight: torch.Tensor | None = None,
    **kwargs,
) -> None:

    constraint = WMatrix(
        dim=dim,
        norm_matrix_constraint=norm_matrix_constraint,
        init_log_W=init_log_W,
        init_log_p=init_log_p,
        init_log_q=init_log_q,
        init_W_weight=init_W_weight,
        **kwargs,
    )
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_orthogonal(
    module: nn.Module, parameter_name: str, use_trivialization: bool = True
) -> None:

    param = getattr(module, parameter_name)
    constraint = Orthogonal(param, use_trivialization=use_trivialization)
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_bounded_positive_sigmoid(
    module: nn.Module,
    parameter_name: str,
    max_val: float = 1.0,
) -> None:

    constraint = BoundedPositiveSigmoid(max_val=max_val)
    parametrize.register_parametrization(module, parameter_name, constraint)


def register_lds_matrix(
    module: nn.Module,
    parameter_name: str,
    dim: int,
    norm_matrix_constraint: str = "full_rank",
    init_raw_t: torch.Tensor | None = None,
    init_raw_u: torch.Tensor | None = None,
    init_raw_v: torch.Tensor | None = None,
    init_log_W: torch.Tensor | None = None,
    init_log_p: torch.Tensor | None = None,
    init_log_q: torch.Tensor | None = None,
    init_W_weight: torch.Tensor | None = None,
    **kwargs,
) -> None:

    constraint = LDSMatrix(
        dim=dim,
        norm_matrix_constraint=norm_matrix_constraint,
        init_raw_t=init_raw_t,
        init_raw_u=init_raw_u,
        init_raw_v=init_raw_v,
        init_log_W=init_log_W,
        init_log_p=init_log_p,
        init_log_q=init_log_q,
        init_W_weight=init_W_weight,
        **kwargs,
    )
    parametrize.register_parametrization(module, parameter_name, constraint)
