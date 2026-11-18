import torch
import torch.nn as nn

from rqf_ssl._core.constraints.parameter import WMatrix


class DivisiveNormalization(nn.Module):
    def __init__(
        self,
        num_channels: int,
        sigma_init: float = 0.1,
        learn_sigma: bool = False,
        norm_matrix_constraint: str = "ones",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if sigma_init <= 0:
            raise ValueError(f"sigma_init must be positive, got {sigma_init}")

        self.num_channels = num_channels
        self.norm_matrix_constraint = norm_matrix_constraint
        self.eps = eps

        log_sigma = torch.log(torch.tensor(sigma_init))
        if learn_sigma:
            self.log_sigma = nn.Parameter(log_sigma)
        else:
            self.register_buffer("log_sigma", log_sigma)

        self.w_matrix = WMatrix(
            num_channels, norm_matrix_constraint=norm_matrix_constraint
        )

    @property
    def W(self) -> torch.Tensor:

        return self.w_matrix()

    @property
    def sigma_sq(self) -> torch.Tensor:

        return torch.exp(2.0 * self.log_sigma)

    def forward(
        self, z_fwd: torch.Tensor, z_bwd: torch.Tensor | None = None
    ) -> torch.Tensor:

        mag_sq_fwd = z_fwd.real**2 + z_fwd.imag**2 if z_fwd.is_complex() else z_fwd**2

        if z_bwd is not None:
            mag_sq_bwd = (
                z_bwd.real**2 + z_bwd.imag**2 if z_bwd.is_complex() else z_bwd**2
            )
            mag_sq = torch.cat([mag_sq_fwd, mag_sq_bwd], dim=-1)
        else:
            mag_sq = mag_sq_fwd

        inhibition = torch.matmul(mag_sq, self.W.T)
        return mag_sq / (self.sigma_sq + inhibition + self.eps)

    def extra_repr(self) -> str:

        return (
            f"num_channels={self.num_channels}, "
            f"sigma_init={torch.exp(self.log_sigma).item():.4f}, "
            f"learn_sigma={isinstance(self.log_sigma, nn.Parameter)}, "
            f"constraint={self.norm_matrix_constraint}"
        )
