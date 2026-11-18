import torch
import torch.nn as nn


class SplitTanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.complex(torch.tanh(x.real), torch.tanh(x.imag))


class SplitReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return torch.complex(torch.relu(x.real), torch.relu(x.imag))
