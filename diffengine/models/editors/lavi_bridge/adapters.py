# Refert to: https://github.com/ShihaoZhaoZSH/LaVi-Bridge/blob/main/modules/
# adapters.py#L49
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from torch import nn


class GEGLU(nn.Module):
    """GEGLU."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """Feed Forward."""

    def __init__(self, dim: int, dim_out: int, mult: int = 4,
                 dropout: float=0.1) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = GEGLU(dim, inner_dim)
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        return self.net(x)


@dataclass
class TextAdapterOutput(BaseOutput):
    """Text Adapter Output."""

    sample: torch.FloatTensor


class LaViBridgeTextAdapter(ModelMixin, ConfigMixin):
    """LaViBridge Text Adapter."""

    @register_to_config
    def __init__(self, in_dim: int, int_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.ff1 = FeedForward(in_dim, int_dim)
        self.ff2 = FeedForward(int_dim, out_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(int_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        x = self.ff1(self.norm1(x))
        x = self.ff2(self.norm2(x))
        return TextAdapterOutput(x)
