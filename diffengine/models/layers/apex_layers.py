# flake8: noqa: A002
import torch
from apex.contrib.group_norm import GroupNorm as BaseGN
from apex.normalization.fused_layer_norm import FusedLayerNorm as BaseLN
from torch._guards import detect_fake_mode
from torch.nn import functional as F  # noqa: N812


class FusedLayerNorm(BaseLN):
    """FusedLayerNorm layer with the apex implementation."""

    @torch.compiler.disable
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward method of the FusedLayerNorm layer."""
        fake_mode = detect_fake_mode(input)
        if fake_mode:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)
        return super().forward(input)


class GroupNorm(BaseGN):
    """GroupNorm layer with the apex implementation."""

    @torch.compiler.disable
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward method of the GroupNorm layer."""
        return super().forward(input)
