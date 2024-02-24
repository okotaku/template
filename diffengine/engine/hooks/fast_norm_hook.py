import torch
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from torch import nn
from torch.nn import functional as F  # noqa


def _fast_gn_forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa
    """Faster group normalization forward.

    Copied from
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/
    fast_norm.py
    """
    if torch.is_autocast_enabled():
        dt = torch.get_autocast_gpu_dtype()
        x = x.to(dt)
        weight = self.weight.to(dt)
        bias = self.bias.to(dt) if self.bias is not None else None
    else:
        weight = self.weight
        bias = self.bias

    with torch.cuda.amp.autocast(enabled=False):
        return F.group_norm(x, self.num_groups, weight, bias, self.eps)


class FastNormHook(Hook):
    """Fast Normalization Hook.

    Replace the normalization layer with a faster one.
    """

    priority = "VERY_LOW"

    def __init__(self) -> None:
        super().__init__()

    def _replace_gn_forward(self, module: nn.Module, name: str) -> None:
        """Replace the group normalization forward with a faster one."""
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if isinstance(target_attr, torch.nn.GroupNorm):
                print_log(f"replaced GN Forward: {name}")
                target_attr.forward = _fast_gn_forward.__get__(
                    target_attr, torch.nn.GroupNorm)

        for name, immediate_child_module in module.named_children():
            self._replace_gn_forward(immediate_child_module, name)

    def before_train(self, runner: Runner) -> None:
        """Replace the normalization layer with a faster one.

        Args:
        ----
            runner (Runner): The runner of the training process.

        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if hasattr(model, "unet"):
            self._replace_gn_forward(model.unet, "unet")
        if hasattr(model, "controlnet"):
            self._replace_gn_forward(model.controlnet, "unet")
