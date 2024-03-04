import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner


def device_has_tensor_core() -> bool:
    """Determine if the device has tensor cores."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        min_major = 7
        return major >= min_major
    return False


class MemoryFormatHook(Hook):
    """MemoryFormat Hook."""

    priority = "VERY_LOW"

    def __init__(self) -> None:
        super().__init__()
        self.memory_format: torch.memory_format = (
            torch.channels_last if device_has_tensor_core() else
            torch.contiguous_format)

    def before_train(self, runner: Runner) -> None:
        """Compile the model.

        Args:
        ----
            runner (Runner): The runner of the training process.

        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        model.unet = model.unet.to(memory_format=self.memory_format)
        if hasattr(model, "controlnet"):
            model.controlnet = model.controlnet.to(memory_format=self.memory_format)
