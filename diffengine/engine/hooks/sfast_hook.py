import functools

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

try:
    from sfast.compilers.diffusion_pipeline_compiler import (
        CompilationConfig,
        _build_ts_compiler,
    )
    from sfast.dynamo.backends.sfast_jit import sfast_jit_trace
except ImportError:
    sfast_jit_trace = None
    _build_ts_compiler = None
    CompilationConfig = None


class SFastHook(Hook):
    """SFast Hook.

    This hook is used to compile the model using the stable-fast library.
    https://github.com/chengzeyi/stable-fast

    Args:
    ----
        backend (str): The backend to use for compilation.
            Defaults to "inductor".
        mode (str): The mode to use for compilation. Defaults to None.
        enable_triton (bool): Whether to enable Triton. Defaults to True.

    """

    priority = "VERY_LOW"

    def __init__(self, backend: str = "inductor", mode: str | None = None,
                 *,
                 enable_triton: bool = True,
                 ) -> None:
        super().__init__()
        if sfast_jit_trace is None:
            msg = "stable-fast is not installed."
            raise ImportError(msg)

        self.backend = backend
        self.mode = mode
        self.enable_triton = enable_triton

    def before_train(self, runner: Runner) -> None:
        """Compile the model.

        Args:
        ----
            runner (Runner): The runner of the training process.

        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        config = CompilationConfig.Default()
        config.enable_triton = self.enable_triton
        is_training = model.unet.training
        if not is_training:
            model.train()
        model.unet = torch.compile(model.unet, backend=functools.partial(
            sfast_jit_trace,
            ts_compiler=_build_ts_compiler(
                config,
                enable_triton_layer_norm=self.enable_triton,
                enable_triton_reshape=self.enable_triton,
                )))

        if not is_training:
            model.eval()

        if hasattr(model, "text_encoder"):
            model.text_encoder = torch.compile(
                model.text_encoder, backend=self.backend, mode=self.mode)
        if hasattr(model, "text_encoder_one"):
            model.text_encoder_one = torch.compile(
                model.text_encoder_one, backend=self.backend, mode=self.mode)
        if hasattr(model, "text_encoder_two"):
            model.text_encoder_two = torch.compile(
                model.text_encoder_two, backend=self.backend, mode=self.mode)
        if hasattr(model, "vae"):
            model.vae = torch.compile(
                model.vae, backend=self.backend, mode=self.mode)
        if hasattr(model, "image_encoder"):
            model.image_encoder = torch.compile(
                model.image_encoder, backend=self.backend, mode=self.mode)
        if hasattr(model, "controlnet"):
            model.controlnet = torch.compile(
                model.controlnet, backend=self.backend, mode=self.mode)
