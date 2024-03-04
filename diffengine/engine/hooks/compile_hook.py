import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner


class CompileHook(Hook):
    """Compile Hook.

    Args:
    ----
        backend (str): The backend to use for compilation.
            Defaults to "inductor".
        mode (str): The mode to use for compilation. Defaults to None.

    """

    priority = "VERY_LOW"

    def __init__(self, backend: str = "inductor", mode: str | None = None,
                 ) -> None:
        super().__init__()
        self.backend = backend
        self.mode = mode

    def before_train(self, runner: Runner) -> None:
        """Compile the model.

        Args:
        ----
            runner (Runner): The runner of the training process.

        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        if hasattr(model, "_forward_compile"):
            target = "_forward_compile"
            func = getattr(model, target)
            compiled_func = torch.compile(
                func, backend=self.backend, mode=self.mode)
            setattr(model, target, compiled_func)
        elif hasattr(model, "unet"):
            model.unet = torch.compile(model.unet, backend=self.backend,
                                    mode=self.mode)
        elif hasattr(model, "transformer"):
            model.transformer = torch.compile(
                model.transformer, backend=self.backend, mode=self.mode)
        else:
            msg = "The model has no main network to compile."
            raise NotImplementedError(
                msg)

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
