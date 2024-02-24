import copy
import logging

from mmengine.hooks.ema_hook import EMAHook as Base
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.registry import MODELS
from mmengine.runner import Runner


class EMAHook(Base):
    """EMA Hook.

    Args:
    ----
        ema_key (str): The key of the model to apply EMA.

    """

    def __init__(self, *args, ema_key: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ema_key = ema_key

    def before_run(self, runner: Runner) -> None:
        """Create an ema copy of the model.

        Args:
        ----
            runner (Runner): The runner of the training process.

        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        self.src_model = getattr(model, self.ema_key)
        self.ema_model = MODELS.build(
            self.ema_cfg, default_args={"model": self.src_model})

    def _swap_ema_state_dict(self, checkpoint: dict) -> None:
        """Swap the state dict values of model with ema_model."""
        model_state = checkpoint["state_dict"]
        ema_state = checkpoint["ema_state_dict"]
        for k in ema_state:
            if k[:7] == "module.":
                tmp = ema_state[k]
                # 'module.' -> '{self.ema_key}.'
                ema_state[k] = model_state[f"{self.ema_key}." + k[7:]]
                model_state[f"{self.ema_key}." + k[7:]] = tmp

    def after_load_checkpoint(self, runner: Runner, checkpoint: dict) -> None:
        """Resume ema parameters from checkpoint.

        Args:
        ----
            runner (Runner): The runner of the testing process.
            checkpoint (dict): Model's checkpoint.

        """
        from mmengine.runner.checkpoint import load_state_dict
        if "ema_state_dict" in checkpoint and runner._resume:  # noqa
            # The original model parameters are actually saved in ema
            # field swap the weights back to resume ema state.
            self._swap_ema_state_dict(checkpoint)
            self.ema_model.load_state_dict(
                checkpoint["ema_state_dict"], strict=self.strict_load)

        # Support load checkpoint without ema state dict.
        else:
            if runner._resume:  # noqa
                print_log(
                    "There is no `ema_state_dict` in checkpoint. "
                    "`EMAHook` will make a copy of `state_dict` as the "
                    "initial `ema_state_dict`", "current", logging.WARNING)
            sd = copy.deepcopy(checkpoint["state_dict"])
            new_sd = {}
            for k, v in sd.items():
                if k.startswith(f"{self.ema_key}."):
                    new_sd[k[len(self.ema_key) + 1:]] = v
            load_state_dict(
                self.ema_model.module, new_sd, strict=self.strict_load)
