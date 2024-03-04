import os.path as osp
from collections import OrderedDict

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner


class ControlNetSaveHook(Hook):
    """ControlNet Save Hook.

    Save ControlNet weights with diffusers format and pick up ControlNet
    weights from checkpoint.
    """

    priority = "VERY_LOW"

    def before_save_checkpoint(self, runner: Runner, checkpoint: dict) -> None:
        """Before save checkpoint hook.

        Args:
        ----
            runner (Runner): The runner of the training, validation or testing
                process.
            checkpoint (dict): Model's checkpoint.

        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        # not save no grad key
        new_ckpt = OrderedDict()
        sd_keys = checkpoint["state_dict"].keys()
        for k in sd_keys:
            if "controlnet" in k:
                new_ckpt[k] = checkpoint["state_dict"][k]
        checkpoint["state_dict"] = new_ckpt

    def after_run(self, runner: Runner) -> None:
        """After run hook."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        ckpt_path = osp.join(runner.work_dir, f"step{runner.iter}")
        for p in model.controlnet.parameters():
            is_contiguous = p.is_contiguous()
            break
        if not is_contiguous:
            model.controlnet = model.controlnet.to(
                memory_format=torch.contiguous_format)
        model.controlnet.save_pretrained(osp.join(ckpt_path, "controlnet"))
