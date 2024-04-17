import os.path as osp
from collections import OrderedDict

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner


class CheckpointHook(Hook):
    """Delete 'vae' from checkpoint for efficient save."""

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

        new_ckpt = OrderedDict()
        sd_keys = checkpoint["state_dict"].keys()
        for k in sd_keys:
            if not checkpoint["state_dict"][k].requires_grad:
                continue
            new_k = k.replace("._orig_mod", "")
            if k.startswith(("unet", "adapter")):
                new_ckpt[new_k] = checkpoint["state_dict"][k]
            elif k.startswith("text_encoder") and hasattr(
                    model,
                    "finetune_text_encoder",
            ) and model.finetune_text_encoder:
                # if not finetune text_encoder, then not save
                new_ckpt[new_k] = checkpoint["state_dict"][k]
        checkpoint["state_dict"] = new_ckpt

    def after_run(self, runner: Runner) -> None:  # noqa: C901
        """After run hook."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        ckpt_path = osp.join(runner.work_dir, f"step{runner.iter}")
        if hasattr(model, "prior"):
            model.prior.save_pretrained(osp.join(ckpt_path, "prior"))
        if hasattr(model, "decoder"):
            model.decoder.save_pretrained(osp.join(ckpt_path, "prior"))
        if hasattr(model, "unet"):
            for p in model.unet.parameters():
                is_contiguous = p.is_contiguous()
                break
            if not is_contiguous:
                model.unet = model.unet.to(
                    memory_format=torch.contiguous_format)
            model.unet.save_pretrained(osp.join(ckpt_path, "unet"))
        if hasattr(model, "transformer"):
            model.transformer.save_pretrained(
                osp.join(ckpt_path, "transformer"))
        if hasattr(
                    model,
                    "finetune_text_encoder",
            ) and model.finetune_text_encoder:
            if hasattr(model, "text_encoder"):
                model.text_encoder.save_pretrained(
                    osp.join(ckpt_path, "text_encoder"))
            if hasattr(model, "text_encoder_one"):
                model.text_encoder_one.save_pretrained(
                    osp.join(ckpt_path, "text_encoder_one"))
            if hasattr(model, "text_encoder_two"):
                model.text_encoder_two.save_pretrained(
                    osp.join(ckpt_path, "text_encoder_two"))
        if hasattr(model, "adapter"):
            model.adapter.save_pretrained(osp.join(ckpt_path, "adapter"))
