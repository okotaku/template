# flake8: noqa: C901
import os.path as osp
import shutil
from collections import OrderedDict

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from peft import get_peft_model_state_dict


class PeftSaveHook(Hook):
    """Peft Save Hook.

    Save LoRA weights with diffusers format and pick up LoRA weights from
    checkpoint.
    """

    priority = "VERY_LOW"
    last_step = -1

    def before_save_checkpoint(self, runner: Runner, checkpoint: dict) -> None:  # noqa
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

        ckpt_path = osp.join(runner.work_dir, f"step{runner.iter}")
        if hasattr(model, "unet"):
            model.unet.save_pretrained(osp.join(ckpt_path, "unet"))
            model_keys = ["unet"]
        elif hasattr(model, "prior"):
            # TODO(takuoko): Delete if bug is fixed in diffusers.  # noqa
            model.prior._internal_dict["_name_or_path"] = "prior"  # noqa
            model.prior.save_pretrained(osp.join(ckpt_path, "prior"))
            model_keys = ["prior"]
        elif hasattr(model, "transformer"):
            model.transformer.save_pretrained(
                osp.join(ckpt_path, "transformer"))
            model_keys = ["transformer"]

        if hasattr(model, "adapter"):
            model.adapter.save_pretrained(
                osp.join(ckpt_path, "adapter"))

        if hasattr(model,
                   "finetune_text_encoder") and model.finetune_text_encoder:
            if hasattr(model, "text_encoder"):
                model.text_encoder.save_pretrained(
                    osp.join(ckpt_path, "text_encoder"))
                model_keys.append("text_encoder")
            if hasattr(model, "text_encoder_one"):
                model.text_encoder_one.save_pretrained(
                    osp.join(ckpt_path, "text_encoder_one"))
                model_keys.append("text_encoder_one")
            if hasattr(model, "text_encoder_two"):
                model.text_encoder_two.save_pretrained(
                    osp.join(ckpt_path, "text_encoder_two"))
                model_keys.append("text_encoder_two")

        # remove previous weights
        if self.last_step >= 0:
            shutil.rmtree(osp.join(runner.work_dir, f"step{self.last_step}"))
        self.last_step = runner.iter

        # not save no grad key
        new_ckpt = OrderedDict()
        for model_key in model_keys:
            state_dict = get_peft_model_state_dict(getattr(model, model_key))
            for k in state_dict:
                # add adapter name
                new_k = ".".join(
                    k.split(".")[:-1] + ["default", k.split(".")[-1]])
                new_ckpt[f"{model_key}.{new_k}"] = state_dict[k]
        sd_keys = checkpoint["state_dict"].keys()
        for k in sd_keys:
            if k.startswith("adapter"):
                new_ckpt[k] = checkpoint["state_dict"][k]
        checkpoint["state_dict"] = new_ckpt
