from unittest import TestCase

import pytest
import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from mmengine.optim import OptimWrapper
from mmengine.registry import MODELS
from peft import LoraConfig
from torch.optim import SGD
from transformers import CLIPTextModel, CLIPTokenizer

from diffengine.models.editors import StableDiffusionLCM
from diffengine.models.losses import DeBiasEstimationLoss, HuberLoss, SNRL2Loss
from diffengine.models.utils.timesteps import TimeSteps


class TestStableDiffusionLCM(TestCase):

    def _get_config(self) -> dict:
        base_model = "diffusers/tiny-stable-diffusion-torch"
        return dict(
            type=StableDiffusionLCM,
             model=base_model,
             tokenizer=dict(type=CLIPTokenizer.from_pretrained,
                            pretrained_model_name_or_path=base_model,
                            subfolder="tokenizer"),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            pretrained_model_name_or_path=base_model,
                            subfolder="scheduler"),
             text_encoder=dict(type=CLIPTextModel.from_pretrained,
                               pretrained_model_name_or_path=base_model,
                               subfolder="text_encoder"),
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                pretrained_model_name_or_path=base_model,
                subfolder="vae"),
             unet=dict(type=UNet2DConditionModel.from_pretrained,
                             pretrained_model_name_or_path=base_model,
                             subfolder="unet"),
             loss=dict(type=HuberLoss))

    def test_init(self):
        cfg = self._get_config()
        cfg.update(text_encoder_lora_config=dict(type="dummy"))
        with pytest.raises(
                AssertionError, match="If you want to use LoRA"):
            _ = MODELS.build(cfg)

        cfg = self._get_config()
        cfg.update(unet_lora_config=dict(type="dummy"),
                finetune_text_encoder=True)
        with pytest.raises(
                AssertionError, match="If you want to finetune text"):
            _ = MODELS.build(cfg)

        cfg = self._get_config()
        cfg.update(timesteps_generator=dict(type=TimeSteps))
        with pytest.raises(
                AssertionError, match="timesteps_generator must be"):
            _ = MODELS.build(cfg)

        cfg = self._get_config()
        cfg.update(loss=dict(type=SNRL2Loss))
        with pytest.raises(
                ValueError, match="SNR is not supported"):
            _ = MODELS.build(cfg)

        cfg = self._get_config()
        cfg.update(loss=dict(type=DeBiasEstimationLoss))
        with pytest.raises(
                ValueError, match="SNR is not supported"):
            _ = MODELS.build(cfg)

    def test_infer(self):
        cfg = self._get_config()
        StableDiffuser =  MODELS.build(cfg)

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # test device
        assert StableDiffuser.device.type == "cpu"

        # test infer with negative_prompt
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            negative_prompt="noise",
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

        # test infer with latent output
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            output_type="latent",
            height=64,
            width=64)
        assert len(result) == 1
        assert type(result[0]) == torch.Tensor
        assert result[0].shape == (4, 32, 32)

    def test_infer_v_prediction(self):
        cfg = self._get_config()
        cfg.update(prediction_type="v_prediction")
        StableDiffuser = MODELS.build(cfg)
        assert StableDiffuser.prediction_type == "v_prediction"

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

    def test_infer_with_lora(self):
        cfg = self._get_config()
        cfg.update(
            unet_lora_config=dict(
                type=LoraConfig, r=4,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config = dict(
                type=LoraConfig, r=4,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
            finetune_text_encoder=True,
        )
        StableDiffuser =  MODELS.build(cfg)

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

    def test_infer_with_pre_compute_embs(self):
        cfg = self._get_config()
        cfg.update(pre_compute_text_embeddings=True)
        StableDiffuser = MODELS.build(cfg)

        assert not hasattr(StableDiffuser, "tokenizer")
        assert not hasattr(StableDiffuser, "text_encoder")

        # test infer
        result = StableDiffuser.infer(
            ["an insect robot preparing a delicious meal"],
            height=64,
            width=64)
        assert len(result) == 1
        assert result[0].shape == (64, 64, 3)

    def test_train_step(self):
        # test load with loss module
        cfg = self._get_config()
        StableDiffuser =  MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_v_prediction(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(prediction_type="v_prediction")
        StableDiffuser =  MODELS.build(cfg)
        assert StableDiffuser.prediction_type == "v_prediction"

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_lora(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(
            unet_lora_config=dict(
                type=LoraConfig, r=4,
                target_modules=["to_q", "to_v", "to_k", "to_out.0"]),
            text_encoder_lora_config=dict(
                type=LoraConfig, r=4,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]),
        )
        StableDiffuser =  MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_input_perturbation(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(input_perturbation_gamma=0.1)
        StableDiffuser =  MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_gradient_checkpointing(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(gradient_checkpointing=True)
        StableDiffuser =  MODELS.build(cfg)

        # test train step
        data = dict(
            inputs=dict(img=[torch.zeros((3, 64, 64))], text=["a dog"]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_train_step_with_pre_compute_embs(self):
        # test load with loss module
        cfg = self._get_config()
        cfg.update(pre_compute_text_embeddings=True)
        StableDiffuser = MODELS.build(cfg)

        assert not hasattr(StableDiffuser, "tokenizer")
        assert not hasattr(StableDiffuser, "text_encoder")

        # test train step
        data = dict(
            inputs=dict(
                img=[torch.zeros((3, 64, 64))],
                prompt_embeds=[torch.zeros((77, 32))]))
        optimizer = SGD(StableDiffuser.parameters(), lr=0.1)
        optim_wrapper = OptimWrapper(optimizer)
        log_vars = StableDiffuser.train_step(data, optim_wrapper)
        assert log_vars
        assert isinstance(log_vars["loss"], torch.Tensor)

    def test_val_and_test_step(self):
        cfg = self._get_config()
        StableDiffuser =  MODELS.build(cfg)

        # test val_step
        with pytest.raises(NotImplementedError, match="val_step is not"):
            StableDiffuser.val_step(torch.zeros((1, )))

        # test test_step
        with pytest.raises(NotImplementedError, match="test_step is not"):
            StableDiffuser.test_step(torch.zeros((1, )))
