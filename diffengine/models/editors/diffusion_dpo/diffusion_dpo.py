from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from diffengine.models.editors.diffusion_dpo.data_preprocessor import (
    DPODataPreprocessor,
)
from diffengine.models.editors.stable_diffusion import StableDiffusion
from diffengine.models.losses import L2Loss


class StableDiffusionDPO(StableDiffusion):
    """DPO.

    Args:
    ----
        beta_dpo (int): DPO KL Divergence penalty. Defaults to 2000.
        loss (dict, optional): The loss config. Defaults to
            ``dict(type='L2Loss', loss_weight=1.0, "reduction": "none")``.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`DPODataPreprocessor`.

    """

    def __init__(self,
                 *args,
                 beta_dpo: int = 2000,
                 loss: dict | None = None,
                 data_preprocessor: dict | nn.Module | None = None,
                 **kwargs) -> None:
        if loss is None:
            loss = {"type": L2Loss, "loss_weight": 1.0,
                    "reduction": "none"}
        if data_preprocessor is None:
            data_preprocessor = {"type": DPODataPreprocessor}

        super().__init__(
            *args,
            loss=loss,
            data_preprocessor=data_preprocessor,
            **kwargs)  # type: ignore[misc]

        self.beta_dpo = beta_dpo

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.orig_unet = deepcopy(
            self.unet).requires_grad_(requires_grad=False)

        super().prepare_model()

    def loss(  # type: ignore[override]
        self,
        model_pred: torch.Tensor,
        ref_pred: torch.Tensor,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        noisy_model_input: torch.Tensor,
        sigmas: torch.Tensor | None = None,
        weight: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Calculate loss."""
        if self.edm_style:
            model_pred = self.scheduler.precondition_outputs(
                noisy_model_input, model_pred, sigmas)

        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.scheduler.register_to_config(
                prediction_type=self.prediction_type)

        if self.edm_style:
            gt = latents
        elif self.scheduler.config.prediction_type == "epsilon":
            gt = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            gt = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            msg = f"Unknown prediction type {self.scheduler.config.prediction_type}"
            raise ValueError(msg)

        loss_dict = {}
        # calculate loss in FP32
        if self.loss_module.use_snr:
            model_loss = self.loss_module(
                model_pred.float(),
                gt.float(),
                timesteps,
                self.scheduler.alphas_cumprod,
                self.scheduler.config.prediction_type,
                weight=weight)
            ref_loss = self.loss_module(
                ref_pred.float(),
                gt.float(),
                timesteps,
                self.scheduler.alphas_cumprod,
                self.scheduler.config.prediction_type,
                weight=weight)
        else:
            model_loss = self.loss_module(
                model_pred.float(), gt.float(), weight=weight)
            ref_loss = self.loss_module(
                ref_pred.float(), gt.float(), weight=weight)
            model_loss = model_loss.mean(
                dim=list(range(1, len(model_loss.shape))))
            ref_loss = ref_loss.mean(
                dim=list(range(1, len(ref_loss.shape))))

        model_losses_w, model_losses_l = model_loss.chunk(2)
        model_diff = model_losses_w - model_losses_l

        ref_losses_w, ref_losses_l = ref_loss.chunk(2)
        ref_diff = ref_losses_w - ref_losses_l
        scale_term = -0.5 * self.beta_dpo
        inside_term = scale_term * (model_diff - ref_diff)
        loss = -1 * F.logsigmoid(inside_term).mean()
        loss_dict["loss"] = loss
        return loss_dict

    def forward(
            self,
            inputs: dict,
            data_samples: Optional[list] = None,  # noqa
            mode: str = "loss") -> dict:
        """Forward function.

        Args:
        ----
            inputs (dict): The input dict.
            data_samples (Optional[list], optional): The data samples.
                Defaults to None.
            mode (str, optional): The mode. Defaults to "loss".

        Returns:
        -------
            dict: The loss dict.

        """
        assert mode == "loss"
        num_batches = len(inputs["img"])

        latents = self._forward_vae(inputs["img"].to(self.weight_dtype), num_batches)

        noise = self.noise_generator(latents[:num_batches // 2])
        # repeat noise for each sample set
        noise = noise.repeat(2, 1, 1, 1)

        timesteps = self.timesteps_generator(
            self.scheduler, num_batches // 2, self.device)
        # repeat timesteps for each sample set
        timesteps = timesteps.repeat(2)

        noisy_model_input, inp_noisy_latents, sigmas = self._preprocess_model_input(
            latents, noise, timesteps)

        if not self.pre_compute_text_embeddings:
            inputs["text"] = self.tokenizer(
                inputs["text"],
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt").input_ids.to(self.device)
            encoder_hidden_states = self.text_encoder(inputs["text"])[0]
        else:
            encoder_hidden_states = inputs["prompt_embeds"].to(self.weight_dtype)
        # repeat text embeds for each sample set
        encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)

        model_pred = self.unet(
            inp_noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states).sample
        with torch.no_grad():
            ref_pred = self.orig_unet(
                inp_noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states).sample

        return self.loss(model_pred, ref_pred, noise, latents, timesteps,
                         noisy_model_input, sigmas)
