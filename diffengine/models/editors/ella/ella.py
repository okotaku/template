from typing import Optional

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from mmengine.registry import MODELS

from diffengine.models.editors.stable_diffusion import StableDiffusion


class StableDiffusionELLA(StableDiffusion):
    """ELLA.

    Args:
    ----
        adapter (dict): The adapter config.
        max_length (int, optional): The max length. Defaults to 77.

    """

    def __init__(self,
                 *args,
                 adapter: dict,
                 max_length: int | None = 128,
                 **kwargs) -> None:

        self.adapter_config = adapter

        super().__init__(
            *args,
            **kwargs)  # type: ignore[misc]
        if max_length is not None:
            self.tokenizer.model_max_length = max_length

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.adapter = MODELS.build(self.adapter_config)

        super().prepare_model()
        self.unet.requires_grad_(requires_grad=False)

    @torch.no_grad()
    def infer(self,
              prompt: list[str],
              negative_prompt: str | None = None,
              height: int = 512,
              width: int = 512,
              num_inference_steps: int = 50,
              guidance_scale: float = 7.5,
              output_type: str = "pil",
              seed: int = 0) -> list[np.ndarray]:
        """Inference function.

        Args:
        ----
            prompt (`List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`Optional[str]`):
                The prompt or prompts to guide the image generation.
                Defaults to None.
            height (int):
                The height in pixels of the generated image. Defaults to 512.
            width (int):
                The width in pixels of the generated image. Defaults to 512.
            num_inference_steps (int): Number of inference steps.
                Defaults to 50.
            guidance_scale (float): The guidance scale for the model.
                Defaults to 7.5.
            output_type (str): The output format of the generate image.
                Choose between 'pil' and 'latent'. Defaults to 'pil'.
            seed (int): The seed for random number generator.
                Defaults to 0.

        """
        if self.pre_compute_text_embeddings:
            msg = "Pre-computed text embeddings are not supported yet."
            raise NotImplementedError(msg)
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.model,
            vae=self.vae,
            text_encoder=None,
            tokenizer=None,
            unet=self.unet,
            safety_checker=None,
            torch_dtype=self.weight_dtype,
        )
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            scheduler_args = {"prediction_type": self.prediction_type}
            pipeline.scheduler = pipeline.scheduler.from_config(
                pipeline.scheduler.config, **scheduler_args)
        pipeline.to(self.device)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.scheduler.set_timesteps(num_inference_steps)
        images = []
        for i, p in enumerate(prompt):
            generator = torch.Generator(
                device=self.device).manual_seed(i + seed)
            # Text embeddings
            text_inputs = self.tokenizer(
                p,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt", truncation=True)
            text_embeddings = self.text_encoder(
                text_inputs.input_ids.to(self.device),
                attention_mask=text_inputs.attention_mask.to(self.device),
                ).last_hidden_state

            uncond_input = self.tokenizer(
                "" if negative_prompt is None else negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt", truncation=True)
            # Convert the text embedding back to full precision
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device),
                attention_mask=uncond_input.attention_mask.to(self.device),
                ).last_hidden_state
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            # Latent preparation
            latents = randn_tensor(
                (1, self.unet.in_channels, height // 8, width // 8),
                generator=generator, device=self.device)
            latents = latents * pipeline.scheduler.init_noise_sigma

            # Model prediction
            for t in pipeline.scheduler.timesteps:
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = pipeline.scheduler.scale_model_input(
                    latent_model_input, timestep=t)
                encoder_hidden_states = self.adapter(text_embeddings, t)
                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=encoder_hidden_states).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
                latents = pipeline.scheduler.step(
                    noise_pred, t, latents).prev_sample

            # Decoding
            latents = latents / self.vae.config.scaling_factor

            if output_type == "latent":
                images.append(latents)
            else:
                image = self.vae.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1).squeeze()
                image = (
                    image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
                images.append(image)

        del pipeline
        torch.cuda.empty_cache()

        return images

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

        noise = self.noise_generator(latents)

        timesteps = self.timesteps_generator(self.scheduler, num_batches,
                                            self.device)

        noisy_model_input, inp_noisy_latents, sigmas = self._preprocess_model_input(
            latents, noise, timesteps)

        if not self.pre_compute_text_embeddings:
            text_inputs = self.tokenizer(
                inputs["text"],
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt")
            inputs["text"] = text_inputs.input_ids.to(self.device)
            inputs["attention_mask"] = text_inputs.attention_mask.to(self.device)
            encoder_hidden_states = self.text_encoder(
                inputs["text"], attention_mask=inputs["attention_mask"],
            ).last_hidden_state
        else:
            msg = "Pre-computed text embeddings are not supported yet."
            raise NotImplementedError(msg)

        encoder_hidden_states = self.adapter(encoder_hidden_states,
                                             timesteps)

        model_pred = self.unet(
            inp_noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states).sample

        return self.loss(model_pred, noise, latents, timesteps,
                         noisy_model_input, sigmas)
