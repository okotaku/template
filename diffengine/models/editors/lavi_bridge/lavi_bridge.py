from typing import Optional

import numpy as np
import torch
from mmengine.registry import MODELS

from diffengine.models.editors.stable_diffusion import StableDiffusion

from .pipeline import StableDiffusionLaViBridgePipeline


class StableDiffusionLaViBridge(StableDiffusion):
    """LaVi-Bridge.

    Args:
    ----
        adapter (dict): The adapter config.
        max_length (int, optional): The max length. Defaults to 77.

    """

    def __init__(self,
                 *args,
                 adapter: dict,
                 max_length: int | None = 77,
                 **kwargs) -> None:

        self.adapter_config = adapter

        super().__init__(
            *args,
            **kwargs)  # type: ignore[misc]
        if max_length is not None:
            self.tokenizer.model_max_length = max_length
        self.tokenizer.pad_token = "[PAD]"  # noqa: S105

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.adapter = MODELS.build(self.adapter_config)

        super().prepare_model()

    @torch.no_grad()
    def infer(self,
              prompt: list[str],
              negative_prompt: str | None = None,
              height: int = 512,
              width: int = 512,
              num_inference_steps: int = 50,
              output_type: str = "pil",
              seed: int = 0,
              **kwargs) -> list[np.ndarray]:
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
            output_type (str): The output format of the generate image.
                Choose between 'pil' and 'latent'. Defaults to 'pil'.
            seed (int): The seed for random number generator.
                Defaults to 0.
            **kwargs: Other arguments.

        """
        if self.pre_compute_text_embeddings:
            msg = "Pre-computed text embeddings are not supported yet."
            raise NotImplementedError(msg)
        pipeline = StableDiffusionLaViBridgePipeline.from_pretrained(
            self.model,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            adapter=self.adapter,
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
        images = []
        for i, p in enumerate(prompt):
            generator = torch.Generator(device=self.device).manual_seed(i + seed)
            image = pipeline(
                p,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                output_type=output_type,
                generator=generator,
                **kwargs).images[0]
            if output_type == "latent":
                images.append(image)
            else:
                images.append(np.array(image))

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
            inputs["text"] = self.tokenizer(
                inputs["text"],
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt").input_ids.to(self.device)
            encoder_hidden_states = self.text_encoder(
                inputs["text"], output_hidden_states=True).hidden_states[-1]
        else:
            msg = "Pre-computed text embeddings are not supported yet."
            raise NotImplementedError(msg)

        encoder_hidden_states = self.adapter(encoder_hidden_states).sample

        model_pred = self.unet(
            inp_noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states).sample

        return self.loss(model_pred, noise, latents, timesteps,
                         noisy_model_input, sigmas)
