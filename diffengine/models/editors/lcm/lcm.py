from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from diffusers import LCMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from mmengine.registry import MODELS
from transformers import CLIPTextConfig

from diffengine.models.editors.stable_diffusion import StableDiffusion
from diffengine.models.utils import DDIMTimeSteps

from .lcm_modules import (
    DDIMSolver,
    extract_into_tensor,
    guidance_scale_embedding,
    scalings_for_boundary_conditions,
)


class StableDiffusionLCM(StableDiffusion):
    """Latent Consistency Models.

    Args:
    ----
        timesteps_generator (dict, optional): The timesteps generator config.
            Defaults to ``dict(type=DDIMTimeSteps)``.
        num_ddim_timesteps (int): Number of DDIM timesteps. Defaults to 50.
        time_cond_proj_dim (int): The time condition projection dimension.
            Defaults to 256.
        w_min (float): Minimum guidance scale. Defaults to 3.0.
        w_max (float): Maximum guidance scale. Defaults to 15.0.
        ema_type (str): The type of EMA.
            Defaults to 'ExponentialMovingAverage'.
        ema_momentum (float): The EMA momentum. Defaults to 0.05.

    """

    def __init__(self,
                 *args,
                 timesteps_generator: dict | None = None,
                 num_ddim_timesteps: int = 50,
                 time_cond_proj_dim: int = 256,
                 w_min: float = 3.0,
                 w_max: float = 15.0,
                 ema_type: str = "ExponentialMovingAverage",
                 ema_momentum: float = 0.05,
                 **kwargs) -> None:
        if timesteps_generator is None:
            timesteps_generator = {"type": DDIMTimeSteps,
                                   "num_ddim_timesteps": num_ddim_timesteps}

        self.ema_cfg = dict(type=ema_type, momentum=ema_momentum)
        self.time_cond_proj_dim = time_cond_proj_dim

        super().__init__(
            *args,
            timesteps_generator=timesteps_generator,
            **kwargs)  # type: ignore[misc]

        if self.loss_module.use_snr:
            msg = "SNR is not supported for LCM."
            raise ValueError(msg)

        assert isinstance(self.timesteps_generator, DDIMTimeSteps),(
            "timesteps_generator must be an instance of DDIMTimeSteps.")

        self.num_ddim_timesteps = num_ddim_timesteps
        self.w_min = w_min
        self.w_max = w_max

        self.solver = DDIMSolver(
            self.scheduler.alphas_cumprod,
            timesteps=self.scheduler.config.num_train_timesteps,
            ddim_timesteps=num_ddim_timesteps,
        )

    def prepare_model(self) -> None:
        """Prepare model for training.

        Disable gradient for some models.
        """
        self.teacher_unet = deepcopy(
            self.unet).requires_grad_(requires_grad=False)
        if self.unet_lora_config is None:
            self.target_unet = MODELS.build(
                self.ema_cfg, default_args=dict(model=self.unet))
            self.guidance_scale = 7.5
            self.mode = "lcm"

            del self.unet
            torch.cuda.empty_cache()

            self.unet = UNet2DConditionModel.from_config(
                self.teacher_unet.config,
                time_cond_proj_dim=self.time_cond_proj_dim)
            self.unet.load_state_dict(self.teacher_unet.state_dict(), strict=False)
        else:
            self.guidance_scale = 0.0
            self.mode = "lora"

        self.register_buffer("alpha_schedule",
                             torch.sqrt(self.scheduler.alphas_cumprod))
        self.register_buffer("sigma_schedule",
                             torch.sqrt(1 - self.scheduler.alphas_cumprod))

        if self.pre_compute_text_embeddings:
            text_encoder_config = CLIPTextConfig.from_pretrained(
                self.model, subfolder="text_encoder")
            prompt_embeds_size = text_encoder_config.hidden_size
        else:
            prompt_embeds_size = self.text_encoder.config.hidden_size
        self.register_buffer("uncond_prompt_embeds",
                            torch.zeros(1, 77, prompt_embeds_size))

        super().prepare_model()

    def set_xformers(self) -> None:
        """Set xformers for model."""
        if self.enable_xformers:
            from diffusers.utils.import_utils import is_xformers_available
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
                self.teacher_unet.enable_xformers_memory_efficient_attention()
                if self.unet_lora_config is None:
                    self.target_unet.enable_xformers_memory_efficient_attention()
            else:
                msg = "Please install xformers to enable memory efficient attention."
                raise ImportError(
                    msg,
                )

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
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model,
                vae=self.vae,
                unet=self.unet,
                scheduler=LCMScheduler.from_pretrained(
                    self.model, subfolder="scheduler"),
                safety_checker=None,
                torch_dtype=self.weight_dtype,
            )
        else:
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                unet=self.unet,
                scheduler=LCMScheduler.from_pretrained(
                    self.model, subfolder="scheduler"),
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
                guidance_scale=self.guidance_scale,
                **kwargs).images[0]
            if output_type == "latent":
                images.append(image)
            else:
                images.append(np.array(image))

        del pipeline
        torch.cuda.empty_cache()

        return images

    def loss(self, *args, **kwargs) -> None:  # noqa: ARG002
        """Loss function."""
        msg = "Loss function is not implemented."
        raise NotImplementedError(msg)

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

        start_timesteps = self.timesteps_generator(self.scheduler, num_batches,
                                            self.device)

        steps = (
            self.scheduler.config.num_train_timesteps // self.num_ddim_timesteps
        )
        timesteps = start_timesteps - steps
        timesteps = torch.where(timesteps < 0, 0, timesteps).long()
        index = (start_timesteps / steps).long()

        # Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(
            start_timesteps)
        c_skip, c_out = scalings_for_boundary_conditions(timesteps)

        noisy_model_input, inp_noisy_latents, sigmas = self._preprocess_model_input(
            latents, noise, timesteps)

        # Sample a random guidance scale w from U[w_min, w_max] and embed it
        w = (self.w_max - self.w_min) * torch.rand((num_batches,)) + self.w_min
        if self.mode == "lcm":
            w_embedding = guidance_scale_embedding(
                w, embedding_dim=self.time_cond_proj_dim)
            w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)
        w = w.reshape(num_batches, 1, 1, 1)
        w = w.to(device=latents.device, dtype=latents.dtype)

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

        # Get online LCM prediction on z_{t_{n + k}}, w, c, t_{n + k}
        noise_pred = self.unet(
            noisy_model_input,
            start_timesteps,
            timestep_cond=w_embedding if mode == "lcm" else None,
            encoder_hidden_states=encoder_hidden_states).sample
        pred_x_0 = self._predict_origin(
            noise_pred,
            start_timesteps,
            noisy_model_input,
        )
        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

        # Use the ODE solver to predict the kth step in the augmented PF-ODE
        # trajectory after noisy_latents with both the conditioning embedding
        # c and unconditional embedding 0. Get teacher model prediction on
        # noisy_latents and conditional embedding
        with torch.no_grad():
            cond_teacher_output = self.teacher_unet(
                noisy_model_input,
                start_timesteps,
                encoder_hidden_states=encoder_hidden_states).sample
            cond_pred_x0 = self._predict_origin(
                cond_teacher_output,
                start_timesteps,
                noisy_model_input,
            )
            cond_pred_noise = self._predict_noise(
                cond_teacher_output,
                start_timesteps,
                noisy_model_input,
            )

            # Get teacher model prediction on noisy_latents and unconditional embedding
            uncond_teacher_output = self.teacher_unet(
                noisy_model_input,
                start_timesteps,
                encoder_hidden_states=self.uncond_prompt_embeds.repeat(
                    num_batches, 1, 1)).sample
            uncond_pred_x0 = self._predict_origin(
                uncond_teacher_output,
                start_timesteps,
                noisy_model_input,
            )
            uncond_pred_noise = self._predict_noise(
                uncond_teacher_output,
                start_timesteps,
                noisy_model_input,
            )

            # Perform "CFG" to get x_prev estimate
            # (using the LCM paper's CFG formulation)
            pred_x0 = cond_pred_x0 + w * (
                cond_pred_x0 - uncond_pred_x0)
            pred_noise = cond_pred_noise + w * (
                cond_pred_noise - uncond_pred_noise)
            x_prev = self.solver.ddim_step(pred_x0, pred_noise, index)

        # Get target LCM prediction on x_prev, w, c, t_n
        with torch.no_grad():
            if mode == "lcm":
                target_noise_pred = self.target_unet(
                    x_prev,
                    timesteps,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=encoder_hidden_states).sample
            else:
                # LCM LoRA
                target_noise_pred = self.unet(
                    x_prev,
                    timesteps,
                    timestep_cond=None,
                    encoder_hidden_states=encoder_hidden_states).sample
            pred_x_0 = self._predict_origin(
                target_noise_pred,
                timesteps,
                x_prev,
            )
            target = c_skip * x_prev + c_out * pred_x_0

        loss_dict = {}
        loss = self.loss_module(model_pred.float(), target.float())
        loss_dict["loss"] = loss
        return loss_dict

    def _predict_origin(self,
                        model_output: torch.Tensor,
                        timesteps: torch.Tensor,
                        sample: torch.Tensor) -> torch.Tensor:
        """Predict the origin of the model output.

        Args:
        ----
            model_output (torch.Tensor): The model output.
            timesteps (torch.Tensor): The timesteps.
            sample (torch.Tensor): The sample.

        """
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.scheduler.register_to_config(
                prediction_type=self.prediction_type)

        sigmas = extract_into_tensor(self.sigma_schedule, timesteps)
        alphas = extract_into_tensor(self.alpha_schedule, timesteps)

        if self.scheduler.config.prediction_type == "epsilon":
            pred_x_0 = (sample - sigmas * model_output) / alphas
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_x_0 = alphas * sample - sigmas * model_output
        else:
            msg = f"Prediction type {self.prediction_type} currently not supported."
            raise ValueError(msg)

        return pred_x_0

    def _predict_noise(self,
                        model_output: torch.Tensor,
                        timesteps: torch.Tensor,
                        sample: torch.Tensor) -> torch.Tensor:
        """Predict the noise of the model output.

        Args:
        ----
            model_output (torch.Tensor): The model output.
            timesteps (torch.Tensor): The timesteps.
            sample (torch.Tensor): The sample.

        """
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.scheduler.register_to_config(
                prediction_type=self.prediction_type)

        sigmas = extract_into_tensor(self.sigma_schedule, timesteps)
        alphas = extract_into_tensor(self.alpha_schedule, timesteps)

        if self.scheduler.config.prediction_type == "epsilon":
            pred_epsilon = model_output
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_epsilon = alphas * model_output + sigmas * sample
        else:
            msg = f"Prediction type {self.prediction_type} currently not supported."
            raise ValueError(msg)

        return pred_epsilon
