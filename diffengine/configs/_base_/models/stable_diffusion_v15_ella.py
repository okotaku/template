from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import T5EncoderModel, T5Tokenizer

from diffengine.models.editors import (
    ELLA,
    StableDiffusionELLA,
)

base_model = "runwayml/stable-diffusion-v1-5"
llm_model = "google/flan-t5-xl"
model = dict(type=StableDiffusionELLA,
             model=base_model,
             tokenizer=dict(type=T5Tokenizer.from_pretrained,
                            pretrained_model_name_or_path=llm_model),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            subfolder="scheduler"),
             text_encoder=dict(type=T5EncoderModel.from_pretrained,
                               pretrained_model_name_or_path=llm_model),
             adapter=dict(type=ELLA),
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                subfolder="vae"),
             unet=dict(type=UNet2DConditionModel.from_pretrained,
                             subfolder="unet"),
    gradient_checkpointing=True)
