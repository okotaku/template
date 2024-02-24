from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from diffengine.models.editors import StableDiffusionInpaint

base_model = "runwayml/stable-diffusion-inpainting"
model = dict(type=StableDiffusionInpaint,
             model=base_model,
             tokenizer=dict(type=CLIPTokenizer.from_pretrained,
                            subfolder="tokenizer"),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            subfolder="scheduler"),
             text_encoder=dict(type=CLIPTextModel.from_pretrained,
                               subfolder="text_encoder"),
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                subfolder="vae"),
             unet=dict(type=UNet2DConditionModel.from_pretrained,
                             subfolder="unet"))
