from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import AutoModelForCausalLM, LlamaTokenizer

from diffengine.models.editors import (
    LaViBridgeTextAdapter,
    StableDiffusionLaViBridge,
)

base_model = "runwayml/stable-diffusion-v1-5"
llm_model = "mistralai/Mistral-7B-v0.1"
model = dict(type=StableDiffusionLaViBridge,
             model=base_model,
             tokenizer=dict(type=LlamaTokenizer.from_pretrained,
                            pretrained_model_name_or_path=llm_model),
             scheduler=dict(type=DDPMScheduler.from_pretrained,
                            subfolder="scheduler"),
             text_encoder=dict(type=AutoModelForCausalLM.from_pretrained,
                               pretrained_model_name_or_path=llm_model),
             adapter=dict(type=LaViBridgeTextAdapter,
                          in_dim=4096, int_dim=2432, out_dim=768),
             vae=dict(
                type=AutoencoderKL.from_pretrained,
                subfolder="vae"),
             unet=dict(type=UNet2DConditionModel.from_pretrained,
                             subfolder="unet"),
    text_encoder_lora_config=dict(
        type=LoraConfig,
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]),
    unet_lora_config=dict(
        type=LoraConfig,
        r=32,
        lora_alpha=32,
        target_modules=["to_q", "to_v", "to_k", "to_out.0",
                        "proj", "proj_in", "proj_out", "time_emb_proj",
                        "ff.net.2", "conv_shortcut", "conv1", "conv2",
                        "conv_out", "conv"]),
    gradient_checkpointing=True)
