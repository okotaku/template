# Stable Diffusion LoRA Training

You can also check [`configs/lora/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/lora/README.md) file.

## Configs

All configuration files are placed under the [`configs/lora`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/lora/) folder.

Following is the example config from the stable_diffusion_v15_lora_pokemon_blip config file in [`configs/lora/stable_diffusion_v15_lora_pokemon_blip.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/lora/stable_diffusion_v15_lora_pokemon_blip.py):

```
from mmengine.config import read_base

from diffengine.engine.hooks import PeftSaveHook, VisualizationHook

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15_lora import *
    from .._base_.schedules.stable_diffusion_50e import *

custom_hooks = [
    dict(type=VisualizationHook, prompt=["yoda pokemon"] * 4),
    dict(type=PeftSaveHook),  # Need to change from CheckpointHook
]
```

## Run LoRA training

Run LoRA training:

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train stable_diffusion_v15_lora_pokemon_blip

# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from peft import PeftModel

checkpoint = Path('work_dirs/stable_diffusion_v15_lora_pokemon_blip/step10450')
prompt = 'yoda pokemon'

pipe = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
pipe.to('cuda')
pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint / "unet", adapter_name="default")

image = pipe(
    prompt,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```
