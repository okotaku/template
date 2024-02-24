# Stable Diffusion

[Stable Diffusion](https://github.com/CompVis/stable-diffusion)

## Abstract

TBD.

<div align=center>
<img src="https://github.com/okotaku/diffengine/assets/24734142/542189e1-e88f-4e80-9a51-de241b37d994"/>
</div>

## Citation

```
```

## Run Training

Run Training

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}

# Example.
$ diffengine train stable_diffusion_v15_pokemon_blip
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel

prompt = 'yoda pokemon'
checkpoint = 'work_dirs/stable_diffusion_v15_pokemon_blip/step10450'

unet = UNet2DConditionModel.from_pretrained(
    checkpoint, subfolder='unet', torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', unet=unet, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```

You can see more details on [`docs/source/run_guides/run.md`](../../docs/source/run_guides/run.md#inference-with-diffusers).

## Results Example

#### stable_diffusion_v15_pokemon_blip

![example1](https://github.com/okotaku/diffengine/assets/24734142/24d5254d-95be-46eb-8982-b38b6a11f1ba)
