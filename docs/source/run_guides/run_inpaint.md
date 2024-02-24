# Stable Diffusion Inpaint Training

You can also check [`configs/inpaint/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/inpaint/README.md) file.

## Configs

All configuration files are placed under the [`configs/inpaint`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/inpaint/) folder.

Following is the example config from the stable_diffusion_inpaint_dog config file in [`configs/inpaint/stable_diffusion_inpaint_dog.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/inpaint/stable_diffusion_inpaint_dog.py):

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.dog_inpaint import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_inpaint import *
    from .._base_.schedules.stable_diffusion_1k import *
```

## Run training

Run train

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train stable_diffusion_inpaint_dog

# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
import torch
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
from diffusers.utils import load_image

prompt = 'a photo of sks dog'
img = 'https://github.com/okotaku/diffengine/assets/24734142/8e02bd0e-9dcc-49b6-94b0-86ab3b40bc2b'
mask = 'https://github.com/okotaku/diffengine/assets/24734142/d0de4fb9-9183-418a-970d-582e9324f05d'
checkpoint = 'work_dirs/stable_diffusion_inpaint_dog/step999'

unet = UNet2DConditionModel.from_pretrained(
    checkpoint, subfolder='unet', torch_dtype=torch.float16)
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    'runwayml/stable-diffusion-inpainting', unet=unet, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    load_image(img).convert("RGB"),
    load_image(mask).convert("L"),
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```
