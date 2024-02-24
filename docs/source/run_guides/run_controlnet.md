# Stable Diffusion ControlNet Training

You can also check [`configs/controlnet/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/controlnet/README.md) file.

## Configs

All configuration files are placed under the [`configs/controlnet`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/controlnet/) folder.

Following is the example config from the stable_diffusion_v15_controlnet_fill50k config file in [`configs/controlnet/stable_diffusion_v15_controlnet_fill50k.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/controlnet/stable_diffusion_v15_controlnet_fill50k.py):

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.fill50k_controlnet import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15_controlnet import *
    from .._base_.schedules.stable_diffusion_1e import *
```


## Run training

Run train

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train stable_diffusion_v15_controlnet_fill50k

# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}
```

## Inference with diffusers

Once you have trained a model, specify the path to the saved model and utilize it for inference using the `diffusers.pipeline` module.

```py
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

checkpoint = 'work_dirs/stable_diffusion_v15_controlnet_fill50k/step6250'
prompt = 'cyan circle with brown floral background'
condition_image = load_image(
    'https://github.com/okotaku/diffengine/assets/24734142/1af9dbb0-b056-435c-bc4b-62a823889191'
)

controlnet = ControlNetModel.from_pretrained(
        checkpoint, subfolder='controlnet', torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', controlnet=controlnet, torch_dtype=torch.float16)
pipe.to('cuda')

image = pipe(
    prompt,
    condition_image,
    num_inference_steps=50,
).images[0]
image.save('demo.png')
```
