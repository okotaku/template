# Stable Diffusion DremBooth Training

You can also check [`configs/dreambooth/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/dreambooth/README.md) file.

## Configs

All configuration files are placed under the [`configs/dreambooth`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/dreambooth/) folder.

Following is the example config from the stable_diffusion_v15_dreambooth_lora_dog config file in [`configs/dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/dreambooth/stable_diffusion_v15_dreambooth_lora_dog.py):

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.dog_dreambooth import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15_lora import *
    from .._base_.schedules.stable_diffusion_1k import *
```

## Run DreamBooth training

Run DreamBooth train

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train stable_diffusion_v15_dreambooth_lora_dog

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

checkpoint = Path('work_dirs/stable_diffusion_v15_dreambooth_lora_dog/step999')
prompt = 'A photo of sks dog in a bucket'

pipe = DiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
pipe.to('cuda')
pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint / "unet", adapter_name="default")

image = pipe(
    prompt,
    num_inference_steps=50
).images[0]
image.save('demo.png')
```
