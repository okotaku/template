# Stable Diffusion Training

You can also check [`configs/stable_diffusion/README.md`](https://github.com/okotaku/diffengine/tree/main/diffengine/configs/stable_diffusion/README.md) file.

## Configs

All configuration files are placed under the [`configs/stable_diffusion`](https://github.com/okotaku/diffengine/blob/main/diffengine/configs/stable_diffusion) folder.

Following is the example config from the stable_diffusion_v15_pokemon_blip config file in [`configs/stable_diffusion/stable_diffusion_v15_pokemon_blip.py`](https://github.com/okotaku/diffengine/blob/main/diffengine/configs/stable_diffusion/stable_diffusion_v15_pokemon_blip.py):

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15 import *
    from .._base_.schedules.stable_diffusion_50e import *
```

#### Finetuning the text encoder and UNet

The script also allows you to finetune the text_encoder along with the unet.

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15 import *
    from .._base_.schedules.stable_diffusion_50e import *

model.update(finetune_text_encoder=True)  # fine tune text encoder
```

#### Finetuning with Unet EMA

The script also allows you to finetune with Unet EMA.

```
from mmengine.config import read_base
from diffengine.engine.hooks import EMAHook

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15 import *
    from .._base_.schedules.stable_diffusion_50e import *

custom_hooks = [  # Hook is list, we should write all custom_hooks again.
    dict(type=VisualizationHook, prompt=['yoda pokemon'] * 4),
    dict(type=CheckpointHook),
    dict(type=EMAHook, ema_key="unet", momentum=1e-4, priority='ABOVE_NORMAL')  # setup EMA Hook
]
```

#### Finetuning with other losses

The script also allows you to finetune with [Min-SNR Weighting Strategy](https://arxiv.org/abs/2303.09556).

```
from mmengine.config import read_base
from diffengine.models.losses import SNRL2Loss

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15 import *
    from .._base_.schedules.stable_diffusion_50e import *

model.update(loss=dict(type=SNRL2Loss, snr_gamma=5.0, loss_weight=1.0))  # setup Min-SNR Weighting Strategy
```

#### Finetuning with other noises

The script also allows you to finetune with [OffsetNoise](https://www.crosslabs.org/blog/diffusion-with-offset-noise).

```
from mmengine.config import read_base
from diffengine.models.utils import OffsetNoise

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15 import *
    from .._base_.schedules.stable_diffusion_50e import *

model.update(noise_generator=dict(type=OffsetNoise, offset_weight=0.05))  # setup OffsetNoise
```

#### Finetuning with other timesteps

The script also allows you to finetune with EarlierTimeSteps.

```
from mmengine.config import read_base
from diffengine.models.utils import EarlierTimeSteps

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15 import *
    from .._base_.schedules.stable_diffusion_50e import *

model.update(timesteps_generator=dict(type=EarlierTimeSteps))  # setup EarlierTimeSteps
```

#### Finetuning with pre-computed text embeddings

The script also allows you to finetune with pre-computed text embeddings.

```
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.pokemon_blip_pre_compute import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15 import *
    from .._base_.schedules.stable_diffusion_50e import *

model.update(pre_compute_text_embeddings=True)
```

## Run training

Run train

```
# single gpu
$ diffengine train ${CONFIG_FILE}
# Example
$ diffengine train stable_diffusion_v15_pokemon_blip

# multi gpus
$ NPROC_PER_NODE=${GPU_NUM} diffengine train ${CONFIG_FILE}
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

## Convert weights for diffusers format

You can convert weights for diffusers format. The converted weights will be saved in the specified directory.

```bash
$ diffengine convert ${CONFIG_FILE} ${INPUT_FILENAME} ${OUTPUT_DIR} --save-keys ${SAVE_KEYS}
# Example
$ diffengine convert stable_diffusion_v15_pokemon_blip work_dirs/stable_diffusion_v15_pokemon_blip/epoch_50.pth work_dirs/stable_diffusion_v15_pokemon_blip --save-keys unet
```
