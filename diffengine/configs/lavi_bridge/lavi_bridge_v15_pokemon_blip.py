from mmengine.config import read_base

from diffengine.engine.hooks import PeftSaveHook

with read_base():
    from .._base_.datasets.pokemon_blip_baseline import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15_lavi_bridge import *
    from .._base_.schedules.stable_diffusion_50e_baseline import *

train_dataloader.update(dict(batch_size=2))

default_hooks.update(dict(checkpoint=dict(save_optimizer=False)))

custom_hooks = [
    dict(type=VisualizationHook, prompt=["yoda pokemon"] * 4),
    dict(type=PeftSaveHook),
]
