from mmengine.config import read_base

from diffengine.engine.hooks import CompileHook, FastNormHook

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15 import *
    from .._base_.schedules.stable_diffusion_50e import *

env_cfg.update(
    cudnn_benchmark=True,
)

custom_hooks = [
    dict(type=VisualizationHook, prompt=["yoda pokemon"] * 4),
    dict(type=CheckpointHook),
    dict(type=FastNormHook),
    dict(type=CompileHook),
]
