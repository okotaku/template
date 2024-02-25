from mmengine.config import read_base
from mmengine.optim import OptimWrapper

from diffengine.engine.hooks import FastNormHook, SFastHook

with read_base():
    from .._base_.datasets.pokemon_blip import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15 import *
    from .._base_.schedules.stable_diffusion_50e import *

model.update(weight_dtype="bf16")

env_cfg.update(
    cudnn_benchmark=True,
)

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=FusedAdam, lr=1e-5, weight_decay=1e-2),
    clip_grad=dict(max_norm=1.0))


custom_hooks = [
    dict(type=VisualizationHook, prompt=["yoda pokemon"] * 4),
    dict(type=CheckpointHook),
    dict(type=FastNormHook),
    dict(type=SFastHook),
]
