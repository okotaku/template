from mmengine.config import read_base

from diffengine.engine.hooks import (
    CheckpointHook,
    LCMEMAUpdateHook,
    MemoryFormatHook,
    VisualizationHook,
)

with read_base():
    from .._base_.datasets.pokemon_blip_pre_compute import *
    from .._base_.default_runtime import *
    from .._base_.models.lcm_v15 import *
    from .._base_.schedules.stable_diffusion_50e import *

model.update(pre_compute_text_embeddings=True,
             weight_dtype="bf16")

custom_hooks = [
    dict(type=VisualizationHook, prompt=["yoda pokemon"] * 4),
    dict(type=CheckpointHook),
    dict(type=MemoryFormatHook),
    dict(type=LCMEMAUpdateHook),
]
