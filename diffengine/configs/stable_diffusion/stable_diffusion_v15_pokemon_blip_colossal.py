from mmengine._strategy import ColossalAIStrategy
from mmengine.config import read_base
from mmengine.runner import FlexibleRunner

with read_base():
    from .._base_.datasets.pokemon_blip_baseline import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15 import *
    from .._base_.schedules.stable_diffusion_50e_baseline import *

default_hooks.update(
    checkpoint=dict(save_param_scheduler=False))  # no scheduler in this config

runner_type = FlexibleRunner
strategy = dict(type=ColossalAIStrategy,
                mixed_precision="fp16",
                plugin=dict(type="LowLevelZeroPlugin",
                            stage=2,
                            max_norm=1.0))
