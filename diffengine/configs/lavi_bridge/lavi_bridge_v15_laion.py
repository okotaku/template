from mmengine.config import read_base
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor

from diffengine.engine.hooks import PeftSaveHook

with read_base():
    from .._base_.datasets.laion import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15_lavi_bridge import *
    from .._base_.schedules.stable_diffusion_20k_baseline import *

model.update(data_preprocessor=dict(type=BaseDataPreprocessor))

default_hooks.update(dict(checkpoint=dict(save_optimizer=False)))

custom_hooks = [
    dict(type=VisualizationHook,
         prompt=["Two cats playing chess on a tree branch",
                 "A monk in an orange robe by a round window in a spaceship in dramatic lighting.",  # noqa
                 "Concept art of a mythical sky alligator with wings, nature documentary.",  # noqa
                 "A galaxy-colored figurine is floating over the sea at sunset, photorealistic."],  # noqa
        by_epoch=False,
        width=512,
        height=512,
        interval=10000),
    dict(type=PeftSaveHook),
]
