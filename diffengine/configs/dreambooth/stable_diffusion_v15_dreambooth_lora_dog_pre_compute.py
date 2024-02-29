from mmengine.config import read_base

with read_base():
    from .._base_.datasets.dog_dreambooth_pre_compute import *
    from .._base_.default_runtime import *
    from .._base_.models.stable_diffusion_v15_lora import *
    from .._base_.schedules.stable_diffusion_1k import *

model.update(pre_compute_text_embeddings=True,
             weight_dtype="bf16")
