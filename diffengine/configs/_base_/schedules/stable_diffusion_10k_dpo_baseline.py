from apex.optimizers import FusedAdam
from mmengine.hooks import CheckpointHook
from mmengine.optim import AmpOptimWrapper
from mmengine.optim.scheduler import CosineAnnealingLR, LinearLR
from mmengine.runner import IterBasedTrainLoop

optim_wrapper = dict(
    type=AmpOptimWrapper,
    dtype="bfloat16",
    optimizer=dict(type=FusedAdam, lr=5e-5, weight_decay=1e-2),
    clip_grad=dict(max_norm=1.0),
    accumulative_counts=128)

param_scheduler = [
    dict(type=LinearLR, start_factor=0.01, by_epoch=False, begin=0, end=128 * 10),
    # Use a cosine learning rate at [100, 900) iterations
    dict(
        type=CosineAnnealingLR,
        T_max=128 * 1000 - 128 * 10,
        by_epoch=False,
        begin=128 * 10,
        end=128 * 1000),
]

# train, val, test setting
train_cfg = dict(type=IterBasedTrainLoop, max_iters=128 * 1000)
val_cfg = None
test_cfg = None

default_hooks = dict(
    checkpoint=dict(
        type=CheckpointHook,
        interval=10000,
        by_epoch=False,
        max_keep_ckpts=3,
        save_optimizer=True,
    ))
log_processor = dict(by_epoch=False)
