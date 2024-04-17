from apex.optimizers import FusedAdam
from mmengine.hooks import CheckpointHook
from mmengine.optim import AmpOptimWrapper
from mmengine.runner import IterBasedTrainLoop

optim_wrapper = dict(
    type=AmpOptimWrapper,
    dtype="bfloat16",
    optimizer=dict(type=FusedAdam, lr=1e-4, weight_decay=1e-2),
    clip_grad=dict(max_norm=1.0),
    accumulative_counts=512)

# train, val, test setting
train_cfg = dict(type=IterBasedTrainLoop, max_iters=200000)
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
