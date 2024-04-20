from mmengine.hooks import CheckpointHook
from mmengine.optim import OptimWrapper
from mmengine.runner import IterBasedTrainLoop
from optimi import AdamW

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=1e-4, weight_decay=1e-2),
    clip_grad=dict(max_norm=1.0))

# train, val, test setting
train_cfg = dict(type=IterBasedTrainLoop, max_iters=1000)
val_cfg = None
test_cfg = None

default_hooks = dict(
    checkpoint=dict(
        type=CheckpointHook,
        interval=100,
        by_epoch=False,
        max_keep_ckpts=3,
        save_optimizer=True,
    ))
log_processor = dict(by_epoch=False)
