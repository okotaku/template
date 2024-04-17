from diffengine.datasets import DALILAIONIterator
from diffengine.engine.hooks import CheckpointHook, VisualizationHook

train_dataloader = dict(
    type=DALILAIONIterator,
    batch_size=64,
    num_workers=16,
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

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
    dict(type=CheckpointHook),
]
