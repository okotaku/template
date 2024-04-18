import torchvision
from mmengine.dataset import DefaultSampler

from diffengine.datasets import HFDPODataset
from diffengine.datasets.transforms import (
    ConcatMultipleImgs,
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    RandomTextDrop,
    TorchVisonTransformWrapper,
)
from diffengine.engine.hooks import (
    CheckpointHook,
    CompileHook,
    MemoryFormatHook,
    VisualizationHook,
)

train_pipeline = [
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Resize,
         size=512, interpolation="bilinear"),
    dict(type=RandomCrop, size=512),
    dict(type=RandomHorizontalFlip, p=0.5),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.ToTensor),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Normalize, mean=[0.5], std=[0.5]),
    dict(type=RandomTextDrop),
    dict(type=ConcatMultipleImgs),
    dict(type=PackInputs),
]
train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type=HFDPODataset,
        dataset="yuvalkirstain/pickapic_v2",
        image_columns=["jpg_0", "jpg_1"],
        caption_column="caption",
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
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
    dict(type=MemoryFormatHook),
    dict(type=CompileHook),
]
