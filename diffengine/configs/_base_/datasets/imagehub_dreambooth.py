import torchvision
from mmengine.dataset import InfiniteSampler

from diffengine.datasets import ImageHubDreamBoothDataset
from diffengine.datasets.transforms import (
    DumpImage,
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
    TorchVisonTransformWrapper,
)
from diffengine.engine.hooks import (
    CompileHook,
    ImageHubVisualizationHook,
    MemoryFormatHook,
    PeftSaveHook,
)

train_pipeline = [
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Resize,
         size=512, interpolation="bilinear"),
    dict(type=RandomCrop, size=512),
    dict(type=RandomHorizontalFlip, p=0.5),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.ToTensor),
    dict(type=DumpImage, max_imgs=10, dump_dir="work_dirs/dump"),
    dict(type=TorchVisonTransformWrapper,
         transform=torchvision.transforms.Normalize, mean=[0.5], std=[0.5]),
    dict(type=PackInputs),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=ImageHubDreamBoothDataset,
        dataset="ImagenHub/DreamBooth_Concepts",
        subject="dog",
        pipeline=train_pipeline),
    sampler=dict(type=InfiniteSampler, shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type=ImageHubVisualizationHook,
        by_epoch=False,
        interval=100),
    dict(type=PeftSaveHook),
    dict(type=MemoryFormatHook),
    dict(type=CompileHook),
]
