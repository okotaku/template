import torchvision
from mmengine.dataset import DefaultSampler
from transformers import CLIPTextModel, CLIPTokenizer

from diffengine.datasets import HFDatasetPreComputeEmbs
from diffengine.datasets.transforms import (
    PackInputs,
    RandomCrop,
    RandomHorizontalFlip,
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
    dict(type=PackInputs, input_keys=["img", "prompt_embeds"]),
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type=HFDatasetPreComputeEmbs,
        dataset="diffusers/pokemon-gpt4-captions",
        text_hasher="text_pokemon_blip_v1-5",
        model="runwayml/stable-diffusion-v1-5",
        tokenizer=dict(type=CLIPTokenizer.from_pretrained,
                    subfolder="tokenizer"),
        text_encoder=dict(type=CLIPTextModel.from_pretrained,
                        subfolder="text_encoder"),
        proportion_empty_prompts=0.1,
        pipeline=train_pipeline),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = None
val_evaluator = None
test_dataloader = val_dataloader
test_evaluator = val_evaluator

custom_hooks = [
    dict(type=VisualizationHook, prompt=["yoda pokemon"] * 4),
    dict(type=CheckpointHook),
    dict(type=MemoryFormatHook),
    dict(type=CompileHook),
]
