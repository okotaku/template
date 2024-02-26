# flake8: noqa: S311,RUF012
import gc
import os
import random
from collections.abc import Sequence
from pathlib import Path

import torch
from datasets import load_dataset
from mmengine.dataset.base_dataset import Compose
from mmengine.registry import MODELS
from PIL import Image
from torch.utils.data import Dataset

from diffengine.datasets.utils import encode_prompt

Image.MAX_IMAGE_PIXELS = 1000000000


class HFDreamBoothDataset(Dataset):
    """DreamBooth Dataset for huggingface datasets.

    Args:
    ----
        dataset (str): Dataset name.
        instance_prompt (str):
            The prompt with identifier specifying the instance.
        image_column (str): Image column name. Defaults to 'image'.
        dataset_sub_dir (optional, str): Dataset sub directory name.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        csv (str, optional): Image path csv file name when loading local
            folder. If None, the dataset will be loaded from image folders.
            Defaults to None.
        cache_dir (str, optional): The directory where the downloaded datasets
            will be stored.Defaults to None.

    """

    def __init__(self,
                 dataset: str,
                 instance_prompt: str,
                 image_column: str = "image",
                 dataset_sub_dir: str | None = None,
                 pipeline: Sequence = (),
                 csv: str | None = None,
                 cache_dir: str | None = None) -> None:

        self.dataset_name = dataset
        self.csv = csv

        if Path(dataset).exists():
            # load local folder
            if csv is not None:
                data_file = os.path.join(dataset, csv)
                self.dataset = load_dataset(
                    "csv", data_files=data_file, cache_dir=cache_dir)["train"]
            else:
                self.dataset = load_dataset(dataset, cache_dir=cache_dir)["train"]
        else:  # noqa
            # load huggingface online
            if dataset_sub_dir is not None:
                self.dataset = load_dataset(
                    dataset, dataset_sub_dir, cache_dir=cache_dir)["train"]
            else:
                self.dataset = load_dataset(
                    dataset, cache_dir=cache_dir)["train"]

        self.pipeline = Compose(pipeline)

        self.instance_prompt = instance_prompt
        self.image_column = image_column

    def __len__(self) -> int:
        """Get the length of dataset.

        Returns
        -------
            int: The length of filtered dataset.

        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """Get item.

        Get the idx-th image and data information of dataset after
        ``self.pipeline`.

        Args:
        ----
            idx (int): The index of self.data_list.

        Returns:
        -------
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.

        """
        data_info = self.dataset[idx]
        image = data_info[self.image_column]
        if isinstance(image, str):
            if self.csv is not None:
                image = os.path.join(self.dataset_name, image)
            image = Image.open(image)
        image = image.convert("RGB")
        result = {"img": image, "text": self.instance_prompt}
        return self.pipeline(result)


class HFDreamBoothDatasetPreComputeEmbs(HFDreamBoothDataset):
    """DreamBooth Dataset for huggingface datasets.

    The difference from DreamBooth is
        1. pre-compute Text Encoder embeddings to save memory.

    Args:
    ----
        tokenizer (dict): Config of tokenizer.
        text_encoder (dict): Config of text encoder.
        model (str): pretrained model name of stable diffusion.
            Defaults to 'runwayml/stable-diffusion-v1-5'.
        device (str): Device used to compute embeddings. Defaults to 'cuda'.
        proportion_empty_prompts (float): The probabilities to replace empty
            text. Defaults to 0.0.

    """

    def __init__(self,
                 *args,
                 tokenizer: dict,
                 text_encoder: dict,
                 model: str = "runwayml/stable-diffusion-v1-5",
                 device: str = "cuda",
                 proportion_empty_prompts: float = 0.0,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.proportion_empty_prompts = proportion_empty_prompts

        tokenizer = MODELS.build(
            tokenizer,
            default_args={"pretrained_model_name_or_path": model})
        text_encoder = MODELS.build(
            text_encoder,
            default_args={"pretrained_model_name_or_path": model}).to(device)

        self.embed = encode_prompt(
            {"text": [self.instance_prompt, ""]},
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            caption_column="text",
        )

        del text_encoder, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    def __getitem__(self, idx: int) -> dict:
        """Get item.

        Get the idx-th image and data information of dataset after
        ``self.train_transforms`.

        Args:
        ----
            idx (int): The index of self.data_list.

        Returns:
        -------
            dict: The idx-th image and data information of dataset after
            ``self.train_transforms``.

        """
        data_info = self.dataset[idx]
        image = data_info[self.image_column]
        if isinstance(image, str):
            if self.csv is not None:
                image = os.path.join(self.dataset_name, image)
            image = Image.open(image)
        image = image.convert("RGB")
        result = {
            "img": image,
            "prompt_embeds": self.embed["prompt_embeds"][0] if (
                random.random() < self.proportion_empty_prompts
                ) else self.embed["prompt_embeds"][1],
        }
        return self.pipeline(result)
