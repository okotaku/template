# flake8: noqa: TRY004,S311
import functools
import gc
import io
import os
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from datasets.fingerprint import Hasher
from mmengine.dataset.base_dataset import Compose
from mmengine.registry import MODELS
from PIL import Image
from torch.utils.data import Dataset

from diffengine.datasets.utils import encode_prompt

Image.MAX_IMAGE_PIXELS = 1000000000


class HFDPODataset(Dataset):
    """DPO Dataset for huggingface datasets.

    Args:
    ----
        dataset (str): Dataset name or path to dataset.
        image_columns (list[str]): Image column names. Defaults to ['image'].
        caption_column (str): Caption column name. Defaults to 'text'.
        label_column (str): Label column name of whether image_columns[0] is
            better than image_columns[1]. Defaults to 'label_0'.
        csv (str): Caption csv file name when loading local folder.
            Defaults to 'metadata.csv'.
        pipeline (Sequence): Processing pipeline. Defaults to an empty tuple.
        split (str): Dataset split. Defaults to 'train'.
        cache_dir (str, optional): The directory where the downloaded datasets
            will be stored.Defaults to None.

    """

    def __init__(self,
                 dataset: str,
                 image_columns: list[str] | None = None,
                 caption_column: str = "text",
                 label_column: str = "label_0",
                 csv: str = "metadata.csv",
                 pipeline: Sequence = (),
                 split: str = "train",
                 cache_dir: str | None = None) -> None:
        if image_columns is None:
            image_columns = ["image", "image2"]
        self.dataset_name = dataset
        if Path(dataset).exists():
            # load local folder
            data_file = os.path.join(dataset, csv)
            self.dataset = load_dataset(
                "csv", data_files=data_file, cache_dir=cache_dir)[split]
        else:
            # load huggingface online
            self.dataset = load_dataset(dataset, cache_dir=cache_dir)[split]
        self.pipeline = Compose(pipeline)

        self.image_columns = image_columns
        self.caption_column = caption_column
        self.label_column = label_column

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
        images = []
        for image_column in self.image_columns:
            image = data_info[image_column]
            if isinstance(image, str):
                image = Image.open(os.path.join(self.dataset_name, image))
            elif not isinstance(image, Image.Image):
                image = Image.open(io.BytesIO(image))
            image = image.convert("RGB")
            images.append(image)
        label = data_info[self.label_column]
        if not label:
            images = images[::-1]
        caption = data_info[self.caption_column]
        if isinstance(caption, str):
            pass
        elif isinstance(caption, list | np.ndarray):
            # take a random caption if there are multiple
            caption = random.choice(caption)
        else:
            msg = (f"Caption column `{self.caption_column}` should "
                   "contain either strings or lists of strings.")
            raise ValueError(msg)
        result = {"img": images, "text": caption}
        return self.pipeline(result)


class HFDPODatasetPreComputeEmbs(HFDPODataset):
    """DPO Dataset for huggingface datasets.

    The difference from HFDataset is
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
                 text_hasher: str = "text",
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

        new_fingerprint = Hasher.hash(text_hasher)
        compute_embeddings_fn = functools.partial(
            encode_prompt,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            caption_column=self.caption_column,
        )
        self.dataset = self.dataset.map(
            compute_embeddings_fn,
            batched=True,
            batch_size=32,
            new_fingerprint=new_fingerprint)
        self.empty_embed = encode_prompt(
            {"text": [""]},
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
        images = []
        for image_column in self.image_columns:
            image = data_info[image_column]
            if isinstance(image, str):
                image = Image.open(os.path.join(self.dataset_name, image))
            elif not isinstance(image, Image.Image):
                image = Image.open(io.BytesIO(image))
            image = image.convert("RGB")
            images.append(image)
        label = data_info[self.label_column]
        if not label:
            images = images[::-1]
        result = {
            "img": images,
            "prompt_embeds": data_info["prompt_embeds"] if (
                random.random() < self.proportion_empty_prompts
                ) else self.empty_embed["prompt_embeds"][0],
        }
        return self.pipeline(result)
