# flake8: noqa: S311,RUF012
import os
from collections.abc import Sequence
from pathlib import Path

from datasets import load_dataset
from mmengine.dataset.base_dataset import Compose
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = 1000000000


class ImageHubDreamBoothDataset(Dataset):
    """DreamBooth Dataset for ImageHub datasets.

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
                 subject: str,
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

        # pickup subject data
        ds = [data for data in self.dataset if data["subject"] == subject]
        self.dataset = ds

        self.pipeline = Compose(pipeline)

        self.subject = subject
        self.identifier = self.dataset[0]["identifier"]
        subject_prompt = subject.replace("_", " ")
        self.instance_prompt = f"A {self.identifier} {subject_prompt}"
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
