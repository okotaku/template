from typing import Optional

import mmengine
import numpy as np
from datasets import load_dataset
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from PIL import Image

from diffengine.evaluation import CLIPT, DINOScore


class ImageHubVisualizationHook(Hook):
    """Basic hook that invoke visualizers after train epoch.

    Args:
    ----
        dataset (str): The dataset name.
            Defaults to "ImagenHub/Subject_Driven_Image_Generation".
        interval (int): Visualization interval (every k iterations).
            Defaults to 1.
        by_epoch (bool): Whether to visualize by epoch. Defaults to True.
        height (int): The height in pixels of the generated image.
            Defaults to 512.
        width (int): The width in pixels of the generated image.
            Defaults to 512.

    """

    priority = "NORMAL"

    def __init__(self,
                 dataset: str = "ImagenHub/Subject_Driven_Image_Generation",
                 interval: int = 1,
                 height: int = 512,
                 width: int = 512,
                 *,
                 by_epoch: bool = True,
                 **kwargs) -> None:
        self.dataset = load_dataset(dataset)["eval"]
        self.kwargs = kwargs
        self.interval = interval
        self.by_epoch = by_epoch
        self.height = height
        self.width = width

        self.clipt = CLIPT()

    def _visualize_and_eval(self, runner: Runner, step: int,
                            suffix: str = "step") -> None:
        """Visualize and evaluate."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        images = model.infer(
            self.prompt,
            height=self.height,
            width=self.width,
            **self.kwargs)
        for i, image in enumerate(images):
            runner.visualizer.add_image(
                f"image{i}_{suffix}", image, step=step)

        # Evaluate
        clipt_scores = []
        dino_scores = []
        for image, text in zip(images, self.prompt, strict=True):
            clipt_scores.append(
                self.clipt(Image.fromarray(image),
                           text.replace(f"{self.identifier} ", "")))
            dino_scores.append(self.dino_score(Image.fromarray(image)))
        clip_score_text = f"CLIPT is {np.mean(clipt_scores)}"
        runner.logger.info(clip_score_text)
        dino_score_text = f"DINOScore is {np.mean(dino_scores)}"
        runner.logger.info(dino_score_text)
        mmengine.dump({
            "CLIPT": np.mean(clipt_scores),
            "DINOScore": np.mean(dino_scores),
        }, f"{runner.work_dir}/scores.json")

    def before_train(self, runner: Runner) -> None:
        """Before train hook."""
        # pickup test captions for target subject
        self.subject = runner.train_dataloader.dataset.subject
        self.identifier = runner.train_dataloader.dataset.identifier
        image_column = runner.train_dataloader.dataset.image_column
        prompt = [
            data["prompt"].replace(
                "<token>", self.identifier,
            ) for data in self.dataset if data["subject"] == self.subject]
        self.prompt = prompt
        msg = f"prompt is {self.prompt}"
        runner.logger.info(msg)

        imgs = [d[image_column] for d in runner.train_dataloader.dataset.dataset]
        self.dino_score = DINOScore(imgs)

        self._visualize_and_eval(runner, runner.iter, suffix="before_train")

    def after_train_iter(
            self,
            runner: Runner,
            batch_idx: int,
            data_batch: DATA_BATCH = None,  # noqa
            outputs: Optional[dict] = None) -> None:  # noqa
        """After train iter hook.

        Args:
        ----
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch.
            data_batch (DATA_BATCH, optional): The current data batch.
            outputs (dict, optional): The outputs of the current batch.

        """
        if self.by_epoch:
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            self._visualize_and_eval(runner, runner.iter)

    def after_train_epoch(self, runner: Runner) -> None:
        """After train epoch hook.

        Args:
        ----
            runner (Runner): The runner of the training process.

        """
        if self.by_epoch:
            self._visualize_and_eval(runner, runner.epoch)
