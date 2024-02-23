import copy
from os import path as osp

import cv2
import mmengine
import numpy as np
import torch
from torch.multiprocessing import Value


class DumpImage:
    """Dump the image processed by the pipeline.

    Args:
    ----
        max_imgs (int): Maximum value of output.
        dump_dir (str): Dump output directory.

    """

    def __init__(self, max_imgs: int, dump_dir: str) -> None:
        self.max_imgs = max_imgs
        self.dump_dir = dump_dir
        mmengine.mkdir_or_exist(self.dump_dir)
        self.num_dumped_imgs = Value("i", 0)

    def __call__(self, results: dict) -> dict:
        """Dump the input image to the specified directory.

        No changes will be
        made.

        Args:
        ----
            results (dict): Result dict from loading pipeline.

        Returns:
        -------
            results (dict): Result dict from loading pipeline. (same as input)

        """
        enable_dump = False
        with self.num_dumped_imgs.get_lock():
            if self.num_dumped_imgs.value < self.max_imgs:
                self.num_dumped_imgs.value += 1
                enable_dump = True
                dump_id = self.num_dumped_imgs.value

        if enable_dump:
            img = copy.deepcopy(results["img"])
            if img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0) * 255
            out_file = osp.join(self.dump_dir, f"{dump_id}_image.png")
            cv2.imwrite(out_file, img.numpy().astype(np.uint8)[..., ::-1])

            if "condition_img" in results:
                condition_img = results["condition_img"]
                if condition_img.shape[0] in [1, 3]:
                    condition_img = condition_img.permute(1, 2, 0) * 255
                cond_out_file = osp.join(self.dump_dir, f"{dump_id}_cond.png")
                cv2.imwrite(cond_out_file,
                            condition_img.numpy().astype(np.uint8)[..., ::-1])

            if "clip_img" in results:
                clip_img = results["clip_img"]
                if clip_img.shape[0] in [1, 3]:
                    clip_img = (
                        clip_img.permute(1, 2, 0) * torch.Tensor(
                            [0.26862954, 0.26130258, 0.27577711])
                        + torch.Tensor([0.48145466, 0.4578275, 0.40821073])) * 255
                clip_out_file = osp.join(self.dump_dir, f"{dump_id}_clip.png")
                cv2.imwrite(clip_out_file,
                            clip_img.numpy().astype(np.uint8)[..., ::-1])

            if "mask" in results:
                mask = results["mask"]
                if mask.shape[0] in [1, 3]:
                    mask = mask.permute(1, 2, 0) * 255
                mask_out_file = osp.join(self.dump_dir, f"{dump_id}_mask.png")
                cv2.imwrite(mask_out_file,
                            mask.numpy().astype(np.uint8))

        return results


class DumpMaskedImage:
    """Dump Masked the image processed by the pipeline.

    Args:
    ----
        max_imgs (int): Maximum value of output.
        dump_dir (str): Dump output directory.

    """

    def __init__(self, max_imgs: int, dump_dir: str) -> None:
        self.max_imgs = max_imgs
        self.dump_dir = dump_dir
        mmengine.mkdir_or_exist(self.dump_dir)
        self.num_dumped_imgs = Value("i", 0)

    def __call__(self, results: dict) -> dict:
        """Dump the input image to the specified directory.

        No changes will be
        made.

        Args:
        ----
            results (dict): Result dict from loading pipeline.

        Returns:
        -------
            results (dict): Result dict from loading pipeline. (same as input)

        """
        enable_dump = False
        with self.num_dumped_imgs.get_lock():
            if self.num_dumped_imgs.value < self.max_imgs:
                self.num_dumped_imgs.value += 1
                enable_dump = True
                dump_id = self.num_dumped_imgs.value

        if enable_dump:
            masked_image = results["masked_image"]
            masked_image = (masked_image / 2 + 0.5).clamp(0, 1)
            if masked_image.shape[0] in [1, 3]:
                masked_image = masked_image.permute(1, 2, 0) * 255
            masked_image_out_file = osp.join(
                self.dump_dir, f"{dump_id}_masked_image.png")
            cv2.imwrite(masked_image_out_file,
                        masked_image.numpy().astype(np.uint8)[..., ::-1])

        return results
