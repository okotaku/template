import random
from collections.abc import Sequence

import numpy as np
import torch
import torchvision
from mmengine.dataset.base_dataset import Compose
from torchvision.transforms.functional import crop
from torchvision.transforms.transforms import InterpolationMode

from diffengine.datasets.transforms.base import BaseTransform


def _str_to_torch_dtype(t: str):  # noqa
    """Map to torch.dtype."""
    import torch  # noqa: F401
    return eval(f"torch.{t}")  # noqa


def _interpolation_modes_from_str(t: str):  # noqa
    """Map to Interpolation."""
    t = t.lower()
    inverse_modes_mapping = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "box": InterpolationMode.BOX,
        "hammimg": InterpolationMode.HAMMING,
        "lanczos": InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[t]


class TorchVisonTransformWrapper:
    """TorchVisonTransformWrapper.

    We can use torchvision.transforms like `dict(type='torchvision/Resize',
    size=512)`

    Args:
    ----
        transform (str): The name of transform. For example
            `torchvision/Resize`.
        keys (List[str]): `keys` to apply augmentation from results.

    """

    def __init__(self,
                 transform,  # noqa
                 *args,
                 keys: list[str] | None = None,
                 **kwargs) -> None:
        if keys is None:
            keys = ["img"]
        self.keys = keys
        if "interpolation" in kwargs and isinstance(kwargs["interpolation"],
                                                    str):
            kwargs["interpolation"] = _interpolation_modes_from_str(
                kwargs["interpolation"])
        if "dtype" in kwargs and isinstance(kwargs["dtype"], str):
            kwargs["dtype"] = _str_to_torch_dtype(kwargs["dtype"])
        self.t = transform(*args, **kwargs)

    def __call__(self, results: dict) -> dict:
        """Call transform."""
        for k in self.keys:
            if not isinstance(results[k], list):
                results[k] = self.t(results[k])
            else:
                results[k] = [self.t(img) for img in results[k]]
        return results

    def __repr__(self) -> str:
        """Repr."""
        return f"TorchVision{self.t!r}"


class RandomCrop(BaseTransform):
    """RandomCrop.

    The difference from torchvision/RandomCrop is
        1. save crop top left as 'crop_top_left' and `crop_bottom_right` in
        results
        2. apply same random parameters to multiple `keys` like ['img',
        'condition_img'].

    Args:
    ----
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted
            as (size[0], size[0])
        keys (List[str]): `keys` to apply augmentation from results.
        force_same_size (bool): Force same size for all keys. Defaults to True.

    """

    def __init__(self,
                 *args,
                 size: Sequence[int] | int,
                 keys: list[str] | None = None,
                 force_same_size: bool = True,
                 **kwargs) -> None:
        if keys is None:
            keys = ["img"]
        if not isinstance(size, Sequence):
            size = (size, size)
        self.size = size
        self.keys = keys
        self.force_same_size = force_same_size
        self.pipeline = torchvision.transforms.RandomCrop(
            *args, size, **kwargs)

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.

        Returns:
        -------
            dict: 'crop_top_left' and  `crop_bottom_right` key is added as crop
                point.

        """
        components = dict()
        for k in self.keys:
            if not isinstance(results["img"], list):
                components[k] = [results[k]]
            else:
                components[k] = results[k]

        crop_top_left = []
        crop_bottom_right = []
        before_crop_size = []
        for i in range(len(components["img"])):
            if self.force_same_size:
                assert all(
                    components["img"][i].size == components[k][i].size
                    for k in self.keys), (
                    "Size mismatch."
                )
            before_crop_size.append([components["img"][i].height,
                                     components["img"][i].width])

            y1, x1, h, w = self.pipeline.get_params(components["img"][i],
                                                    self.size)
            for k in self.keys:
                components[k][i] = crop(components[k][i], y1, x1, h, w)
            crop_top_left.append([y1, x1])
            crop_bottom_right.append([y1 + h, x1 + w])

        if not isinstance(results["img"], list):
            for k in self.keys:
                components[k] = components[k][0]
            crop_top_left = crop_top_left[0]
            crop_bottom_right = crop_bottom_right[0]
            before_crop_size = before_crop_size[0]

        for k in self.keys:
            results[k] = components[k]

        results["crop_top_left"] = crop_top_left
        results["crop_bottom_right"] = crop_bottom_right
        results["before_crop_size"] = before_crop_size
        return results


class CenterCrop(BaseTransform):
    """CenterCrop.

    The difference from torchvision/CenterCrop is
        1. save crop top left as 'crop_top_left' and `crop_bottom_right` in
        results

    Args:
    ----
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted
            as (size[0], size[0])
        keys (List[str]): `keys` to apply augmentation from results.

    """

    def __init__(self,
                 *args,
                 size: Sequence[int] | int,
                 keys: list[str] | None = None,
                 **kwargs) -> None:
        if keys is None:
            keys = ["img"]
        if not isinstance(size, Sequence):
            size = (size, size)
        self.size = size
        self.keys = keys
        self.pipeline = torchvision.transforms.CenterCrop(
            *args, size, **kwargs)

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.

        Returns:
        -------
            dict: 'crop_top_left' key is added as crop points.

        """
        components = dict()
        for k in self.keys:
            if not isinstance(results["img"], list):
                components[k] = [results[k]]
            else:
                components[k] = results[k]

        crop_top_left: list = []
        crop_bottom_right: list = []
        before_crop_size: list = []
        for i in range(len(components["img"])):
            assert all(
                components["img"][i].size == components[k][i].size
                for k in self.keys), (
                "Size mismatch."
            )
            before_crop_size.append([components["img"][i].height,
                                     components["img"][i].width])

            y1 = max(0, int(round(
                (components["img"][i].height - self.size[0]) / 2.0)))
            x1 = max(0, int(round(
                (components["img"][i].width - self.size[1]) / 2.0)))
            y2 = max(0, int(round(
                (components["img"][i].height + self.size[0]) / 2.0)))
            x2 = max(0, int(round(
                (components["img"][i].width + self.size[1]) / 2.0)))
            for k in self.keys:
                components[k][i] = self.pipeline(components[k][i])
            crop_top_left.append([y1, x1])
            crop_bottom_right.append([y2, x2])

        if not isinstance(results["img"], list):
            for k in self.keys:
                components[k] = components[k][0]
            crop_top_left = crop_top_left[0]
            crop_bottom_right = crop_bottom_right[0]
            before_crop_size = before_crop_size[0]

        for k in self.keys:
            results[k] = components[k]
        results["crop_top_left"] = crop_top_left
        results["crop_bottom_right"] = crop_bottom_right
        results["before_crop_size"] = before_crop_size
        return results


class MultiAspectRatioResizeCenterCrop(BaseTransform):
    """Multi Aspect Ratio Resize and Center Crop.

    Args:
    ----
        sizes (List[sequence]): List of desired output size of the crop.
            Sequence like (h, w).
        keys (List[str]): `keys` to apply augmentation from results.
        interpolation (str): Desired interpolation enum defined by
            torchvision.transforms.InterpolationMode.
            Defaults to 'bilinear'.

    """

    def __init__(
            self,
            *args,  # noqa
            sizes: list[Sequence[int]],
            keys: list[str] | None = None,
            interpolation: str = "bilinear",
            **kwargs) -> None:  # noqa
        if keys is None:
            keys = ["img"]
        self.sizes = sizes
        self.aspect_ratios = np.array([s[0] / s[1] for s in sizes])
        self.pipelines = []
        for s in self.sizes:
            self.pipelines.append(
                Compose([
                    TorchVisonTransformWrapper(
                        torchvision.transforms.Resize,
                        size=min(s),
                        interpolation=interpolation,
                        keys=keys),
                    CenterCrop(size=s, keys=keys),
                ]))

    def transform(self, results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.

        """
        assert not isinstance(results["img"], list), (
            "MultiAspectRatioResizeCenterCrop only support single image.")
        aspect_ratio = results["img"].height / results["img"].width
        bucked_id = np.argmin(np.abs(aspect_ratio - self.aspect_ratios))
        return self.pipelines[bucked_id](results)


class RandomHorizontalFlip(BaseTransform):
    """RandomHorizontalFlip.

    The difference from torchvision/RandomHorizontalFlip is
        1. update 'crop_top_left' and `crop_bottom_right` if exists.
        2. apply same random parameters to multiple `keys` like ['img',
        'condition_img'].

    Args:
    ----
        p (float): probability of the image being flipped.
            Default value is 0.5.
        keys (List[str]): `keys` to apply augmentation from results.

    """

    def __init__(self, *args, p: float = 0.5,
                 keys: list[str] | None = None, **kwargs) -> None:
        if keys is None:
            keys = ["img"]
        self.p = p
        self.keys = keys
        self.pipeline = torchvision.transforms.RandomHorizontalFlip(
            *args, p=1.0, **kwargs)

    def transform(self, results: dict) -> dict | tuple[list, list] | None:  # noqa: C901,PLR0912
        """Transform.

        Args:
        ----
            results (dict): The result dict.

        Returns:
        -------
            dict: 'crop_top_left' key is fixed.

        """
        components = dict()
        additional_keys = [
            "crop_top_left", "crop_bottom_right", "before_crop_size",
            ] if "crop_top_left" in results else []
        for k in self.keys + additional_keys:
            if not isinstance(results["img"], list):
                components[k] = [results[k]]
            else:
                components[k] = results[k]

        crop_top_left = []
        for i in range(len(components["img"])):
            if random.random() < self.p:
                assert all(components["img"][i].size == components[k][i].size
                        for k in self.keys)
                for k in self.keys:
                    components[k][i] = self.pipeline(components[k][i])
                if "crop_top_left" in results:
                    y1 = components["crop_top_left"][i][0]
                    x1 = (
                        components["before_crop_size"][i][1] - components[
                            "crop_bottom_right"][i][1])
                    crop_top_left.append([y1, x1])
            elif "crop_top_left" in results:
                crop_top_left.append(components["crop_top_left"][i])

        if not isinstance(results["img"], list):
            for k in self.keys:
                components[k] = components[k][0]
            if "crop_top_left" in results:
                crop_top_left = crop_top_left[0]

        for k in self.keys:
            results[k] = components[k]
        if "crop_top_left" in results:
            results["crop_top_left"] = crop_top_left
        return results


class ConcatMultipleImgs(BaseTransform):
    """ConcatMultipleImgs.

    Args:
    ----
        keys (List[str], optional): `keys` to apply augmentation from results.
            Defaults to None.

    """

    def __init__(self, keys: list[str] | None = None) -> None:
        if keys is None:
            keys = ["img"]
        self.keys = keys

    def transform(self,
                  results: dict) -> dict | tuple[list, list] | None:
        """Transform.

        Args:
        ----
            results (dict): The result dict.

        """
        for k in self.keys:
            results[k] = torch.cat(results[k], dim=0)
        return results
