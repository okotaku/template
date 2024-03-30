import os.path as osp
from unittest import TestCase

import numpy as np
import pytest
import torch
import torchvision
from mmengine.dataset.base_dataset import Compose
from mmengine.registry import TRANSFORMS
from mmengine.utils import digit_version
from PIL import Image
from torchvision import transforms

from diffengine.datasets import (
    CenterCrop,
    ConcatMultipleImgs,
    MultiAspectRatioResizeCenterCrop,
    RandomCrop,
    RandomHorizontalFlip,
    TorchVisonTransformWrapper,
)


class TestVisionTransformWrapper(TestCase):

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {"img": Image.open(img_path)}

        # test normal transform
        vision_trans = transforms.RandomResizedCrop(224)
        vision_transformed_img = vision_trans(data["img"])
        trans = TRANSFORMS.build(
            dict(type=TorchVisonTransformWrapper,
                 transform=torchvision.transforms.RandomResizedCrop,
                 size=224))
        transformed_img = trans(data)["img"]
        np.equal(np.array(vision_transformed_img), np.array(transformed_img))

        # test convert type dtype
        data = {"img": torch.randn(3, 224, 224)}
        vision_trans = transforms.ConvertImageDtype(torch.float)
        vision_transformed_img = vision_trans(data["img"])
        trans = TRANSFORMS.build(
            dict(type=TorchVisonTransformWrapper,
                 transform=torchvision.transforms.ConvertImageDtype,
                 dtype="float"))
        transformed_img = trans(data)["img"]
        np.equal(np.array(vision_transformed_img), np.array(transformed_img))

        # test transform with interpolation
        data = {"img": Image.open(img_path)}
        if digit_version(torchvision.__version__) > digit_version("0.8.0"):
            from torchvision.transforms import InterpolationMode
            interpolation_t = InterpolationMode.NEAREST
        else:
            interpolation_t = Image.NEAREST
        vision_trans = transforms.Resize(224, interpolation_t)
        vision_transformed_img = vision_trans(data["img"])
        trans = TRANSFORMS.build(
            dict(type=TorchVisonTransformWrapper,
                 transform=torchvision.transforms.Resize,
                 size=224, interpolation="nearest"))
        transformed_img = trans(data)["img"]
        np.equal(np.array(vision_transformed_img), np.array(transformed_img))

        # test compose transforms
        data = {"img": Image.open(img_path)}
        vision_trans = transforms.Compose([
            transforms.Resize(176),
            transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        vision_transformed_img = vision_trans(data["img"])

        pipeline_cfg = [
            dict(type=TorchVisonTransformWrapper,
                 transform=torchvision.transforms.Resize,
                 size=176),
            dict(type=RandomHorizontalFlip),
            dict(type=TorchVisonTransformWrapper,
                 transform=torchvision.transforms.PILToTensor),
            dict(type=TorchVisonTransformWrapper,
                 transform=torchvision.transforms.ConvertImageDtype,
                 dtype="float"),
            dict(
                type=TorchVisonTransformWrapper,
                transform=torchvision.transforms.Normalize,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
        pipeline = [TRANSFORMS.build(t) for t in pipeline_cfg]
        pipe = Compose(transforms=pipeline)
        transformed_img = pipe(data)["img"]
        np.equal(np.array(vision_transformed_img), np.array(transformed_img))

class TestRandomCrop(TestCase):
    crop_size = 32

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {"img": Image.open(img_path)}

        # test transform
        trans = TRANSFORMS.build(dict(type=RandomCrop, size=self.crop_size))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        assert data["img"].height == data["img"].width == self.crop_size
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.crop_size
        assert right == left + self.crop_size
        np.equal(
            np.array(data["img"]),
            np.array(Image.open(img_path).crop((left, upper, right, lower))))

    def test_transform_multiple_keys(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": Image.open(img_path),
            "condition_img": Image.open(img_path),
        }

        # test transform
        trans = TRANSFORMS.build(
            dict(
                type=RandomCrop,
                size=self.crop_size,
                keys=["img", "condition_img"]))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        assert data["img"].height == data["img"].width == self.crop_size
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.crop_size
        assert right == left + self.crop_size
        np.equal(
            np.array(data["img"]),
            np.array(Image.open(img_path).crop((left, upper, right, lower))))
        np.equal(np.array(data["img"]), np.array(data["condition_img"]))

        # size mismatch
        data = {
            "img": Image.open(img_path),
            "condition_img": Image.open(img_path).resize((298, 398)),
        }
        with pytest.raises(
                AssertionError, match="Size mismatch"):
            data = trans(data)

        # test transform force_same_size=False
        trans = TRANSFORMS.build(
            dict(
                type=RandomCrop,
                size=self.crop_size,
                force_same_size=False,
                keys=["img", "condition_img"]))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        assert data["img"].height == data["img"].width == self.crop_size
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.crop_size
        assert right == left + self.crop_size

    def test_transform_list(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {"img": [Image.open(img_path),
                        Image.open(img_path).resize((64, 64))]}

        # test transform
        trans = TRANSFORMS.build(dict(type=RandomCrop, size=self.crop_size))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        for i in range(len(data["img"])):
            assert (
                data["img"][i].height == data["img"][i].width == self.crop_size
            )
            upper, left = data["crop_top_left"][i]
            lower, right = data["crop_bottom_right"][i]
            assert lower == upper + self.crop_size
            assert right == left + self.crop_size
            np.equal(
                np.array(data["img"][i]),
                np.array(
                    Image.open(img_path).crop((left, upper, right, lower))))

    def test_transform_multiple_keys_list(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": [
                Image.open(img_path), Image.open(img_path).resize((64, 64))],
            "condition_img": [
                Image.open(img_path), Image.open(img_path).resize((64, 64))],
        }

        # test transform
        trans = TRANSFORMS.build(
            dict(
                type=RandomCrop,
                size=self.crop_size,
                keys=["img", "condition_img"]))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        for i in range(len(data["img"])):
            assert (
                data["img"][i].height == data["img"][i].width == self.crop_size
            )
            upper, left = data["crop_top_left"][i]
            lower, right = data["crop_bottom_right"][i]
            assert lower == upper + self.crop_size
            assert right == left + self.crop_size
            np.equal(
                np.array(data["img"][i]),
                np.array(
                    Image.open(img_path).crop((left, upper, right, lower))))
            np.equal(np.array(data["img"][i]),
                     np.array(data["condition_img"][i]))

        # size mismatch
        data = {
            "img": [Image.open(img_path),
                    Image.open(img_path).resize((64, 64))],
            "condition_img": [
                Image.open(img_path).resize((298, 398)),
                Image.open(img_path).resize((64, 64))],
        }
        with pytest.raises(
                AssertionError, match="Size mismatch"):
            data = trans(data)

        # test transform force_same_size=False
        trans = TRANSFORMS.build(
            dict(
                type=RandomCrop,
                size=self.crop_size,
                force_same_size=False,
                keys=["img", "condition_img"]))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        for i in range(len(data["img"])):
            assert (
                data["img"][i].height == data["img"][i].width == self.crop_size
            )
            upper, left = data["crop_top_left"][i]
            lower, right = data["crop_bottom_right"][i]
            assert lower == upper + self.crop_size
            assert right == left + self.crop_size


class TestCenterCrop(TestCase):
    crop_size = 32

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {"img": Image.open(img_path)}

        # test transform
        trans = TRANSFORMS.build(dict(type=CenterCrop, size=self.crop_size))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        assert data["img"].height == data["img"].width == self.crop_size
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.crop_size
        assert right == left + self.crop_size
        np.equal(
            np.array(data["img"]),
            np.array(Image.open(img_path).crop((left, upper, right, lower))))

    def test_transform_multiple_keys(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": Image.open(img_path),
            "condition_img": Image.open(img_path),
        }

        # test transform
        trans = TRANSFORMS.build(
            dict(
                type=CenterCrop,
                size=self.crop_size,
                keys=["img", "condition_img"]))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        assert data["img"].height == data["img"].width == self.crop_size
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.crop_size
        assert right == left + self.crop_size
        np.equal(
            np.array(data["img"]),
            np.array(Image.open(img_path).crop((left, upper, right, lower))))
        np.equal(np.array(data["img"]), np.array(data["condition_img"]))

    def test_transform_list(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {"img": [
            Image.open(img_path), Image.open(img_path).resize((64, 64))]}

        # test transform
        trans = TRANSFORMS.build(dict(type=CenterCrop, size=self.crop_size))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        for i in range(len(data["img"])):
            assert (
                data["img"][i].height == data["img"][i].width == self.crop_size
            )
            upper, left = data["crop_top_left"][i]
            lower, right = data["crop_bottom_right"][i]
            assert lower == upper + self.crop_size
            assert right == left + self.crop_size
            np.equal(
                np.array(data["img"][i]),
                np.array(
                    Image.open(img_path).crop((left, upper, right, lower))))

    def test_transform_multiple_keys_list(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": [
                Image.open(img_path), Image.open(img_path).resize((64, 64))],
            "condition_img": [
                Image.open(img_path), Image.open(img_path).resize((64, 64))],
        }

        # test transform
        trans = TRANSFORMS.build(
            dict(
                type=CenterCrop,
                size=self.crop_size,
                keys=["img", "condition_img"]))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        for i in range(len(data["img"])):
            assert (
                data["img"][i].height == data["img"][i].width == self.crop_size
            )
            upper, left = data["crop_top_left"][i]
            lower, right = data["crop_bottom_right"][i]
            assert lower == upper + self.crop_size
            assert right == left + self.crop_size
            np.equal(
                np.array(data["img"][i]),
                np.array(
                    Image.open(img_path).crop((left, upper, right, lower))))
            np.equal(np.array(data["img"][i]),
                     np.array(data["condition_img"][i]))


class TestRandomHorizontalFlip(TestCase):

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": Image.open(img_path),
            "crop_top_left": [0, 0],
            "crop_bottom_right": [200, 200],
            "before_crop_size": [224, 224],
        }

        # test transform
        trans = TRANSFORMS.build(dict(type=RandomHorizontalFlip, p=1.))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        self.assertListEqual(data["crop_top_left"],
                             [0, data["before_crop_size"][1] - 200])

        np.equal(
            np.array(data["img"]),
            np.array(Image.open(img_path).transpose(Image.FLIP_LEFT_RIGHT)))

        # test transform p=0.0
        data = {
            "img": Image.open(img_path),
            "crop_top_left": [0, 0],
            "crop_bottom_right": [200, 200],
            "before_crop_size": [224, 224],
        }
        trans = TRANSFORMS.build(dict(type=RandomHorizontalFlip, p=0.))
        data = trans(data)
        assert "crop_top_left" in data
        self.assertListEqual(data["crop_top_left"], [0, 0])

        np.equal(np.array(data["img"]), np.array(Image.open(img_path)))

    def test_transform_multiple_keys(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": Image.open(img_path),
            "condition_img": Image.open(img_path),
            "crop_top_left": [0, 0],
            "crop_bottom_right": [200, 200],
            "before_crop_size": [224, 224],
        }

        # test transform
        trans = TRANSFORMS.build(
            dict(
                type=RandomHorizontalFlip,
                p=1.,
                keys=["img", "condition_img"]))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        self.assertListEqual(data["crop_top_left"],
                             [0, data["before_crop_size"][1] - 200])

        np.equal(
            np.array(data["img"]),
            np.array(Image.open(img_path).transpose(Image.FLIP_LEFT_RIGHT)))
        np.equal(np.array(data["img"]), np.array(data["condition_img"]))

    def test_transform_list(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": [
                Image.open(img_path), Image.open(img_path).resize((64, 64))],
            "crop_top_left": [[0, 0], [10, 10]],
            "crop_bottom_right": [[200, 200], [220, 220]],
            "before_crop_size": [[224, 224], [256, 256]],
        }

        # test transform
        trans = TRANSFORMS.build(dict(type=RandomHorizontalFlip, p=1.))
        transformed_data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        for i in range(len(data["img"])):
            self.assertListEqual(
                data["crop_top_left"][i],
                [data["crop_top_left"][i][0],
                 data["before_crop_size"][i][1] - data[
                    "crop_bottom_right"][i][1]])

            np.equal(
                np.array(transformed_data["img"][i]),
                np.array(
                    data["img"][i].transpose(Image.FLIP_LEFT_RIGHT)))

        # test transform p=0.0
        data = {
            "img": [
                Image.open(img_path), Image.open(img_path).resize((64, 64))],
            "crop_top_left": [[0, 0], [10, 10]],
            "crop_bottom_right": [[200, 200], [220, 220]],
            "before_crop_size": [[224, 224], [256, 256]],
        }
        trans = TRANSFORMS.build(dict(type=RandomHorizontalFlip, p=0.))
        transformed_data = trans(data)
        assert "crop_top_left" in data
        for i in range(len(data["img"])):
            self.assertListEqual(data["crop_top_left"][i],
                                 data["crop_top_left"][i])
            np.equal(np.array(transformed_data["img"][i]),
                     np.array(data["img"][i]))

    def test_transform_multiple_keys_list(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": [
                Image.open(img_path), Image.open(img_path).resize((64, 64))],
            "condition_img": [
                Image.open(img_path), Image.open(img_path).resize((64, 64))],
            "crop_top_left": [[0, 0], [10, 10]],
            "crop_bottom_right": [[200, 200], [220, 220]],
            "before_crop_size": [[224, 224], [256, 256]],
        }

        # test transform
        trans = TRANSFORMS.build(
            dict(
                type=RandomHorizontalFlip,
                p=1.,
                keys=["img", "condition_img"]))
        transformed_data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        for i in range(len(data["img"])):
            self.assertListEqual(
                data["crop_top_left"][i],
                [data["crop_top_left"][i][0],
                 data["before_crop_size"][i][1] - data[
                    "crop_bottom_right"][i][1]])

            np.equal(
                np.array(data["img"][i]),
                np.array(
                    transformed_data["img"][i].transpose(Image.FLIP_LEFT_RIGHT)))
            np.equal(np.array(data["img"][i]),
                     np.array(data["condition_img"][i]))


class TestMultiAspectRatioResizeCenterCrop(TestCase):
    sizes = [(32, 32), (16, 48)]  # noqa

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {"img": Image.open(img_path).resize((32, 36))}

        # test transform
        trans = TRANSFORMS.build(
            dict(type=MultiAspectRatioResizeCenterCrop, sizes=self.sizes))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        self.assertTupleEqual((data["img"].height, data["img"].width),
                              self.sizes[0])
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.sizes[0][0]
        assert right == left + self.sizes[0][1]
        np.equal(
            np.array(data["img"]),
            np.array(
                Image.open(img_path).resize((32, 36)).crop(
                    (left, upper, right, lower))))

        # test 2nd size
        data = {"img": Image.open(img_path).resize((55, 16))}
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        self.assertTupleEqual((data["img"].height, data["img"].width),
                              self.sizes[1])
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.sizes[1][0]
        assert right == left + self.sizes[1][1]
        np.equal(
            np.array(data["img"]),
            np.array(
                Image.open(img_path).resize((55, 16)).crop(
                    (left, upper, right, lower))))

    def test_transform_multiple_keys(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": Image.open(img_path).resize((32, 36)),
            "condition_img": Image.open(img_path).resize((32, 36)),
        }

        # test transform
        trans = TRANSFORMS.build(
            dict(
                type=MultiAspectRatioResizeCenterCrop,
                sizes=self.sizes,
                keys=["img", "condition_img"]))
        data = trans(data)
        assert "crop_top_left" in data
        assert len(data["crop_top_left"]) == 2
        self.assertTupleEqual((data["img"].height, data["img"].width),
                              self.sizes[0])
        upper, left = data["crop_top_left"]
        lower, right = data["crop_bottom_right"]
        assert lower == upper + self.sizes[0][0]
        assert right == left + self.sizes[0][1]
        np.equal(
            np.array(data["img"]),
            np.array(
                Image.open(img_path).resize((32, 36)).crop(
                    (left, upper, right, lower))))
        np.equal(np.array(data["img"]), np.array(data["condition_img"]))

    def test_transform_list(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {"img": [Image.open(img_path).resize((32, 36)),
                        Image.open(img_path).resize((55, 16))]}

        # test transform
        trans = TRANSFORMS.build(
            dict(type=MultiAspectRatioResizeCenterCrop, sizes=self.sizes))
        with pytest.raises(
                AssertionError, match="MultiAspectRatioResizeCenterCrop only"):
            _ = trans(data)

    def test_transform_multiple_keys_list(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        data = {
            "img": [Image.open(img_path).resize((32, 36)),
                        Image.open(img_path).resize((55, 16))],
            "condition_img": [Image.open(img_path).resize((32, 36)),
                              Image.open(img_path).resize((55, 16))]}

        # test transform
        trans = TRANSFORMS.build(
            dict(type=MultiAspectRatioResizeCenterCrop, sizes=self.sizes))
        with pytest.raises(
                AssertionError, match="MultiAspectRatioResizeCenterCrop only"):
            _ = trans(data)


class TestConcatMultipleImgs(TestCase):

    def test_transform_list(self):
        data = {"img": [torch.zeros((3, 32, 32))] * 2}

        # test transform
        trans = TRANSFORMS.build(dict(type=ConcatMultipleImgs))
        data = trans(data)
        assert data["img"].shape == (6, 32, 32)  # type: ignore[attr-defined]
