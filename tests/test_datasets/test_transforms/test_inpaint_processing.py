import os.path as osp
from unittest import TestCase

import numpy as np
import pytest
import torch
from mmengine.registry import TRANSFORMS
from PIL import Image

from diffengine.datasets import (
    GetMaskedImage,
    MaskToTensor,
)


class TestMaskToTensor(TestCase):

    def test_transform(self):
        data = {"mask": np.zeros((32, 32, 1))}

        # test transform
        trans = TRANSFORMS.build(dict(type=MaskToTensor))
        data = trans(data)
        assert data["mask"].shape == (1, 32, 32)

    def test_transform_list(self):
        data = {"mask": [np.zeros((32, 32, 1))] * 2}

        # test transform
        trans = TRANSFORMS.build(dict(type=MaskToTensor))
        with pytest.raises(
                AssertionError, match="MaskToTensor only support"):
            _ = trans(data)


class TestGetMaskedImage(TestCase):

    def test_transform(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        img = torch.Tensor(np.array(Image.open(img_path)))
        mask = np.zeros((img.shape[0], img.shape[1], 1))
        mask[:10, :10] = 1
        mask = torch.Tensor(mask)
        data = {"img": img, "mask": mask}

        # test transform
        trans = TRANSFORMS.build(dict(type=GetMaskedImage))
        data = trans(data)
        assert "masked_image" in data
        assert data["masked_image"].shape == img.shape
        assert torch.allclose(data["masked_image"][10:, 10:], img[10:, 10:])
        assert data["masked_image"][:10, :10].sum() == 0

    def test_transform_list(self):
        img_path = osp.join(osp.dirname(__file__), "../../testdata/color.jpg")
        img = torch.Tensor(np.array(Image.open(img_path)))
        mask = np.zeros((img.shape[0], img.shape[1], 1))
        mask[:10, :10] = 1
        mask = torch.Tensor(mask)
        data = {"img": [img, img], "mask": [mask, mask]}

        # test transform
        trans = TRANSFORMS.build(dict(type=GetMaskedImage))
        with pytest.raises(
                AssertionError, match="GetMaskedImage only support"):
            _ = trans(data)
