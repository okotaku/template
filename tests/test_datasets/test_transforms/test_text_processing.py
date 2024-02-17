from unittest import TestCase

from mmengine.registry import TRANSFORMS

from diffengine.datasets import (
    AddConstantCaption,
    RandomTextDrop,
)


class TestRandomTextDrop(TestCase):

    def test_transform(self):
        data = {
            "text": "a dog",
        }

        # test transform
        trans = TRANSFORMS.build(dict(type=RandomTextDrop, p=1.))
        data = trans(data)
        assert data["text"] == ""

        # test transform p=0.0
        data = {
            "text": "a dog",
        }
        trans = TRANSFORMS.build(dict(type=RandomTextDrop, p=0.))
        data = trans(data)
        assert data["text"] == "a dog"


class TestAddConstantCaption(TestCase):

    def test_transform(self):
        data = {
            "text": "a dog.",
        }

        # test transform
        trans = TRANSFORMS.build(dict(type=AddConstantCaption,
                                      constant_caption="in szn style"))
        data = trans(data)
        assert data["text"] == "a dog. in szn style"
