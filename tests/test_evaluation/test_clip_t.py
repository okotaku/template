import os

import pytest
from mmengine.testing import RunnerTestCase
from PIL import Image

from diffengine.evaluation import CLIPT


@pytest.mark.skipif("GITHUB_ACTION" in os.environ,
                    reason="skip external api call during CI")
class TestCLIPT(RunnerTestCase):

    def test_clip_t(self):
        clip_t = CLIPT()
        img = Image.open("tests/testdata/color.jpg")
        prompt = "A dog"
        score = clip_t(img, prompt)
        assert isinstance(score, float)
        assert 0 <= score <= 1
