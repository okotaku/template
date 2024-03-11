import os

import pytest
from mmengine.testing import RunnerTestCase
from PIL import Image

from diffengine.evaluation import DINOScore


@pytest.mark.skipif("GITHUB_ACTION" in os.environ,
                    reason="skip external api call during CI")
class TestDINOScore(RunnerTestCase):

    def test_dino_score(self):
        img = Image.open("tests/testdata/color.jpg")
        dino = DINOScore([img])
        score = dino(img)
        assert isinstance(score, float)
        assert 0 <= score <= 1
