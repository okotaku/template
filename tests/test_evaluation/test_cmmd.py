import os

import pytest
from mmengine.testing import RunnerTestCase

from diffengine.evaluation import CMMD


@pytest.mark.skipif("GITHUB_ACTION" in os.environ,
                    reason="skip external api call during CI")
class TestCMMD(RunnerTestCase):

    def test_clip_t(self):
        cmmd = CMMD()
        ref_imgs = ["tests/testdata/color.jpg"]
        pred_imgs = ["tests/testdata/cond.jpg"]
        score = cmmd(ref_imgs, pred_imgs)
        assert isinstance(score, float)
        assert score >= 0
