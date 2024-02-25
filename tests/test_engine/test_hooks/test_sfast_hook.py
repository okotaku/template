import copy
import unittest

from diffusers import AutoencoderKL, UNet2DConditionModel
from mmengine.config import Config
from mmengine.testing import RunnerTestCase
from transformers import CLIPTextModel

from diffengine.engine.hooks import SFastHook


def has_sfast() -> bool:
    try:
        import sfast  # noqa: F401
    except ImportError:
        return False
    else:
        return True


@unittest.skipIf(not has_sfast(), "stable-fast is not installed")
class TestSFastHook(RunnerTestCase):

    def test_init(self) -> None:
        SFastHook()

    def test_before_train(self) -> None:
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/sd.py").model
        runner = self.build_runner(cfg)
        hook = SFastHook()
        assert isinstance(runner.model.unet, UNet2DConditionModel)
        assert isinstance(runner.model.vae, AutoencoderKL)
        assert isinstance(runner.model.text_encoder, CLIPTextModel)
        # compile
        hook.before_train(runner)
        assert not isinstance(runner.model.unet, UNet2DConditionModel)
        assert not isinstance(runner.model.vae, AutoencoderKL)
        assert not isinstance(runner.model.text_encoder, CLIPTextModel)
