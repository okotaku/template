import copy

from mmengine.config import Config
from mmengine.testing import RunnerTestCase

from diffengine.engine.hooks import MemoryFormatHook


class TestMemoryFormatHook(RunnerTestCase):

    def test_init(self) -> None:
        MemoryFormatHook()

    def test_before_train(self) -> None:
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/sd.py").model
        runner = self.build_runner(cfg)
        hook = MemoryFormatHook()
        for p in runner.model.unet.parameters():
            is_contiguous = p.is_contiguous()
            break
        assert is_contiguous

        # run hook
        hook.before_train(runner)

        for p in runner.model.unet.parameters():
            is_contiguous = p.is_contiguous()
            break
        assert not is_contiguous

        # Test StableDiffusionControlNet
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/sdcn.py").model
        runner = self.build_runner(cfg)
        hook = MemoryFormatHook()
        for p in runner.model.unet.parameters():
            is_contiguous = p.is_contiguous()
            break
        assert is_contiguous

        # run hook
        hook.before_train(runner)

        for p in runner.model.unet.parameters():
            is_contiguous = p.is_contiguous()
            break
        assert not is_contiguous
