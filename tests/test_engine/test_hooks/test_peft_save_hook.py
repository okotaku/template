import copy
import os.path as osp
import shutil
from pathlib import Path

from mmengine.config import Config
from mmengine.model import BaseModel
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase
from torch import nn

from diffengine.engine.hooks import PeftSaveHook


class DummyWrapper(BaseModel):

    def __init__(self, model) -> None:
        super().__init__()
        if not isinstance(model, nn.Module):
            model = MODELS.build(model)
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class TestPeftSaveHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name="DummyWrapper", module=DummyWrapper)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop("DummyWrapper")
        return super().tearDown()

    def test_init(self):
        PeftSaveHook()

    def test_before_save_checkpoint(self):
        # with text encoder
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = Config.fromfile("tests/configs/sd_lora.py").model
        runner = self.build_runner(cfg)
        checkpoint = dict(
            state_dict=MODELS.build(cfg.model).state_dict())
        hook = PeftSaveHook()
        hook.before_save_checkpoint(runner, checkpoint)

        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/unet",
                     "adapter_model.safetensors")).exists()
        assert Path(
            osp.join(runner.work_dir, f"step{runner.iter}/text_encoder",
                     "adapter_model.safetensors")).exists()
        shutil.rmtree(
            osp.join(runner.work_dir, f"step{runner.iter}"))

        for key in checkpoint["state_dict"]:
            assert key.startswith(("unet", "text_encoder"))
            assert "default" in key
