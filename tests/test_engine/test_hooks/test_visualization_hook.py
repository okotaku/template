import copy
from unittest.mock import MagicMock

from mmengine.config import Config
from mmengine.runner import EpochBasedTrainLoop
from mmengine.testing import RunnerTestCase

from diffengine.engine.hooks import VisualizationHook


class TestVisualizationHook(RunnerTestCase):

    def test_before_train(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(prompt=["a dog"],
            height=64,
            width=64)
        hook.before_train(runner)

    def test_before_train_with_condition(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(
            prompt=["a dog"], condition_image=["testdata/color.jpg"],
            height=64,
            width=64)
        hook.before_train(runner)

    def test_after_train_epoch(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(prompt=["a dog"],
            height=64,
            width=64)
        hook.after_train_epoch(runner)

    def test_after_train_epoch_with_condition(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(
            prompt=["a dog"], condition_image=["testdata/color.jpg"],
            height=64,
            width=64)
        hook.after_train_epoch(runner)

    def test_after_train_epoch_with_example_iamge(self):
        runner = MagicMock()

        # test epoch-based
        runner.train_loop = MagicMock(spec=EpochBasedTrainLoop)
        runner.epoch = 5
        hook = VisualizationHook(
            prompt=["a dog"], example_image=["testdata/color.jpg"],
            height=64,
            width=64)
        hook.after_train_epoch(runner)

    def test_after_train_iter(self):
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.train_cfg.max_iters = 100
        cfg.model = Config.fromfile("tests/configs/sd.py").model
        runner = self.build_runner(cfg)
        hook = VisualizationHook(
            prompt=["a dog"],
            by_epoch=False,
            height=64,
            width=64)
        for i in range(3):
            hook.after_train_iter(runner, i)
            runner.train_loop._iter += 1

    def test_after_train_iter_with_condition(self):
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.train_cfg.max_iters = 100
        cfg.model = Config.fromfile("tests/configs/sdcn.py").model
        runner = self.build_runner(cfg)
        hook = VisualizationHook(
            prompt=["a dog"],
            condition_image=["tests/testdata/cond.jpg"],
            height=64,
            width=64,
            by_epoch=False)
        for i in range(3):
            hook.after_train_iter(runner, i)
            runner.train_loop._iter += 1
