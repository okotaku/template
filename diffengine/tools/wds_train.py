import argparse
import os
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.registry import DATASETS
from mmengine.runner import Runner

from diffengine.configs import cfgs_name_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--resume", action="store_true", help="Whether to resume checkpoint.")
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="enable automatic-mixed-precision training")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher")
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = Path("./work_dirs") / Path(args.config).stem

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.get("type", "OptimWrapper")
        if optim_wrapper not in ["OptimWrapper", "AmpOptimWrapper"]:
            msg = (
                "`--amp` is not supported custom optimizer wrapper type "
                f"`{optim_wrapper}."
            )
            raise ValueError(msg)
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.setdefault("loss_scale", "dynamic")

    # resume training
    if args.resume:
        cfg.resume = True
        cfg.load_from = None

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main() -> None:
    """Train models."""
    args = parse_args()

    # parse config
    if not Path(args.config).exists():
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError as exc:
            msg = f"Cannot find {args.config}"
            raise FileNotFoundError(msg) from exc

    # load config
    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    train_dataloader =  DATASETS.build(cfg["train_dataloader"])
    runner = Runner(
        model=cfg["model"],
        work_dir=cfg["work_dir"],
        train_dataloader=train_dataloader,
        val_dataloader=cfg.get("val_dataloader"),
        test_dataloader=cfg.get("test_dataloader"),
        train_cfg=cfg.get("train_cfg"),
        val_cfg=cfg.get("val_cfg"),
        test_cfg=cfg.get("test_cfg"),
        auto_scale_lr=cfg.get("auto_scale_lr"),
        optim_wrapper=cfg.get("optim_wrapper"),
        param_scheduler=cfg.get("param_scheduler"),
        val_evaluator=cfg.get("val_evaluator"),
        test_evaluator=cfg.get("test_evaluator"),
        default_hooks=cfg.get("default_hooks"),
        custom_hooks=cfg.get("custom_hooks"),
        data_preprocessor=cfg.get("data_preprocessor"),
        load_from=cfg.get("load_from"),
        resume=cfg.get("resume", False),
        launcher=cfg.get("launcher", "none"),
        env_cfg=cfg.get("env_cfg", dict(dist_cfg=dict(backend="nccl"))),
        log_processor=cfg.get("log_processor"),
        log_level=cfg.get("log_level", "INFO"),
        visualizer=cfg.get("visualizer"),
        default_scope=cfg.get("default_scope", "mmengine"),
        randomness=cfg.get("randomness", dict(seed=None)),
        experiment_name=cfg.get("experiment_name"),
        cfg=cfg,
    )

    # start training
    runner.train()


if __name__ == "__main__":
    main()
