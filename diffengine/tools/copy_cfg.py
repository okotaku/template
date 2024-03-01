# flake8: noqa: T201
# Copyright (c) OpenMMLab. All rights reserved.
# Copied from xtuner.tools.copy_cfg
import argparse
import shutil
from pathlib import Path

from mmengine.utils import mkdir_or_exist

from diffengine.configs import cfgs_name_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", help="config name")
    parser.add_argument("save_dir", help="save directory for copied config")
    return parser.parse_args()


def add_copy_suffix(string: str) -> str:
    """Add suffix to copied config."""
    return f"{Path(string).stem}_copy{Path(string).suffix}"


def main() -> None:
    """Copy configs."""
    args = parse_args()
    mkdir_or_exist(args.save_dir)
    config_path = cfgs_name_path[args.config_name]
    save_path = Path(args.save_dir) / add_copy_suffix(Path(config_path).name)
    shutil.copyfile(config_path, save_path)
    print(f"Copy to {save_path}")


if __name__ == "__main__":
    main()
