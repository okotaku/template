# flake8: noqa: T201
# Copyright (c) OpenMMLab. All rights reserved.
# Copied from xtuner.tools.list_cfg
import argparse

from diffengine.configs import cfgs_name_path  # type: ignore[attr-defined]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--pattern", default=None, help="Pattern for fuzzy matching")
    return parser.parse_args()


def main() -> None:
    """List configs."""
    args = parse_args()
    configs_names = sorted(cfgs_name_path.keys())
    print("==========================CONFIGS===========================")
    if args.pattern is not None:
        print(f"PATTERN: {args.pattern}")
        print("-------------------------------")
    for name in configs_names:
        if args.pattern is None or args.pattern.lower() in name.lower():
            print(name)
    print("=============================================================")


if __name__ == "__main__":
    main()
