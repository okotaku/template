# Copied from xtuner.configs.__init__
import os
from pathlib import Path


def get_cfgs_name_path() -> dict:
    """Get all config names and paths."""
    path = Path(__file__).parent
    mapping = {}
    for root, _, files in os.walk(path):
        # Skip if it is a base config
        if "_base_" in root:
            continue
        for file_ in files:
            if file_.endswith(
                (".py", ".json"),
            ) and not file_.startswith(".") and not file_.startswith("_"):
                mapping[Path(file_).stem] = Path(root) / file_
    return mapping


cfgs_name_path = get_cfgs_name_path()

__all__ = ["cfgs_name_path"]
