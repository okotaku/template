import argparse
from pathlib import Path

import mmengine
from mmengine.logging import print_log


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mean score for a list of dictionaries.")
    parser.add_argument("work_dir", help="Path to work_dirs")
    return parser.parse_args()


def calculate_mean_for_dicts(*dicts) -> dict:
    """Calculate the mean for each key in a list of dictionaries."""
    # Initialize an empty dictionary to store the cumulative sum for each key
    key_sums: dict[str, float] = {}
    # Initialize a dictionary to keep track of the count of values for each key
    key_counts: dict[str, float] = {}

    # Iterate through each dictionary
    for d in dicts:
        for key, value in d.items():
            # Update the cumulative sum for the key
            key_sums.setdefault(key, 0)
            key_sums[key] += value
            # Update the count for the key
            key_counts.setdefault(key, 0)
            key_counts[key] += 1

    # Calculate the mean for each key
    return {key: key_sums[key] / key_counts[key] for key in key_sums}


def main() -> None:
    args = parse_args()

    dicts_path = Path(args.work_dir).glob("**/scores.json")
    dicts: tuple[dict] = tuple([mmengine.load(d) for d in dicts_path])
    result = calculate_mean_for_dicts(*dicts)
    print_log(result)


if __name__ == "__main__":
    main()
