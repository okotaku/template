import argparse
import os.path as osp
from pathlib import Path

import joblib
import mmengine
import numpy as np
import pandas as pd
from mmengine.config import Config
from PIL import Image
from tqdm import tqdm

from diffengine.configs import cfgs_name_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a checkpoint to be published")
    parser.add_argument("config", help="Path to config")
    parser.add_argument("--n_jobs", help="Number of jobs.", type=int,
                        default=4)
    parser.add_argument("--out", help="Output path", default="bucked_ids.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.config).exists():
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError as exc:
            msg = f"Cannot find {args.config}"
            raise FileNotFoundError(msg) from exc

    cfg = Config.fromfile(args.config)
    data_dir = cfg.train_dataloader.dataset.get("dataset")
    img_dir = cfg.train_dataloader.dataset.get("img_dir", "")
    csv = cfg.train_dataloader.dataset.get("csv", "metadata.csv")
    image_column = cfg.train_dataloader.dataset.get("image_column", "image")
    csv_path = osp.join(data_dir, csv)
    img_df = pd.read_csv(csv_path)

    i = 0
    while True:
        if hasattr(cfg.train_dataloader.dataset.pipeline[i], "sizes"):
            sizes = cfg.train_dataloader.dataset.pipeline[i].get("sizes")
            break
        i += 1
    aspect_ratios = np.array([s[0] / s[1] for s in sizes])

    def get_bucket_id(file_name):
        image = osp.join(data_dir, img_dir, file_name)
        image = Image.open(image)
        aspect_ratio = image.height / image.width
        return np.argmin(np.abs(aspect_ratios - aspect_ratio))

    bucket_ids = joblib.Parallel(n_jobs=args.n_jobs, verbose=10)(
        joblib.delayed(get_bucket_id)(file_name)
        for file_name in tqdm(img_df[image_column].values))

    print(pd.DataFrame(bucket_ids).value_counts())

    mmengine.dump(bucket_ids, args.out)

if __name__ == "__main__":
    main()
