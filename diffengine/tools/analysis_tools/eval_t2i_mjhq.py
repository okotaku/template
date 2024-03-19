import argparse
from pathlib import Path

import mmengine
import pandas as pd
import torch
from cleanfid import fid
from diffusers import DiffusionPipeline
from tqdm import tqdm

from diffengine.evaluation import CLIPT, CMMD


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the IP adapter on a set of images.")
    parser.add_argument("--model", help="Model name",
                        type=str, default="runwayml/stable-diffusion-v1-5")
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model.split("/")[-1]
    out_dir = f"work_dirs/t2i_mjhq_{model_name}"
    data_dir = "work_dirs/mjhq"
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    clipt = CLIPT()
    cmmd = CMMD()

    eval_ds = mmengine.load(f"{data_dir}/meta_data.json")
    generator = torch.Generator(device="cuda").manual_seed(0)
    results = []
    for i, d in tqdm(enumerate(eval_ds.values())):
        img = pipe(d["prompt"], generator=generator).images[0]
        img.save(f"{out_dir}/img_{i}.jpg")

        results.append([d["category"], clipt(img, d["prompt"])])
    results_df = pd.DataFrame(results, columns=["category", "clipt"])
    print(results_df.mean())
    fid_score = fid.compute_fid(data_dir, out_dir)
    print("FID", fid_score)
    ref_imgs = list(Path(data_dir).glob("**/*.jpg"))
    pred_imgs = list(Path(out_dir).glob("*.jpg"))
    cmmd_score = cmmd(ref_imgs, pred_imgs)
    print("CMMD", cmmd_score)
    results_df.to_csv(f"{out_dir}/eval.csv", index=False)

if __name__ == "__main__":
    main()
