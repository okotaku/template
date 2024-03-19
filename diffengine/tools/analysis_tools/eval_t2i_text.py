import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from diffusers import DiffusionPipeline
from tqdm import tqdm

from diffengine.evaluation import CLIPT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the IP adapter on a set of images.")
    parser.add_argument("--model", help="Model name",
                        type=str, default="runwayml/stable-diffusion-v1-5")
    return parser.parse_args()


def main():
    args = parse_args()

    model_name = args.model.split("/")[-1]
    out_dir = f"work_dirs/t2i_text_{model_name}"
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    clipt = CLIPT()

    eval_ds = load_dataset("ImagenHub/Text_to_Image")["eval"]
    generator = torch.Generator(device="cuda").manual_seed(0)
    results = []
    for i, d in tqdm(enumerate(eval_ds)):
        img = pipe(d["prompt"], generator=generator).images[0]
        img.save(f"{out_dir}/img_{i}.jpg")

        results.append([d["category"], clipt(img, d["prompt"])])
    results_df = pd.DataFrame(results, columns=["category", "clipt"])
    print(results_df.mean())
    results_df.to_csv(f"{out_dir}/eval.csv", index=False)

if __name__ == "__main__":
    main()
