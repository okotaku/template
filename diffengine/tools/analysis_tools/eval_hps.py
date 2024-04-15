import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from diffusers import DiffusionPipeline
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the IP adapter on a set of images.")
    parser.add_argument("--model", help="Model name",
                        type=str, default="runwayml/stable-diffusion-v1-5")
    return parser.parse_args()


def main():
    try:
        import hpsv2
    except ImportError as e:
        msg = "Please install hpsv2"
        raise ImportError(msg) from e
    args = parse_args()

    model_name = args.model.split("/")[-1]
    out_dir = f"work_dirs/hpdv2_{model_name}"
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    eval_ds = load_dataset("ymhao/HPDv2", split="test")
    generator = torch.Generator(device="cuda").manual_seed(0)
    results = []
    for i, d in tqdm(enumerate(eval_ds)):
        img = pipe(d["prompt"], generator=generator).images[0]
        img.save(f"{out_dir}/img_{i}.jpg")

        results.append(hpsv2.score(img, d["prompt"], hps_version="v2.1")[0])
    results_df = pd.DataFrame(results, columns=["hpsv2"])
    print(results_df.mean())
    results_df.to_csv(f"{out_dir}/eval.csv", index=False)

if __name__ == "__main__":
    main()
