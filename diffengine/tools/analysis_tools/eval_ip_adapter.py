import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from diffusers import DiffusionPipeline
from transformers import CLIPVisionModelWithProjection

from diffengine.evaluation import CLIPT, DINOScore


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the IP adapter on a set of images.")
    parser.add_argument("--weight", help="Model name",
                        type=str, default="ip-adapter_sd15.bin")
    return parser.parse_args()


def main():
    args = parse_args()

    Path("work_dirs/ip_adapter_eval").mkdir(exist_ok=True, parents=True)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "h94/IP-Adapter",
        subfolder="models/image_encoder",
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        image_encoder=image_encoder,
        torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.load_ip_adapter("h94/IP-Adapter",
                         subfolder="models",
                         weight_name=args.weight)
    pipe.set_ip_adapter_scale(0.6)

    clipt = CLIPT()

    ds = load_dataset("ImagenHub/DreamBooth_Concepts")["train"]
    eval_ds = load_dataset("ImagenHub/Subject_Driven_Image_Generation")["eval"]
    subject = list({d["subject"] for d in ds})
    generator = torch.Generator(device="cuda").manual_seed(0)
    results = []
    for s in subject:
        ref_imgs = [d["image"] for d in ds if d["subject"] == s]
        dino_score = DINOScore(ref_imgs)
        clipt_scores = []
        dino_scores = []
        for i, d in enumerate(eval_ds):
            if d["subject"] != s:
                continue
            prompt = d["prompt"].replace("<token> ", "")
            img = pipe(prompt,
                       ip_adapter_image=ref_imgs[0],
                       generator=generator).images[0]
            img.save(f"work_dirs/ip_adapter_eval/{s}_{i}.jpg")
            clipt_scores.append(clipt(img, prompt))
            dino_scores.append(dino_score(img))

        results.append([s, np.mean(clipt_scores), np.mean(dino_scores)])
        del dino_score
        torch.cuda.empty_cache()
    results_df = pd.DataFrame(results, columns=["subject", "clipt", "dino"])
    print(results_df.mean())
    results_df.to_csv("work_dirs/ip_adapter_eval/eval.csv", index=False)

if __name__ == "__main__":
    main()
