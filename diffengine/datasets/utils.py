# flake8: noqa: S311,ANN001
import random

import numpy as np
import torch


def encode_prompt(batch,
                text_encoder,
                tokenizer,
                caption_column,
                *,
                is_train: bool = True) -> dict[str, torch.Tensor]:
    """Encode prompt."""
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, list | np.ndarray):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
        )[0]

    return {
        "prompt_embeds": prompt_embeds.cpu(),
    }
