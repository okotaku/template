# Refer to https://github.com/TencentQQGYLab/ELLA/blob/main/model.py
from collections import OrderedDict

import torch
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization module."""

    def __init__(self,
                 embedding_dim: int,
                 time_embedding_dim: int | None = None) -> None:
        super().__init__()

        if time_embedding_dim is None:
            time_embedding_dim = embedding_dim

        self.silu = nn.SiLU()
        self.linear = nn.Linear(time_embedding_dim, 2 * embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, timestep_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function."""
        emb = self.linear(self.silu(timestep_embedding))
        shift, scale = emb.view(len(x), 1, -1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class SquaredReLU(nn.Module):
    """Squared ReLU activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        return torch.square(torch.relu(x))


class PerceiverAttentionBlock(nn.Module):
    """Perceiver Attention Block."""

    def __init__(
        self, d_model: int, n_heads: int,
        time_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("sq_relu", SquaredReLU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ],
            ),
        )

        self.ln_1 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_2 = AdaLayerNorm(d_model, time_embedding_dim)
        self.ln_ff = AdaLayerNorm(d_model, time_embedding_dim)

    def attention(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Attention function."""
        attn_output, _ = self.attn(q, kv, kv, need_weights=False)
        return attn_output

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        timestep_embedding: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward function."""
        normed_latents = self.ln_1(latents, timestep_embedding)
        latents = latents + self.attention(
            q=normed_latents,
            kv=torch.cat([normed_latents, self.ln_2(x, timestep_embedding)], dim=1),
        )
        return latents + self.mlp(self.ln_ff(latents, timestep_embedding))


class PerceiverResampler(nn.Module):
    """Perceiver Resampler."""

    def __init__(
        self,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        output_dim: int | None = None,
        input_dim: int | None = None,
        time_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.latents = nn.Parameter(width**-0.5 * torch.randn(num_latents, width))
        self.time_aware_linear = nn.Linear(
            time_embedding_dim or width, width, bias=True,
        )

        if self.input_dim is not None:
            self.proj_in = nn.Linear(input_dim, width)

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    width, heads, time_embedding_dim=time_embedding_dim,
                )
                for _ in range(layers)
            ],
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(width, output_dim), nn.LayerNorm(output_dim),
            )

    def forward(self, x: torch.Tensor, timestep_embedding: torch.Tensor = None,
                ) -> torch.Tensor:
        """Forward function."""
        learnable_latents = self.latents.unsqueeze(dim=0).repeat(len(x), 1, 1)
        latents = learnable_latents + self.time_aware_linear(
            torch.nn.functional.silu(timestep_embedding),
        )
        if self.input_dim is not None:
            x = self.proj_in(x)
        for p_block in self.perceiver_blocks:
            latents = p_block(x, latents, timestep_embedding=timestep_embedding)

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        return latents


class ELLA(nn.Module):
    """ELLA model."""

    def __init__(
        self,
        time_channel: int = 320,
        time_embed_dim: int = 768,
        act_fn: str = "silu",
        out_dim: int | None = None,
        width: int = 768,
        layers: int = 6,
        heads: int = 8,
        num_latents: int = 64,
        input_dim: int = 2048,
    ) -> None:
        super().__init__()

        self.position = Timesteps(
            time_channel, flip_sin_to_cos=True, downscale_freq_shift=0,
        )
        self.time_embedding = TimestepEmbedding(
            in_channels=time_channel,
            time_embed_dim=time_embed_dim,
            act_fn=act_fn,
            out_dim=out_dim,
        )

        self.connector = PerceiverResampler(
            width=width,
            layers=layers,
            heads=heads,
            num_latents=num_latents,
            input_dim=input_dim,
            time_embedding_dim=time_embed_dim,
        )

    def forward(self, text_encode_features: torch.Tensor,
                timesteps: torch.Tensor) -> torch.Tensor:
        """Forward function."""
        device = text_encode_features.device
        dtype = text_encode_features.dtype

        ori_time_feature = self.position(timesteps.view(-1)).to(device, dtype=dtype)
        ori_time_feature = (
            ori_time_feature.unsqueeze(dim=1)
            if ori_time_feature.ndim == 2  # noqa: PLR2004
            else ori_time_feature
        )
        ori_time_feature = ori_time_feature.expand(len(text_encode_features), -1, -1)
        time_embedding = self.time_embedding(ori_time_feature)

        return self.connector(
            text_encode_features, timestep_embedding=time_embedding,
        )
