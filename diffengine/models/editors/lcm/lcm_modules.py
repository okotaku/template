# copied from https://github.com/huggingface/diffusers/blob/main/examples/
#    consistency_distillation/train_lcm_distill_sdxl_wds.py
import torch
from torch import nn


def extract_into_tensor(x: torch.Tensor, timesteps: torch.Tensor,
                        ) -> torch.Tensor:
    """Extract time-dependent values from a tensor."""
    b = timesteps.shape[0]
    out = x.gather(-1, timesteps)
    return out.reshape(b, 1, 1, 1)


def scalings_for_boundary_conditions(
    timestep: torch.Tensor, sigma_data: float=0.5,
    timestep_scaling: float = 10.0) -> tuple:
    """Scalings for boundary conditions.

    From LCMScheduler.get_scalings_for_boundary_condition_discrete
    """
    b = timestep.shape[0]
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip.reshape(b, 1, 1, 1), c_out.reshape(b, 1, 1, 1)


def guidance_scale_embedding(
    w: torch.Tensor,
    embedding_dim: int = 512) -> torch.Tensor:
    """Generate guidance scale embedding.

    See https://github.com/google-research/vdm/blob/
    dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
    ----
        w (torch.Tensor):
            generate embedding vectors at these timesteps
        embedding_dim (int):
            dimension of the embeddings to generate. Defaults to 512

    Returns:
    -------
        `torch.FloatTensor`: Embedding vectors with shape
        `(len(timesteps), embedding_dim)`

    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=w.dtype) * -emb)
    emb = w[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


class DDIMSolver(nn.Module):
    """DDIM solver."""

    def __init__(self, alpha_cumprods: torch.Tensor,
                 timesteps: int = 1000,
                 ddim_timesteps: int = 50) -> None:
        super().__init__()
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps

        ddim_timesteps_tensor = (torch.arange(1, ddim_timesteps + 1) * step_ratio) - 1
        ddim_alpha_cumprods = alpha_cumprods[ddim_timesteps_tensor]
        ddim_alpha_cumprods_prev = torch.cat([
            alpha_cumprods[:1], alpha_cumprods[ddim_timesteps_tensor[:-1]]])

        # convert to torch tensors
        self.register_buffer("ddim_timesteps", ddim_timesteps_tensor.long())
        self.register_buffer("ddim_alpha_cumprods", ddim_alpha_cumprods)
        self.register_buffer("ddim_alpha_cumprods_prev",
                             ddim_alpha_cumprods_prev)

    def ddim_step(self, pred_x0: torch.Tensor, pred_noise: torch.Tensor,
                  timestep_index: torch.Tensor) -> torch.Tensor:
        """DDIM step."""
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev.to(pred_x0.dtype)
