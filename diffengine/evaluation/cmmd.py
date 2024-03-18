from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


def mmd(x: torch.Tensor, y: torch.Tensor, sigma: int = 10,
        scale: int = 1000) -> float:
    """MMD implementation.

    Reference: https://github.com/sayakpaul/cmmd-pytorch

    Args:
    ----
        x (torch.Tensor):
            The first set of embeddings of shape (n, embedding_dim).
        y (torch.Tensor):
            The second set of embeddings of shape (n, embedding_dim).
        sigma (int): The bandwidth parameter for the Gaussian RBF kernel.
            Defaults to 10.
        scale (int): The scaling factor for the MMD distance.
            Defaults to 1000.

    Returns:
    -------
        The MMD distance between x and y embedding sets.

    """
    x_sqnorms = torch.diag(torch.matmul(x, x.T))
    y_sqnorms = torch.diag(torch.matmul(y, y.T))

    gamma = 1 / (2 * sigma**2)
    k_xx = torch.mean(
        torch.exp(
            -gamma * (
                -2 * torch.matmul(
                    x, x.T,
                    ) + torch.unsqueeze(
                        x_sqnorms, 1) + torch.unsqueeze(x_sqnorms, 0))))
    k_xy = torch.mean(
        torch.exp(
            -gamma * (
                -2 * torch.matmul(
                    x, y.T,
                    ) + torch.unsqueeze(
                        x_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0))))
    k_yy = torch.mean(
        torch.exp(
            -gamma * (
                -2 * torch.matmul(
                    y, y.T,
                    ) + torch.unsqueeze(
                        y_sqnorms, 1) + torch.unsqueeze(y_sqnorms, 0))))

    return (scale * (k_xx + k_yy - 2 * k_xy)).item()


class CMMD:
    """CMMD Score.

    Args:
    ----
        model (str): The name of the model to use.
            Defaults to "openai/clip-vit-large-patch14-336".
        device (str): The device to use.
            Defaults to "cuda:0".

    """

    def __init__(self,
                 model: str = "openai/clip-vit-large-patch14-336",
                 device: str = "cuda:0") -> None:
        self.preprocess = CLIPImageProcessor.from_pretrained(model)
        self.model = CLIPVisionModelWithProjection.from_pretrained(model)

        self.model.eval()
        self.model.to(device)
        self.device = device

    def _get_embs(self, img: str | Path) -> torch.Tensor:
        """Get the image embeddings."""
        img = Image.open(img).convert("RGB")
        img = self.preprocess(
            img, return_tensors="pt",
        ).pixel_values[0].unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model(img).image_embeds.cpu()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features

    def __call__(self,
                 ref_imgs: list[str] | list[Path],
                 pred_imgs: list[str] | list[Path]) -> float:
        """Calculate the CLIP text score.

        Args:
        ----
            ref_imgs (list[str]): The list of reference image paths.
            pred_imgs (list[str]): The list of predicted image paths.

        """
        ref_embs = torch.cat([self._get_embs(img) for img in ref_imgs])
        pred_embs = torch.cat([self._get_embs(img) for img in pred_imgs])
        return mmd(ref_embs, pred_embs)
