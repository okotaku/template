import torch
import torch.nn.functional as F  # noqa

from diffengine.models.losses.base import BaseLoss
from diffengine.models.losses.utils import compute_snr


class DeBiasEstimationLoss(BaseLoss):
    """DeBias Estimation loss.

    https://arxiv.org/abs/2310.08442

    Args:
    ----
        loss_weight (float): Weight of this loss item.
            Defaults to ``1.``.
        reduction: (str): The reduction method for the loss.
            Defaults to 'mean'.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'l2'.

    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 reduction: str = "mean",
                 loss_name: str = "debias_estimation") -> None:

        super().__init__()
        assert reduction in ["mean", "none"], (
            f"reduction should be 'mean' or 'none', got {reduction}"
        )
        self.loss_weight = loss_weight
        self.reduction = reduction
        self._loss_name = loss_name

    @property
    def use_snr(self) -> bool:
        """Whether or not this loss uses SNR."""
        return True

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                timesteps: torch.Tensor,
                alphas_cumprod: torch.Tensor,
                prediction_type: str,
                weight: torch.Tensor | None = None) -> torch.Tensor:
        """Forward function.

        Args:
        ----
            pred (torch.Tensor): The predicted tensor.
            gt (torch.Tensor): The ground truth tensor.
            timesteps (torch.Tensor): The timestep tensor.
            alphas_cumprod (torch.Tensor): The alphas_cumprod from the
                scheduler.
            prediction_type (str): The prediction type from scheduler.
            weight (torch.Tensor | None, optional): The loss weight.
                Defaults to None.

        Returns:
        -------
            torch.Tensor: loss

        """
        snr = compute_snr(timesteps, alphas_cumprod)
        if prediction_type == "v_prediction":
            # Velocity objective requires that we add one to SNR values before
            # we divide by them.
            snr = snr + 1
        mse_loss_weights = 1 / torch.sqrt(snr)
        loss = F.mse_loss(pred, gt, reduction="none")
        loss = loss.mean(
            dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        if weight is not None:
            loss = loss * weight
        if self.reduction == "mean":
            loss = loss.mean()
        return loss * self.loss_weight
