"""Reusable losses for training a local residual refiner."""

from typing import Optional

import torch
import torch.nn.functional as F

from .functional import binary_dice_per_batch, masked_mse_per_batch


def binary_dice_loss_per_batch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return 1.0 - binary_dice_per_batch(pred, target, eps=eps)


def weighted_gradient_loss_per_batch(
    dvf: torch.Tensor,
    weight_map: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted first-order DVF smoothness loss for every batch item."""

    weight_map = weight_map.float()
    dx = dvf[:, :, 1:, :, :] - dvf[:, :, :-1, :, :]
    dy = dvf[:, :, :, 1:, :] - dvf[:, :, :, :-1, :]
    dz = dvf[:, :, :, :, 1:] - dvf[:, :, :, :, :-1]
    wx = 0.5 * (weight_map[:, :, 1:, :, :] + weight_map[:, :, :-1, :, :])
    wy = 0.5 * (weight_map[:, :, :, 1:, :] + weight_map[:, :, :, :-1, :])
    wz = 0.5 * (weight_map[:, :, :, :, 1:] + weight_map[:, :, :, :, :-1])
    reduce_dims = (1, 2, 3, 4)
    channels = dvf.shape[1]
    loss_x = (dx.pow(2) * wx).sum(dim=reduce_dims) / (wx.sum(dim=reduce_dims) * channels + eps)
    loss_y = (dy.pow(2) * wy).sum(dim=reduce_dims) / (wy.sum(dim=reduce_dims) * channels + eps)
    loss_z = (dz.pow(2) * wz).sum(dim=reduce_dims) / (wz.sum(dim=reduce_dims) * channels + eps)
    return (loss_x + loss_y + loss_z) / 3.0


def weighted_magnitude_loss_per_batch(
    dvf: torch.Tensor,
    weight_map: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Weighted residual-DVF magnitude loss for every batch item."""

    reduce_dims = (1, 2, 3, 4)
    channels = dvf.shape[1]
    weight_map = weight_map.float()
    return (dvf.pow(2) * weight_map).sum(dim=reduce_dims) / (
        weight_map.sum(dim=reduce_dims) * channels + eps
    )


def jacobian_determinant(dvf: torch.Tensor) -> torch.Tensor:
    """Compute the forward-difference Jacobian determinant of id + DVF."""

    if dvf.ndim != 5 or dvf.shape[1] != 3:
        raise ValueError(f"Expected DVF shape (B,3,D,H,W), got {tuple(dvf.shape)}")
    base = dvf[:, :, :-1, :-1, :-1]
    du_dx = dvf[:, :, 1:, :-1, :-1] - base
    du_dy = dvf[:, :, :-1, 1:, :-1] - base
    du_dz = dvf[:, :, :-1, :-1, 1:] - base

    j00 = 1.0 + du_dx[:, 0]
    j01 = du_dy[:, 0]
    j02 = du_dz[:, 0]
    j10 = du_dx[:, 1]
    j11 = 1.0 + du_dy[:, 1]
    j12 = du_dz[:, 1]
    j20 = du_dx[:, 2]
    j21 = du_dy[:, 2]
    j22 = 1.0 + du_dz[:, 2]
    return (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )


def jacobian_hinge_loss_per_batch(
    dvf: torch.Tensor,
    roi_gate: Optional[torch.Tensor] = None,
    margin: float = 0.05,
    roi_weight: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize low or non-positive Jacobian determinants."""

    jacobian = jacobian_determinant(dvf)
    penalty = F.relu(float(margin) - jacobian).pow(2).unsqueeze(1)
    global_loss = penalty.flatten(1).mean(dim=1)
    if roi_gate is None:
        return global_loss
    roi_inner = (roi_gate[:, :, :-1, :-1, :-1] > 1e-4).float()
    roi_loss = (penalty * roi_inner).flatten(1).sum(dim=1) / roi_inner.flatten(1).sum(dim=1).clamp_min(eps)
    return global_loss + float(roi_weight) * roi_loss


__all__ = [
    "binary_dice_loss_per_batch",
    "jacobian_determinant",
    "jacobian_hinge_loss_per_batch",
    "masked_mse_per_batch",
    "weighted_gradient_loss_per_batch",
    "weighted_magnitude_loss_per_batch",
]
