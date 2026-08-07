"""Tensor operations for anatomy-aware local residual refinement."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


INPUT_MODES: Tuple[str, ...] = ("full", "no-fixed-small", "no-fixed-seg")


def binary_dice_per_batch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Return a soft binary Dice score for every batch item."""

    pred = pred.float()
    target = target.float()
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    denominator = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return torch.where(denominator <= eps, torch.ones_like(dice), dice)


def masked_mse_per_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return masked MSE for every batch item."""

    weight = mask.float()
    while weight.ndim < pred.ndim:
        weight = weight.unsqueeze(1)
    numerator = ((pred.float() - target.float()).pow(2) * weight).flatten(1).sum(dim=1)
    denominator = weight.flatten(1).sum(dim=1) * pred.shape[1] + eps
    return numerator / denominator


def estimate_ct_only_difficulty(
    moving_image: torch.Tensor,
    fixed_image: torch.Tensor,
    warped_moving_image: torch.Tensor,
    base_dvf: torch.Tensor,
    weights: Tuple[float, float, float] = (0.35, 0.45, 0.20),
    flow_scale: float = 20.0,
) -> torch.Tensor:
    """Estimate residual difficulty without fixed segmentation features."""

    initial_mse = (moving_image.float() - fixed_image.float()).pow(2).flatten(1).mean(dim=1).clamp(0.0, 1.0)
    residual_mse = (
        (warped_moving_image.float() - fixed_image.float()).pow(2).flatten(1).mean(dim=1).clamp(0.0, 1.0)
    )
    flow_magnitude = torch.sqrt(base_dvf.float().pow(2).sum(dim=1)).flatten(1)
    flow_score = (torch.quantile(flow_magnitude, q=0.95, dim=1) / max(flow_scale, 1e-6)).clamp(0.0, 1.0)
    w_initial, w_residual, w_flow = weights
    return (w_initial * initial_mse + w_residual * residual_mse + w_flow * flow_score).clamp(0.0, 1.0)


def estimate_anatomy_difficulty(
    fixed_image: torch.Tensor,
    warped_moving_image: torch.Tensor,
    base_dvf: torch.Tensor,
    warped_small_mask: torch.Tensor,
    fixed_small_mask: torch.Tensor,
    warped_bone_mask: torch.Tensor,
    fixed_bone_mask: torch.Tensor,
    image_mask: torch.Tensor,
    weights: Tuple[float, float, float, float] = (0.45, 0.20, 0.25, 0.10),
    flow_scale: float = 20.0,
) -> torch.Tensor:
    """Estimate how much local error remains after the base registrar."""

    small_dice = binary_dice_per_batch(warped_small_mask, fixed_small_mask)
    bone_dice = binary_dice_per_batch(warped_bone_mask, fixed_bone_mask)
    image_residual = masked_mse_per_batch(warped_moving_image, fixed_image, image_mask).clamp(0.0, 1.0)
    flow_magnitude = torch.sqrt(base_dvf.float().pow(2).sum(dim=1)).flatten(1)
    flow_score = (torch.quantile(flow_magnitude, q=0.95, dim=1) / max(flow_scale, 1e-6)).clamp(0.0, 1.0)

    w_small, w_bone, w_image, w_flow = weights
    difficulty = (
        w_small * (1.0 - small_dice)
        + w_bone * (1.0 - bone_dice)
        + w_image * image_residual
        + w_flow * flow_score
    )
    return difficulty.clamp(0.0, 1.0)


def difficulty_to_value(difficulty: torch.Tensor, value_min: float, value_max: float) -> torch.Tensor:
    """Linearly map difficulty in [0, 1] to a configured interval."""

    return value_min + difficulty.float().clamp(0.0, 1.0) * (value_max - value_min)


def difficulty_to_radius(difficulty: torch.Tensor, radius_min: int, radius_max: int) -> torch.Tensor:
    """Map each batch item's difficulty to an integer dilation radius."""

    if radius_min < 0 or radius_max < radius_min:
        raise ValueError(f"Invalid radius range: {radius_min}..{radius_max}")
    radii = difficulty_to_value(difficulty.flatten(), float(radius_min), float(radius_max))
    return torch.round(radii).long().clamp(min=radius_min, max=radius_max)


def build_roi_gate(mask: torch.Tensor, radius: int, smooth_steps: int = 2) -> torch.Tensor:
    """Build one smooth, dilated ROI gate from a binary anatomy mask."""

    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask shape (B,1,D,H,W), got {tuple(mask.shape)}")
    if radius < 0:
        raise ValueError(f"radius must be non-negative, got {radius}")

    hard_mask = (mask > 0).float()
    if radius == 0:
        gate = hard_mask
    else:
        kernel_size = 2 * radius + 1
        gate = F.max_pool3d(hard_mask, kernel_size=kernel_size, stride=1, padding=radius)
    for _ in range(max(0, int(smooth_steps))):
        gate = F.avg_pool3d(gate, kernel_size=3, stride=1, padding=1)
        gate = torch.maximum(gate, hard_mask)
    return gate.clamp(0.0, 1.0)


def build_roi_gate_per_batch(mask: torch.Tensor, radii: torch.Tensor, smooth_steps: int = 2) -> torch.Tensor:
    """Build an ROI gate with a separate dilation radius for each pair."""

    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask shape (B,1,D,H,W), got {tuple(mask.shape)}")
    radii = torch.as_tensor(radii, device=mask.device).flatten().long()
    if radii.numel() == 1:
        radii = radii.expand(mask.shape[0])
    if radii.numel() != mask.shape[0]:
        raise ValueError(f"Expected {mask.shape[0]} radii, got {radii.numel()}")
    gates = [
        build_roi_gate(mask[index : index + 1], int(radii[index].item()), smooth_steps=smooth_steps)
        for index in range(mask.shape[0])
    ]
    return torch.cat(gates, dim=0)


def conditioning_masks(
    input_mode: str,
    fixed_small_mask: torch.Tensor,
    warped_small_mask: torch.Tensor,
    fixed_bone_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Apply the selected information policy to features and ROI sources."""

    if input_mode not in INPUT_MODES:
        raise ValueError(f"Unsupported input mode {input_mode!r}; choices={INPUT_MODES}")
    zeros_small = torch.zeros_like(fixed_small_mask)
    zeros_bone = torch.zeros_like(fixed_bone_mask)
    if input_mode == "full":
        return {
            "fixed_small_feature": fixed_small_mask,
            "fixed_bone_feature": fixed_bone_mask,
            "roi_source": torch.maximum(fixed_small_mask, warped_small_mask.detach()),
            "anatomy_bone": fixed_bone_mask,
        }
    if input_mode == "no-fixed-small":
        return {
            "fixed_small_feature": zeros_small,
            "fixed_bone_feature": fixed_bone_mask,
            "roi_source": warped_small_mask.detach(),
            "anatomy_bone": fixed_bone_mask,
        }
    return {
        "fixed_small_feature": zeros_small,
        "fixed_bone_feature": zeros_bone,
        "roi_source": warped_small_mask.detach(),
        "anatomy_bone": zeros_bone,
    }


def normalize_dvf_magnitude(dvf: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return a per-case normalized DVF magnitude channel."""

    magnitude = torch.sqrt(dvf.float().pow(2).sum(dim=1, keepdim=True) + eps)
    scale = torch.quantile(magnitude.flatten(1), q=0.95, dim=1).clamp_min(eps)
    return (magnitude / scale.view(-1, 1, 1, 1, 1)).clamp(0.0, 1.0)


def build_feature_tensor(
    fixed_image: torch.Tensor,
    warped_moving_image: torch.Tensor,
    fixed_small_mask: torch.Tensor,
    warped_small_mask: torch.Tensor,
    base_dvf: torch.Tensor,
    fixed_bone_mask: torch.Tensor,
    warped_bone_mask: torch.Tensor,
) -> torch.Tensor:
    """Assemble the default seven-channel refinement tensor."""

    return torch.cat(
        (
            fixed_image.float(),
            warped_moving_image.float(),
            fixed_small_mask.float(),
            warped_small_mask.float(),
            normalize_dvf_magnitude(base_dvf),
            fixed_bone_mask.float(),
            warped_bone_mask.float(),
        ),
        dim=1,
    )


def build_anatomy_maps(
    fixed_bone_mask: torch.Tensor,
    roi_gate: torch.Tensor,
    difficulty: torch.Tensor,
    smooth_base: float = 1.0,
    smooth_bone: float = 4.0,
    smooth_boundary: float = 2.0,
    smooth_difficulty: float = 1.0,
    magnitude_inside: float = 0.2,
    magnitude_outside: float = 6.0,
    magnitude_bone: float = 4.0,
) -> Dict[str, torch.Tensor]:
    """Create anatomy-conditioned regularization maps."""

    if fixed_bone_mask.shape != roi_gate.shape:
        raise ValueError(f"fixed_bone_mask shape {tuple(fixed_bone_mask.shape)} != roi_gate {tuple(roi_gate.shape)}")
    bone = fixed_bone_mask.float().clamp(0.0, 1.0)
    roi = roi_gate.float().clamp(0.0, 1.0)
    outside_roi = (1.0 - roi).clamp(0.0, 1.0)
    boundary = (roi - F.avg_pool3d(roi, kernel_size=3, stride=1, padding=1)).abs().clamp(0.0, 1.0)
    diff = difficulty.float().view(-1, 1, 1, 1, 1).clamp(0.0, 1.0)

    smooth = (
        smooth_base
        + smooth_bone * (1.0 + 0.5 * diff) * bone
        + smooth_boundary * (1.0 + diff) * boundary
        + smooth_difficulty * diff * roi
    )
    magnitude = (
        magnitude_inside * (1.0 - 0.5 * diff) * roi
        + magnitude_outside * (1.0 + 0.5 * diff) * outside_roi
        + magnitude_bone * (1.0 + 0.5 * diff) * bone
    )
    return {"smooth": smooth, "magnitude": magnitude, "refine": roi * (1.0 + diff)}
