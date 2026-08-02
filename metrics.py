"""Registration metrics for head-and-neck experiments."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_per_class(
    prediction: np.ndarray,
    target: np.ndarray,
    labels: Iterable[int],
    response_aware: bool = True,
    valid_labels: Optional[Iterable[int]] = None,
) -> Dict[int, float]:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    scores: Dict[int, float] = {}
    valid = None if valid_labels is None else {int(label) for label in valid_labels}
    for label in labels:
        if valid is not None and int(label) not in valid:
            continue
        pred_mask = prediction == int(label)
        target_mask = target == int(label)
        # Without an explicit eligibility set, retain the convenient behavior
        # for standalone callers. Registration evaluation should pass
        # ``valid_labels`` derived from the *unwarped* moving/fixed masks so a
        # failed warp that removes a structure is scored as zero, not skipped.
        if valid is None and response_aware and (not pred_mask.any() or not target_mask.any()):
            continue
        denominator = int(pred_mask.sum()) + int(target_mask.sum())
        scores[int(label)] = 1.0 if denominator == 0 else float(2.0 * np.logical_and(pred_mask, target_mask).sum() / denominator)
    return scores


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def surface_distance_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_dhw: Sequence[float],
) -> Dict[str, float]:
    pred_mask = np.asarray(prediction, dtype=bool)
    target_mask = np.asarray(target, dtype=bool)
    if pred_mask.shape != target_mask.shape:
        raise ValueError("prediction and target shapes must match")
    if pred_mask.ndim != 3:
        raise ValueError("surface metrics require 3D masks")
    if len(spacing_dhw) != 3 or min(float(value) for value in spacing_dhw) <= 0.0:
        raise ValueError("spacing_dhw must contain three positive values")
    if not pred_mask.any() and not target_mask.any():
        return {"hd95": 0.0, "assd": 0.0}
    if not pred_mask.any() or not target_mask.any():
        # An eligible structure lost by the warp is a catastrophic failure,
        # not missing data. Use the physical image diagonal as a transparent,
        # finite surface-distance penalty so cohort summaries cannot hide it.
        diagonal = float(
            np.sqrt(
                sum(
                    ((int(size) - 1) * float(spacing)) ** 2
                    for size, spacing in zip(pred_mask.shape, spacing_dhw)
                )
            )
        )
        return {"hd95": diagonal, "assd": diagonal}
    pred_surface = _surface(pred_mask)
    target_surface = _surface(target_mask)
    distance_to_target = distance_transform_edt(~target_surface, sampling=tuple(float(v) for v in spacing_dhw))
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=tuple(float(v) for v in spacing_dhw))
    pred_to_target = distance_to_target[pred_surface]
    target_to_pred = distance_to_pred[target_surface]
    distances = np.concatenate((pred_to_target, target_to_pred))
    return {
        "hd95": float(np.percentile(distances, 95.0)),
        "assd": float(0.5 * (pred_to_target.mean() + target_to_pred.mean())),
    }


def jacobian_determinant(
    flow: torch.Tensor,
    spacing_dhw: Sequence[float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Jacobian determinant of ``identity + flow`` for DHW voxel flow."""
    if flow.ndim != 5 or flow.shape[1] != 3:
        raise AssertionError("flow must have shape [B,3,D,H,W]")
    if len(spacing_dhw) != 3 or min(float(v) for v in spacing_dhw) <= 0.0:
        raise ValueError("spacing_dhw must contain three positive values")
    gradients = torch.gradient(flow.float(), dim=(2, 3, 4), edge_order=1)
    spacing = flow.new_tensor([float(v) for v in spacing_dhw], dtype=torch.float32)
    jacobian = torch.zeros(
        flow.shape[0],
        *flow.shape[2:],
        3,
        3,
        device=flow.device,
        dtype=torch.float32,
    )
    for output_axis in range(3):
        for spatial_axis in range(3):
            physical_scale = spacing[output_axis] / spacing[spatial_axis]
            jacobian[..., output_axis, spatial_axis] = gradients[spatial_axis][:, output_axis] * physical_scale
            if output_axis == spatial_axis:
                jacobian[..., output_axis, spatial_axis] += 1.0
    return torch.linalg.det(jacobian)


def jacobian_metrics(
    flow: torch.Tensor,
    spacing_dhw: Sequence[float] = (1.0, 1.0, 1.0),
    safe_threshold: float = 0.5,
) -> Dict[str, float]:
    determinant = jacobian_determinant(flow, spacing_dhw=spacing_dhw)
    return {
        "negative_jacobian_ratio": float((determinant <= 0.0).float().mean().detach().cpu()),
        "below_safe_jacobian_ratio": float((determinant < float(safe_threshold)).float().mean().detach().cpu()),
        "minimum_jacobian": float(determinant.min().detach().cpu()),
        "mean_jacobian": float(determinant.mean().detach().cpu()),
    }


def evaluate_segmentation_pair(
    prediction: np.ndarray,
    target: np.ndarray,
    labels: Iterable[int],
    spacing_dhw: Sequence[float],
    response_aware: bool = True,
    valid_labels: Optional[Iterable[int]] = None,
    compute_surface: bool = True,
) -> Dict[str, object]:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    label_values = tuple(int(label) for label in labels)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if prediction.ndim == 3:
        masks = {
            label: (prediction == label, target == label)
            for label in label_values
        }
    elif prediction.ndim == 4 and prediction.shape[0] == len(label_values):
        masks = {
            label: (prediction[index] > 0, target[index] > 0)
            for index, label in enumerate(label_values)
        }
    else:
        raise ValueError(
            "segmentations must be 3D label maps or [number_of_labels,D,H,W] binary arrays"
        )
    valid = None if valid_labels is None else {
        int(label) for label in valid_labels
    }
    dice: Dict[int, float] = {}
    selected_masks = {}
    for label, (pred_mask, target_mask) in masks.items():
        if valid is not None and label not in valid:
            continue
        if valid is None and response_aware and (
            not pred_mask.any() or not target_mask.any()
        ):
            continue
        denominator = int(pred_mask.sum()) + int(target_mask.sum())
        dice[label] = (
            1.0
            if denominator == 0
            else float(
                2.0 * np.logical_and(pred_mask, target_mask).sum()
                / denominator
            )
        )
        selected_masks[label] = (pred_mask, target_mask)
    surfaces: Dict[int, Dict[str, float]] = {}
    if compute_surface:
        for label, (pred_mask, target_mask) in selected_masks.items():
            surfaces[label] = surface_distance_metrics(
                pred_mask,
                target_mask,
                spacing_dhw,
            )
    dice_values = list(dice.values())
    hd95_values = [value["hd95"] for value in surfaces.values() if np.isfinite(value["hd95"])]
    assd_values = [value["assd"] for value in surfaces.values() if np.isfinite(value["assd"])]
    return {
        "dice_per_class": dice,
        "surface_per_class": surfaces,
        "mean_dice": float(np.mean(dice_values)) if dice_values else float("nan"),
        "median_dice": float(np.median(dice_values)) if dice_values else float("nan"),
        "mean_hd95": float(np.mean(hd95_values)) if hd95_values else float("nan"),
        "mean_assd": float(np.mean(assd_values)) if assd_values else float("nan"),
    }


__all__ = [
    "dice_per_class",
    "surface_distance_metrics",
    "jacobian_determinant",
    "jacobian_metrics",
    "evaluate_segmentation_pair",
]
