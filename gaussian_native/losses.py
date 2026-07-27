"""Grouped unsupervised objective for Gaussian-native registration."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .integration import compose_displacements


class LocalNCCLoss(nn.Module):
    """Negative local normalized cross-correlation in float32."""

    def __init__(self, window: int = 9, epsilon: float = 1.0e-5) -> None:
        super().__init__()
        if window <= 0 or window % 2 == 0:
            raise ValueError("NCC window must be a positive odd integer")
        self.window = int(window)
        self.epsilon = float(epsilon)

    def forward(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        if target.is_cuda and torch.is_autocast_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                return self.forward(target.float(), prediction.float())
        target = target.float()
        prediction = prediction.float()
        channels = int(target.shape[1])
        kernel = target.new_ones(channels, 1, self.window, self.window, self.window)
        padding = self.window // 2
        window_volume = float(self.window ** 3)
        target_sum = F.conv3d(target, kernel, padding=padding, groups=channels)
        prediction_sum = F.conv3d(prediction, kernel, padding=padding, groups=channels)
        target_square_sum = F.conv3d(target.square(), kernel, padding=padding, groups=channels)
        prediction_square_sum = F.conv3d(
            prediction.square(), kernel, padding=padding, groups=channels
        )
        product_sum = F.conv3d(target * prediction, kernel, padding=padding, groups=channels)
        target_mean = target_sum / window_volume
        prediction_mean = prediction_sum / window_volume
        cross = product_sum - target_sum * prediction_sum / window_volume
        target_variance = target_square_sum - 2.0 * target_mean * target_sum
        target_variance = target_variance + target_mean.square() * window_volume
        prediction_variance = prediction_square_sum - 2.0 * prediction_mean * prediction_sum
        prediction_variance = prediction_variance + prediction_mean.square() * window_volume
        correlation = cross.square() / (
            target_variance.clamp_min(0.0)
            * prediction_variance.clamp_min(0.0)
            + self.epsilon
        )
        return -correlation.mean()


def _gradient_components(volume: torch.Tensor) -> Sequence[torch.Tensor]:
    return torch.gradient(volume.float(), dim=(2, 3, 4), edge_order=1)


def normalized_gradient_loss(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    target_gradient = torch.stack(_gradient_components(target), dim=1)
    prediction_gradient = torch.stack(_gradient_components(prediction), dim=1)
    dot = (target_gradient * prediction_gradient).sum(dim=1)
    target_norm = target_gradient.square().sum(dim=1)
    prediction_norm = prediction_gradient.square().sum(dim=1)
    agreement = dot.square() / (target_norm * prediction_norm + 1.0e-5)
    return (1.0 - agreement).mean()


def velocity_smoothness(velocity: torch.Tensor) -> torch.Tensor:
    gradients = torch.gradient(velocity.float(), dim=(2, 3, 4), edge_order=1)
    return sum(gradient.square().mean() for gradient in gradients) / 3.0


def jacobian_barrier(flow: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    center = flow[:, :, :-1, :-1, :-1].float()
    dd = flow[:, :, 1:, :-1, :-1].float() - center
    dh = flow[:, :, :-1, 1:, :-1].float() - center
    dw = flow[:, :, :-1, :-1, 1:].float() - center
    j00, j01, j02 = 1.0 + dd[:, 0], dh[:, 0], dw[:, 0]
    j10, j11, j12 = dd[:, 1], 1.0 + dh[:, 1], dw[:, 1]
    j20, j21, j22 = dd[:, 2], dh[:, 2], 1.0 + dw[:, 2]
    determinant = (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )
    return F.relu(float(margin) - determinant).square().mean()


class GaussianNativeObjective(nn.Module):
    """Four interpretable loss groups used by the Gaussian-native model."""

    def __init__(self, config: Mapping[str, object]) -> None:
        super().__init__()
        self.weights = {
            "similarity": float(config.get("similarity", 1.0)),
            "representation": float(config.get("representation", 0.15)),
            "correspondence": float(config.get("correspondence", 0.05)),
            "deformation": float(config.get("deformation", 0.05)),
        }
        if any(value < 0.0 for value in self.weights.values()):
            raise ValueError("loss weights must be nonnegative")
        self.ngf_weight = float(config.get("ngf_weight", 0.15))
        self.coverage_weight = float(config.get("coverage_weight", 0.10))
        self.containment_weight = float(config.get("containment_weight", 0.10))
        self.anchor_offset_weight = float(config.get("anchor_offset_weight", 0.0))
        if self.anchor_offset_weight < 0.0:
            raise ValueError("anchor_offset_weight must be nonnegative")
        self.cycle_weight = float(config.get("cycle_weight", 0.50))
        self.hierarchy_weight = float(config.get("hierarchy_weight", 0.50))
        self.inverse_weight = float(config.get("inverse_weight", 0.50))
        self.jacobian_weight = float(config.get("jacobian_weight", 0.20))
        self.velocity_energy_weight = float(config.get("velocity_energy_weight", 0.01))
        self.motion_hierarchy_weight = float(
            config.get("motion_hierarchy_weight", 0.0)
        )
        if self.motion_hierarchy_weight < 0.0:
            raise ValueError("motion_hierarchy_weight must be nonnegative")
        self.jacobian_margin = float(config.get("jacobian_margin", 0.05))
        self.ncc = LocalNCCLoss(window=int(config.get("ncc_window", 9)))

    def _multi_scale_similarity(
        self,
        target: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        weights = (0.50, 0.30, 0.20)
        result = target.new_zeros((), dtype=torch.float32)
        for index, weight in enumerate(weights):
            if index:
                factor = 2 ** index
                size = tuple(max(4, int(value) // factor) for value in target.shape[2:])
                current_target = F.interpolate(
                    target,
                    size=size,
                    mode="trilinear",
                    align_corners=True,
                )
                current_prediction = F.interpolate(
                    prediction,
                    size=size,
                    mode="trilinear",
                    align_corners=True,
                )
            else:
                current_target, current_prediction = target, prediction
            result = result + float(weight) * self.ncc(current_target, current_prediction)
        edge_size = tuple(max(4, int(value) // 2) for value in target.shape[2:])
        target_edge = F.interpolate(target, size=edge_size, mode="trilinear", align_corners=True)
        prediction_edge = F.interpolate(
            prediction,
            size=edge_size,
            mode="trilinear",
            align_corners=True,
        )
        return result + self.ngf_weight * normalized_gradient_loss(
            target_edge,
            prediction_edge,
        )

    def _representation(self, output: Mapping[str, object]) -> torch.Tensor:
        losses = []
        for key in ("moving_decomposition", "fixed_decomposition"):
            decomposition = output[key]
            for level, target in zip(
                decomposition["levels"],
                decomposition["pyramid_images"],
            ):
                if level.reconstruction is None or level.coverage is None:
                    raise AssertionError("training output must include Gaussian reconstructions")
                reconstruction = F.smooth_l1_loss(level.reconstruction.float(), target.float())
                coverage = F.relu(0.65 - level.coverage.float()).square().mean()
                losses.append(reconstruction + self.coverage_weight * coverage)
        representation = sum(losses) / float(len(losses))

        containment = representation.new_zeros(())
        anchor_offset = representation.new_zeros(())
        count = 0
        anchor_count = 0
        for key in ("moving_decomposition", "fixed_decomposition"):
            levels = output[key]["levels"]
            for level in levels:
                if level.anchor_centers_mm is None or level.anchor_scales_mm is None:
                    continue
                normalized_anchor_offset = (
                    (level.centers_mm - level.anchor_centers_mm)
                    / level.anchor_scales_mm.clamp_min(1.0e-3)
                )
                anchor_offset = anchor_offset + normalized_anchor_offset.square().mean()
                anchor_count += 1
            for child, parent in zip(levels[1:], levels[:-1]):
                parent_index = child.parent_index
                parent_center = parent.centers_mm[:, parent_index]
                parent_scale = parent.scales_mm[:, parent_index]
                parent_rotation = parent.rotations[:, parent_index]
                delta = child.centers_mm - parent_center
                local = torch.einsum(
                    "bkji,bki->bkj",
                    parent_rotation,
                    delta,
                ) / parent_scale.clamp_min(1.0e-3)
                center_penalty = F.relu(
                    torch.linalg.vector_norm(local, dim=-1) - 1.75
                ).square().mean()
                scale_penalty = F.relu(
                    child.scales_mm / parent_scale.clamp_min(1.0e-3) - 0.85
                ).square().mean()
                containment = containment + center_penalty + scale_penalty
                count += 1
        return (
            representation
            + self.containment_weight * containment / float(max(count, 1))
            + self.anchor_offset_weight
            * anchor_offset
            / float(max(anchor_count, 1))
        )

    def _correspondence(self, output: Mapping[str, object]) -> torch.Tensor:
        results = output["correspondence"]
        transport = sum(result["transport_cost"] for result in results) / float(len(results))
        cycle = sum(result["cycle_error"] for result in results) / float(len(results))
        hierarchy = sum(result["hierarchy_error"] for result in results[1:]) / float(
            max(len(results) - 1, 1)
        )
        return transport + self.cycle_weight * cycle + self.hierarchy_weight * hierarchy

    def _motion_hierarchy(self, output: Mapping[str, object]) -> torch.Tensor:
        """Softly centre additive child residuals without cancelling them."""
        parameters = output["local_velocities"]
        levels = output["fixed_decomposition"]["levels"]
        penalty = output["velocity_vox"].new_zeros((), dtype=torch.float32)
        count = 0
        for child_level, parent_level, child_motion in zip(
            levels[1:],
            levels[:-1],
            parameters[1:],
        ):
            children = int(child_level.centers_mm.shape[1] // parent_level.centers_mm.shape[1])
            batch, parent_nodes = parent_level.centers_mm.shape[:2]
            weights = child_level.mass.float().reshape(batch, parent_nodes, children)
            weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1.0e-8)

            def weighted_parent_mean(value: torch.Tensor) -> torch.Tensor:
                grouped = value.float().reshape(
                    batch,
                    parent_nodes,
                    children,
                    *value.shape[2:],
                )
                weight_shape = (*weights.shape, *([1] * (value.ndim - 2)))
                return (grouped * weights.reshape(weight_shape)).sum(dim=2)

            translation_mean = weighted_parent_mean(child_motion.translation_mm)
            normalized_translation = (
                translation_mean / parent_level.scales_mm.float().clamp_min(1.0e-3)
            )
            rotation_mean = weighted_parent_mean(child_motion.rotation_vector)
            strain_mean = weighted_parent_mean(child_motion.strain_parameters)
            penalty = penalty + normalized_translation.square().mean()
            penalty = penalty + rotation_mean.square().mean()
            penalty = penalty + strain_mean.square().mean()
            count += 1
        return penalty / float(max(count, 1))

    def _deformation(self, output: Mapping[str, object]) -> torch.Tensor:
        velocity = output["velocity_vox"].float()
        flow = output["flow"].float()
        inverse = output["inverse_flow"].float()
        inverse_residual = compose_displacements(flow, inverse)
        inverse_consistency = inverse_residual.square().mean()
        smoothness = velocity_smoothness(velocity)
        energy = velocity.square().mean()
        topology = jacobian_barrier(flow, margin=self.jacobian_margin)
        motion_hierarchy = self._motion_hierarchy(output)
        return (
            smoothness
            + self.inverse_weight * inverse_consistency
            + self.jacobian_weight * topology
            + self.velocity_energy_weight * energy
            + self.motion_hierarchy_weight * motion_hierarchy
        )

    def forward(
        self,
        output: Mapping[str, object],
        moving: torch.Tensor,
        fixed: torch.Tensor,
    ) -> dict:
        forward_similarity = self._multi_scale_similarity(fixed, output["warped"])
        reverse_similarity = self._multi_scale_similarity(moving, output["inverse_warped"])
        terms = {
            "similarity": 0.5 * (forward_similarity + reverse_similarity),
            "representation": self._representation(output),
            "correspondence": self._correspondence(output),
            "deformation": self._deformation(output),
        }
        terms["total"] = sum(self.weights[name] * value for name, value in terms.items())
        return terms


__all__ = [
    "GaussianNativeObjective",
    "LocalNCCLoss",
    "jacobian_barrier",
    "normalized_gradient_loss",
    "velocity_smoothness",
]
