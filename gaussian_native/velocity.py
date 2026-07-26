"""Gaussian-local affine stationary velocity prediction and synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn

from .geometry import (
    GaussianLevel,
    physical_grid,
    skew_matrix,
    symmetric_matrix,
)


@dataclass
class GaussianVelocityParameters:
    translation_mm: torch.Tensor
    linear: torch.Tensor
    rotation_vector: torch.Tensor
    strain_parameters: torch.Tensor


class GaussianVelocityHead(nn.Module):
    """Predict a bounded local affine SVF from Gaussian correspondence."""

    def __init__(
        self,
        feature_dim: int = 96,
        hidden_dim: int = 192,
        children_per_parent: int = 4,
        translation_fraction: float = 0.40,
        correspondence_fraction: float = 0.0,
        max_rotation_radians: float = 0.20,
        max_strain: float = 0.08,
        motion_mode: str = "affine",
    ) -> None:
        super().__init__()
        self.children_per_parent = int(children_per_parent)
        self.translation_fraction = float(translation_fraction)
        self.correspondence_fraction = float(correspondence_fraction)
        self.max_rotation_radians = float(max_rotation_radians)
        self.max_strain = float(max_strain)
        self.motion_mode = str(motion_mode).strip().lower()
        if self.motion_mode not in {"translation", "se3", "affine"}:
            raise ValueError("motion_mode must be translation, se3, or affine")
        input_dim = 2 * feature_dim + 15
        self.level_embedding = nn.Parameter(torch.zeros(3, feature_dim))
        nn.init.normal_(self.level_embedding, std=0.02)
        self.network = nn.Sequential(
            nn.Linear(input_dim + feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 12),
        )
        final = self.network[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _zero_parent_mean(self, value: torch.Tensor) -> torch.Tensor:
        batch, nodes = value.shape[:2]
        children = self.children_per_parent
        if nodes % children:
            raise AssertionError("child count must be divisible by children_per_parent")
        grouped = value.reshape(batch, nodes // children, children, *value.shape[2:])
        grouped = grouped - grouped.mean(dim=2, keepdim=True)
        return grouped.reshape_as(value)

    def forward(
        self,
        fixed_levels: List[GaussianLevel],
        correspondence: List[dict],
    ) -> List[GaussianVelocityParameters]:
        parameters = []
        for level_index, (fixed, match) in enumerate(zip(fixed_levels, correspondence)):
            delta = match["matched_center_mm"] - fixed.centers_mm
            normalized_delta = delta / fixed.scales_mm.clamp_min(1.0e-3)
            log_scale_ratio = torch.log(
                match["matched_scale_mm"].clamp_min(1.0e-3)
                / fixed.scales_mm.clamp_min(1.0e-3)
            )
            relative_covariance = (
                fixed.precision_mm2 @ match["matched_covariance_mm2"]
            )
            inputs = torch.cat(
                (
                    fixed.features,
                    match["matched_feature"],
                    normalized_delta,
                    log_scale_ratio,
                    relative_covariance.reshape(*relative_covariance.shape[:-2], 9),
                    self.level_embedding[level_index].view(1, 1, -1).expand(
                        fixed.features.shape[0],
                        fixed.features.shape[1],
                        -1,
                    ),
                ),
                dim=-1,
            )
            raw = self.network(inputs)
            translation_residual = (
                torch.tanh(raw[..., 0:3])
                * fixed.scales_mm
                * self.translation_fraction
            )
            if level_index == 0:
                bounded_delta = torch.maximum(
                    torch.minimum(delta, 2.0 * fixed.scales_mm),
                    -2.0 * fixed.scales_mm,
                )
                translation = (
                    translation_residual
                    + self.correspondence_fraction * bounded_delta
                )
                rotation_vector = (
                    torch.tanh(raw[..., 3:6]) * self.max_rotation_radians
                )
                strain_parameters = torch.tanh(raw[..., 6:12]) * self.max_strain
            else:
                translation = self._zero_parent_mean(translation_residual)
                rotation_vector = self._zero_parent_mean(
                    torch.tanh(raw[..., 3:6]) * self.max_rotation_radians
                )
                strain_parameters = self._zero_parent_mean(
                    torch.tanh(raw[..., 6:12]) * self.max_strain
                )
            if self.motion_mode == "translation":
                rotation_vector = torch.zeros_like(rotation_vector)
                strain_parameters = torch.zeros_like(strain_parameters)
            elif self.motion_mode == "se3":
                strain_parameters = torch.zeros_like(strain_parameters)
            linear = skew_matrix(rotation_vector) + symmetric_matrix(strain_parameters)
            parameters.append(
                GaussianVelocityParameters(
                    translation_mm=translation,
                    linear=linear,
                    rotation_vector=rotation_vector,
                    strain_parameters=strain_parameters,
                )
            )
        return parameters


class GaussianVelocityRasterizer(nn.Module):
    """Synthesize a dense physical SVF from local Gaussian affine velocities."""

    def __init__(self, node_chunk: int = 64, cutoff_sigma: float = 3.5) -> None:
        super().__init__()
        self.node_chunk = int(node_chunk)
        self.cutoff_squared = float(cutoff_sigma) ** 2

    def forward(
        self,
        level: GaussianLevel,
        velocity: GaussianVelocityParameters,
        output_shape: Sequence[int],
        extent_mm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = physical_grid(output_shape, extent_mm.float())
        batch, voxels = grid.shape[:2]
        numerator = grid.new_zeros(batch, voxels, 3)
        denominator = grid.new_zeros(batch, voxels)
        nodes = int(level.centers_mm.shape[1])
        scaled_mass = level.mass.float() * float(nodes)
        for start in range(0, nodes, self.node_chunk):
            stop = min(start + self.node_chunk, nodes)
            delta = grid.unsqueeze(1) - level.centers_mm[:, start:stop].float().unsqueeze(2)
            mahalanobis = torch.einsum(
                "bcvi,bcij,bcvj->bcv",
                delta,
                level.precision_mm2[:, start:stop].float(),
                delta,
            )
            weights = torch.exp(-0.5 * mahalanobis)
            weights = torch.where(
                mahalanobis <= self.cutoff_squared,
                weights,
                torch.zeros_like(weights),
            )
            weights = weights * scaled_mass[:, start:stop].unsqueeze(-1)
            local_velocity = velocity.translation_mm[:, start:stop].float().unsqueeze(2)
            local_velocity = local_velocity + torch.einsum(
                "bcij,bcvj->bcvi",
                velocity.linear[:, start:stop].float(),
                delta,
            )
            numerator = numerator + (weights.unsqueeze(-1) * local_velocity).sum(dim=1)
            denominator = denominator + weights.sum(dim=1)
        field = numerator / denominator.clamp_min(1.0e-6).unsqueeze(-1)
        field = field.reshape(batch, *output_shape, 3).permute(0, 4, 1, 2, 3)
        coverage = (1.0 - torch.exp(-denominator)).reshape(batch, 1, *output_shape)
        return field, coverage


class HierarchicalGaussianVelocitySynthesis(nn.Module):
    """Fixed additive coarse-to-fine Gaussian velocity hierarchy."""

    def __init__(self, node_chunk: int = 64, cutoff_sigma: float = 3.5) -> None:
        super().__init__()
        self.rasterizer = GaussianVelocityRasterizer(
            node_chunk=node_chunk,
            cutoff_sigma=cutoff_sigma,
        )

    def forward(
        self,
        levels: List[GaussianLevel],
        parameters: List[GaussianVelocityParameters],
        output_shape: Sequence[int],
        extent_mm: torch.Tensor,
    ) -> dict:
        fields, coverages = [], []
        for level, local_velocity in zip(levels, parameters):
            field, coverage = self.rasterizer(
                level,
                local_velocity,
                output_shape,
                extent_mm,
            )
            fields.append(field)
            coverages.append(coverage)
        return {
            "level_velocity_mm": fields,
            "level_coverage": coverages,
            "velocity_mm": sum(fields),
        }


__all__ = [
    "GaussianVelocityHead",
    "GaussianVelocityParameters",
    "GaussianVelocityRasterizer",
    "HierarchicalGaussianVelocitySynthesis",
]
