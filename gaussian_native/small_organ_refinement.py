"""Small-organ-adaptive Gaussian residual diffeomorphic refinement.

The module consumes only images and Gaussian outputs at inference.  Training
segmentations may supervise the returned parent-priority logits externally,
but they are deliberately absent from this forward interface.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .geometry import (
    GaussianLevel,
    covariance_from_scale_rotation,
    physical_to_normalized,
)
from .integration import ScalingAndSquaring, warp_tensor
from .velocity import GaussianVelocityParameters, GaussianVelocityRasterizer


def _child_offset_lattice(children: int) -> torch.Tensor:
    """Return a compact, symmetric three-dimensional child layout."""
    children = int(children)
    if children == 1:
        return torch.zeros(1, 3, dtype=torch.float32)
    if children == 7:
        axes = torch.eye(3, dtype=torch.float32)
        return torch.cat((torch.zeros(1, 3), axes, -axes), dim=0)
    if children == 9:
        corners = torch.tensor(
            [
                [d, h, w]
                for d in (-1.0, 1.0)
                for h in (-1.0, 1.0)
                for w in (-1.0, 1.0)
            ],
            dtype=torch.float32,
        ) / math.sqrt(3.0)
        return torch.cat((torch.zeros(1, 3), corners), dim=0)
    if children == 27:
        axis = torch.tensor((-1.0, 0.0, 1.0), dtype=torch.float32)
        dd, hh, ww = torch.meshgrid(axis, axis, axis, indexing="ij")
        return torch.stack((dd, hh, ww), dim=-1).reshape(-1, 3) / math.sqrt(3.0)
    raise ValueError("children_per_parent must be one of 1, 7, 9, or 27")


def _forward_difference(volume: torch.Tensor, axis: int) -> torch.Tensor:
    if axis == 0:
        return F.pad(
            volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :],
            (0, 0, 0, 0, 0, 1),
        )
    if axis == 1:
        return F.pad(
            volume[:, :, :, 1:, :] - volume[:, :, :, :-1, :],
            (0, 0, 0, 1, 0, 0),
        )
    if axis == 2:
        return F.pad(
            volume[:, :, :, :, 1:] - volume[:, :, :, :, :-1],
            (0, 1, 0, 0, 0, 0),
        )
    raise ValueError("axis must be zero, one, or two")


def analytic_measurements(
    volume: torch.Tensor,
    spacing_dhw: torch.Tensor,
) -> torch.Tensor:
    """Return fixed intensity/derivative measurements without a dense CNN."""
    if volume.ndim != 5 or volume.shape[1] != 1:
        raise AssertionError("analytic measurements require [B,1,D,H,W]")
    spacing = spacing_dhw.to(
        device=volume.device,
        dtype=torch.float32,
    )
    if spacing.ndim == 1:
        spacing = spacing.unsqueeze(0).expand(volume.shape[0], -1)
    if spacing.shape != (volume.shape[0], 3):
        raise AssertionError("spacing_dhw must have shape [B,3]")
    value = volume.float()
    derivatives = [
        _forward_difference(value, axis)
        / spacing[:, axis].reshape(-1, 1, 1, 1, 1).clamp_min(1.0e-6)
        for axis in range(3)
    ]
    second = [
        _forward_difference(derivative, axis)
        / spacing[:, axis].reshape(-1, 1, 1, 1, 1).clamp_min(1.0e-6)
        for axis, derivative in enumerate(derivatives)
    ]
    gradient = torch.sqrt(
        sum(derivative.square() for derivative in derivatives) + 1.0e-8
    )
    local_mean = F.avg_pool3d(value, kernel_size=3, stride=1, padding=1)
    local_square_mean = F.avg_pool3d(
        value.square(),
        kernel_size=3,
        stride=1,
        padding=1,
    )
    local_variance = (local_square_mean - local_mean.square()).clamp_min(0.0)
    return torch.cat(
        (
            value,
            derivatives[0],
            derivatives[1],
            derivatives[2],
            gradient,
            sum(second),
            local_variance,
        ),
        dim=1,
    )


def sample_volume_at_physical_points(
    volume: torch.Tensor,
    points_mm: torch.Tensor,
    extent_mm: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Sample `[B,C,D,H,W]` at arbitrary physical DHW points.

    The output shape is `points_mm.shape[:-1] + (C,)`.
    """
    if volume.ndim != 5 or points_mm.ndim < 3 or points_mm.shape[-1] != 3:
        raise AssertionError("invalid volume or point shape")
    if volume.shape[0] != points_mm.shape[0] or extent_mm.shape != (
        volume.shape[0],
        3,
    ):
        raise AssertionError("batch-aware extent must have shape [B,3]")
    leading = points_mm.shape[1:-1]
    flat = points_mm.reshape(points_mm.shape[0], -1, 3)
    normalized_dhw = physical_to_normalized(flat, extent_mm)
    grid = normalized_dhw[..., [2, 1, 0]].reshape(
        points_mm.shape[0],
        -1,
        1,
        1,
        3,
    )
    sampled = F.grid_sample(
        volume.float(),
        grid,
        mode=mode,
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled[:, :, :, 0, 0].transpose(1, 2)
    return sampled.reshape(points_mm.shape[0], *leading, volume.shape[1])


def _batch_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2 or indices.ndim != 2 or values.shape[0] != indices.shape[0]:
        raise AssertionError("values must be [B,N,...] and indices [B,K]")
    batch = torch.arange(values.shape[0], device=values.device).view(-1, 1)
    return values[batch, indices]


def _bounded_vectors(value: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    length = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    scale = (float(maximum_norm) / length.clamp_min(1.0e-6)).clamp(max=1.0)
    return value * scale


class SmallOrganAdaptiveGaussianRefiner(nn.Module):
    """Densify selected fine Gaussians and predict a local residual SVF."""

    def __init__(
        self,
        feature_dim: int,
        selected_parents: int = 48,
        children_per_parent: int = 9,
        descriptor_dim: int = 64,
        hidden_dim: int = 128,
        search_radius_fraction: float = 0.45,
        minimum_search_radius_mm: float = 2.0,
        maximum_search_radius_mm: float = 6.0,
        child_scale_fraction: float = 0.35,
        maximum_residual_mm: float = 3.0,
        synthesis_factor: int = 2,
        match_temperature: float = 0.15,
        position_weight: float = 0.10,
        mismatch_prior_weight: float = 0.50,
        raster_chunk: int = 32,
        cutoff_sigma: float = 3.0,
        integration_steps: int = 7,
        adaptive_priority: bool = True,
        local_correspondence: bool = True,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or descriptor_dim <= 0 or hidden_dim < 2:
            raise ValueError("SAGR feature dimensions must be positive")
        if selected_parents <= 0 or synthesis_factor <= 0:
            raise ValueError("SAGR selection and synthesis factor must be positive")
        if (
            search_radius_fraction <= 0.0
            or minimum_search_radius_mm <= 0.0
            or maximum_search_radius_mm < minimum_search_radius_mm
            or child_scale_fraction <= 0.0
            or maximum_residual_mm <= 0.0
            or match_temperature <= 0.0
            or position_weight < 0.0
            or mismatch_prior_weight < 0.0
        ):
            raise ValueError("invalid SAGR geometry or matching configuration")
        offsets = _child_offset_lattice(children_per_parent)
        self.register_buffer("child_offsets", offsets, persistent=True)
        self.selected_parents = int(selected_parents)
        self.children_per_parent = int(children_per_parent)
        self.search_radius_fraction = float(search_radius_fraction)
        self.minimum_search_radius_mm = float(minimum_search_radius_mm)
        self.maximum_search_radius_mm = float(maximum_search_radius_mm)
        self.child_scale_fraction = float(child_scale_fraction)
        self.maximum_residual_mm = float(maximum_residual_mm)
        self.synthesis_factor = int(synthesis_factor)
        self.match_temperature = float(match_temperature)
        self.position_weight = float(position_weight)
        self.mismatch_prior_weight = float(mismatch_prior_weight)
        self.adaptive_priority = bool(adaptive_priority)
        self.local_correspondence = bool(local_correspondence)

        priority_input_dim = 3 * int(feature_dim) + 13
        self.priority_head = nn.Sequential(
            nn.LayerNorm(priority_input_dim),
            nn.Linear(priority_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        priority_final = self.priority_head[-1]
        nn.init.zeros_(priority_final.weight)
        nn.init.zeros_(priority_final.bias)
        self.parent_projection = nn.Linear(feature_dim, descriptor_dim)
        self.measurement_projection = nn.Linear(7, descriptor_dim)
        self.offset_projection = nn.Linear(3, descriptor_dim)
        self.descriptor_network = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.GELU(),
            nn.Linear(descriptor_dim, descriptor_dim),
            nn.GELU(),
            nn.LayerNorm(descriptor_dim),
        )
        velocity_input_dim = 2 * descriptor_dim + 4
        self.velocity_head = nn.Sequential(
            nn.LayerNorm(velocity_input_dim),
            nn.Linear(velocity_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        final = self.velocity_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.direct_gain_raw = nn.Parameter(torch.zeros(()))
        self.rasterizer = GaussianVelocityRasterizer(
            node_chunk=int(raster_chunk),
            cutoff_sigma=float(cutoff_sigma),
            use_canonical_basis=False,
        )
        self.integration = ScalingAndSquaring(steps=int(integration_steps))

    @staticmethod
    def _measurement_pyramid(
        volume: torch.Tensor,
        spacing_dhw: torch.Tensor,
        factor: int,
    ) -> torch.Tensor:
        if any(int(size) % int(factor) for size in volume.shape[2:]):
            raise ValueError("input shape must be divisible by SAGR synthesis factor")
        size = tuple(int(value) // int(factor) for value in volume.shape[2:])
        reduced = F.interpolate(
            volume.float(),
            size=size,
            mode="trilinear",
            align_corners=True,
        )
        spacing_scale = spacing_dhw.new_tensor(
            [
                float(original - 1) / float(max(reduced_size - 1, 1))
                for original, reduced_size in zip(volume.shape[2:], size)
            ]
        )
        return analytic_measurements(
            reduced,
            spacing_dhw.float() * spacing_scale.view(1, 3),
        )

    def _priority(
        self,
        fixed_level: GaussianLevel,
        match: Mapping[str, torch.Tensor],
        fixed_measurements: torch.Tensor,
        warped_measurements: torch.Tensor,
        extent_mm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fixed_sample = sample_volume_at_physical_points(
            fixed_measurements,
            fixed_level.centers_mm,
            extent_mm,
        )
        warped_sample = sample_volume_at_physical_points(
            warped_measurements,
            fixed_level.centers_mm,
            extent_mm,
        )
        measurement_delta = (fixed_sample - warped_sample).abs()
        normalized_motion = (
            match["transport_delta_mm"].float()
            / fixed_level.scales_mm.float().clamp_min(1.0e-3)
        )
        scalar_metrics = torch.stack(
            (
                match["match_evidence"].float(),
                match["support_entropy"].float(),
                match["matched_mass_fraction"].float(),
            ),
            dim=-1,
        )
        priority_inputs = torch.cat(
            (
                fixed_level.features.float(),
                match["matched_feature"].float(),
                (
                    fixed_level.features.float()
                    - match["matched_feature"].float()
                ).abs(),
                normalized_motion,
                scalar_metrics,
                measurement_delta,
            ),
            dim=-1,
        )
        learned_logits = self.priority_head(priority_inputs).squeeze(-1)
        mismatch = measurement_delta.mean(dim=-1)
        mismatch_mean = mismatch.mean(dim=1, keepdim=True)
        mismatch_scale = mismatch.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).clamp_min(1.0e-4)
        mismatch_logits = ((mismatch - mismatch_mean) / mismatch_scale).clamp(
            -4.0,
            4.0,
        )
        priority_logits = (
            learned_logits
            + self.mismatch_prior_weight * mismatch_logits.detach()
        )
        priority_score = torch.sigmoid(priority_logits)
        return priority_logits, priority_score, torch.sigmoid(mismatch_logits)

    def _selected_indices(self, priority: torch.Tensor) -> torch.Tensor:
        nodes = int(priority.shape[1])
        selected = min(self.selected_parents, nodes)
        if self.adaptive_priority:
            return torch.topk(
                priority,
                k=selected,
                dim=1,
                largest=True,
                sorted=True,
            ).indices
        base = torch.linspace(
            0,
            max(nodes - 1, 0),
            selected,
            device=priority.device,
        ).round().long()
        return base.view(1, -1).expand(priority.shape[0], -1)

    def _descriptors(
        self,
        parent_features: torch.Tensor,
        measurements: torch.Tensor,
        normalized_offsets: torch.Tensor,
    ) -> torch.Tensor:
        parent = self.parent_projection(parent_features).unsqueeze(2)
        measurement = self.measurement_projection(measurements)
        offset = self.offset_projection(normalized_offsets)
        return F.normalize(
            self.descriptor_network(parent + measurement + offset),
            dim=-1,
        )

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        base_flow: torch.Tensor,
        fixed_level: GaussianLevel,
        match: Mapping[str, torch.Tensor],
        spacing_dhw: torch.Tensor,
        extent_mm: torch.Tensor,
    ) -> dict:
        if moving.shape != fixed.shape or moving.ndim != 5:
            raise AssertionError("SAGR moving/fixed images must share [B,1,D,H,W]")
        batch = int(moving.shape[0])
        spacing = spacing_dhw.to(
            device=moving.device,
            dtype=torch.float32,
        )
        if spacing.ndim == 1:
            spacing = spacing.unsqueeze(0).expand(batch, -1)
        base_warped = warp_tensor(
            moving.float(),
            base_flow.float(),
            padding_mode="zeros",
        )
        fixed_measurements = self._measurement_pyramid(
            fixed,
            spacing,
            self.synthesis_factor,
        )
        warped_measurements = self._measurement_pyramid(
            base_warped,
            spacing,
            self.synthesis_factor,
        )
        priority_logits, priority, mismatch_target = self._priority(
            fixed_level,
            match,
            fixed_measurements,
            warped_measurements,
            extent_mm,
        )
        selected_indices = self._selected_indices(priority)
        selected_priority = _batch_gather(priority, selected_indices)
        selected_centers = _batch_gather(
            fixed_level.centers_mm.float(),
            selected_indices,
        )
        selected_scales = _batch_gather(
            fixed_level.scales_mm.float(),
            selected_indices,
        )
        selected_rotations = _batch_gather(
            fixed_level.rotations.float(),
            selected_indices,
        )
        selected_fixed_features = _batch_gather(
            fixed_level.features.float(),
            selected_indices,
        )
        selected_moving_features = _batch_gather(
            match["matched_feature"].float(),
            selected_indices,
        )
        radius = (
            selected_scales.mean(dim=-1, keepdim=True)
            * self.search_radius_fraction
        ).clamp(
            min=self.minimum_search_radius_mm,
            max=self.maximum_search_radius_mm,
        )
        normalized_offsets = self.child_offsets.to(
            device=moving.device,
            dtype=torch.float32,
        ).view(1, 1, self.children_per_parent, 3).expand(
            batch,
            selected_indices.shape[1],
            -1,
            -1,
        )
        local_offsets = normalized_offsets * radius.unsqueeze(2)
        world_offsets = torch.einsum(
            "bkij,bkmj->bkmi",
            selected_rotations,
            local_offsets,
        )
        child_centers = selected_centers.unsqueeze(2) + world_offsets
        child_centers = torch.minimum(
            torch.maximum(child_centers, torch.zeros_like(child_centers)),
            extent_mm[:, None, None, :].float(),
        )
        child_scale = (
            selected_scales.unsqueeze(2) * self.child_scale_fraction
        )
        minimum_scale = spacing[:, None, None, :] * 0.75
        child_scales = torch.maximum(child_scale, minimum_scale).expand(
            -1,
            -1,
            self.children_per_parent,
            -1,
        )
        fixed_child_measurements = sample_volume_at_physical_points(
            fixed_measurements,
            child_centers,
            extent_mm,
        )
        moving_child_measurements = sample_volume_at_physical_points(
            warped_measurements,
            child_centers,
            extent_mm,
        )
        fixed_descriptor = self._descriptors(
            selected_fixed_features,
            fixed_child_measurements,
            normalized_offsets,
        )
        moving_descriptor = self._descriptors(
            selected_moving_features,
            moving_child_measurements,
            normalized_offsets,
        )
        descriptor_similarity = torch.einsum(
            "bkmf,bknf->bkmn",
            fixed_descriptor,
            moving_descriptor,
        )
        normalized_center_delta = (
            child_centers.unsqueeze(3) - child_centers.unsqueeze(2)
        ) / radius.unsqueeze(2).unsqueeze(3).clamp_min(1.0e-3)
        position_cost = normalized_center_delta.square().sum(dim=-1)
        local_logits = (
            descriptor_similarity - self.position_weight * position_cost
        ) / self.match_temperature
        if self.local_correspondence:
            row = torch.softmax(local_logits, dim=3)
            column = torch.softmax(local_logits, dim=2)
            local_transport = row * column
            local_transport = local_transport / local_transport.sum(
                dim=3,
                keepdim=True,
            ).clamp_min(1.0e-8)
        else:
            identity = torch.eye(
                self.children_per_parent,
                device=moving.device,
                dtype=torch.float32,
            )
            local_transport = identity.view(
                1,
                1,
                self.children_per_parent,
                self.children_per_parent,
            ).expand(batch, selected_indices.shape[1], -1, -1)
        matched_centers = torch.einsum(
            "bkmn,bknd->bkmd",
            local_transport,
            child_centers,
        )
        matched_descriptor = torch.einsum(
            "bkmn,bknf->bkmf",
            local_transport,
            moving_descriptor,
        )
        direct_residual_mm = matched_centers - child_centers
        normalized_residual = direct_residual_mm / radius.unsqueeze(2).clamp_min(
            1.0e-3
        )
        effective_priority = (
            selected_priority
            if self.adaptive_priority
            else torch.ones_like(selected_priority)
        )
        priority_gate = effective_priority.unsqueeze(2).unsqueeze(-1).expand(
            -1,
            -1,
            self.children_per_parent,
            -1,
        )
        velocity_inputs = torch.cat(
            (
                fixed_descriptor,
                matched_descriptor,
                normalized_residual,
                priority_gate,
            ),
            dim=-1,
        )
        learned_residual_mm = (
            torch.tanh(self.velocity_head(velocity_inputs))
            * (0.5 * self.maximum_residual_mm)
        )
        direct_gain = torch.tanh(self.direct_gain_raw)
        child_velocity_mm = priority_gate * (
            direct_gain * direct_residual_mm + learned_residual_mm
        )
        child_velocity_mm = _bounded_vectors(
            child_velocity_mm,
            self.maximum_residual_mm,
        )

        children = int(selected_indices.shape[1]) * self.children_per_parent
        flat_centers = child_centers.reshape(batch, children, 3)
        flat_scales = child_scales.reshape(batch, children, 3)
        flat_rotations = selected_rotations.unsqueeze(2).expand(
            -1,
            -1,
            self.children_per_parent,
            -1,
            -1,
        ).reshape(batch, children, 3, 3)
        covariance, precision = covariance_from_scale_rotation(
            flat_scales,
            flat_rotations,
        )
        flat_priority = effective_priority.unsqueeze(-1).expand(
            -1,
            -1,
            self.children_per_parent,
        ).reshape(batch, children)
        mass = flat_priority / flat_priority.sum(dim=1, keepdim=True).clamp_min(
            1.0e-6
        )
        densified_level = GaussianLevel(
            centers_mm=flat_centers,
            covariance_mm2=covariance,
            precision_mm2=precision,
            scales_mm=flat_scales,
            rotations=flat_rotations,
            mass=mass,
            features=fixed_descriptor.reshape(batch, children, -1),
            appearance=fixed_child_measurements.reshape(batch, children, -1),
            parent_index=None,
            anchor_centers_mm=flat_centers,
            anchor_scales_mm=flat_scales,
        )
        flat_velocity = child_velocity_mm.reshape(batch, children, 3)
        zero_linear = flat_velocity.new_zeros(batch, children, 3, 3)
        zero_rotation = flat_velocity.new_zeros(batch, children, 3)
        zero_strain = flat_velocity.new_zeros(batch, children, 6)
        velocity_parameters = GaussianVelocityParameters(
            translation_mm=flat_velocity,
            linear=zero_linear,
            rotation_vector=zero_rotation,
            strain_parameters=zero_strain,
            direct_translation_mm=(
                priority_gate * direct_gain * direct_residual_mm
            ).reshape(batch, children, 3),
            learned_translation_mm=(
                priority_gate * learned_residual_mm
            ).reshape(batch, children, 3),
            motion_evidence=flat_priority,
        )
        synthesis_shape = tuple(
            int(value) // self.synthesis_factor
            for value in moving.shape[2:]
        )
        local_velocity_low_mm, local_coverage_low = self.rasterizer(
            densified_level,
            velocity_parameters,
            synthesis_shape,
            extent_mm,
        )
        local_velocity_mm = F.interpolate(
            local_velocity_low_mm,
            size=moving.shape[2:],
            mode="trilinear",
            align_corners=True,
        )
        local_velocity_vox = local_velocity_mm / spacing.reshape(
            batch,
            3,
            1,
            1,
            1,
        )
        local_flow = self.integration(local_velocity_vox)
        local_inverse_flow = self.integration(-local_velocity_vox)
        local_entropy = -(
            local_transport
            * local_transport.clamp_min(1.0e-8).log()
        ).sum(dim=3) / max(math.log(float(self.children_per_parent)), 1.0)
        return {
            "base_warped": base_warped,
            "priority_logits": priority_logits,
            "priority": priority,
            "mismatch_priority_target": mismatch_target,
            "selected_parent_indices": selected_indices,
            "selected_priority": selected_priority,
            "selected_effective_priority": effective_priority,
            "child_centers_mm": child_centers,
            "child_scales_mm": child_scales,
            "local_transport": local_transport,
            "local_transport_entropy": local_entropy,
            "child_velocity_mm": child_velocity_mm,
            "densified_level": densified_level,
            "local_velocity_low_mm": local_velocity_low_mm,
            "local_velocity_mm": local_velocity_mm,
            "local_velocity_vox": local_velocity_vox,
            "local_coverage_low": local_coverage_low,
            "local_flow": local_flow,
            "local_inverse_flow": local_inverse_flow,
            "direct_gain": direct_gain,
        }


__all__ = [
    "SmallOrganAdaptiveGaussianRefiner",
    "analytic_measurements",
    "sample_volume_at_physical_points",
]
