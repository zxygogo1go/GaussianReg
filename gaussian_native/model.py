"""End-to-end Gaussian-native diffeomorphic registration model."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .correspondence import HierarchicalGaussianCorrespondence
from .decomposition import HierarchicalGaussianDecomposer
from .encoding import HierarchicalGaussianEncoder
from .integration import ScalingAndSquaring, warp_tensor
from .velocity import (
    GaussianVelocityHead,
    HierarchicalGaussianVelocitySynthesis,
)


def _autocast_disabled(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=False)
    return torch.cuda.amp.autocast(enabled=False)


class GaussianNativeRegistration(nn.Module):
    """Gaussian representation → correspondence → SVF → diffeomorphism."""

    architecture_revision = "gaussian_native_v1"

    def __init__(
        self,
        inshape: Sequence[int] = (128, 160, 160),
        spacing_dhw: Sequence[float] = (1.5, 1.5, 1.5),
        root_grid_shape: Sequence[int] = (4, 4, 4),
        children_per_parent: int = 4,
        feature_dim: int = 96,
        hidden_dim: int = 128,
        graph_heads: int = 4,
        graph_neighbors: int = 16,
        graph_blocks_per_level: int = 2,
        samples_per_axis: int = 3,
        pyramid_factors: Sequence[int] = (16, 8, 4),
        sinkhorn_iterations: int = 12,
        match_temperature: float = 0.08,
        position_weight: float = 0.12,
        scale_weight: float = 0.04,
        dustbin_mass: float = 0.15,
        parent_candidates: int = 4,
        velocity_hidden_dim: int = 192,
        raster_chunk: int = 64,
        cutoff_sigma: float = 3.5,
        integration_steps: int = 7,
        motion_mode: str = "affine",
        integration_mode: str = "svf",
        covariance_mode: str = "full",
    ) -> None:
        super().__init__()
        self.inshape = tuple(int(value) for value in inshape)
        self.pyramid_factors = tuple(int(value) for value in pyramid_factors)
        self.integration_mode = str(integration_mode).strip().lower()
        if self.integration_mode not in {"svf", "direct"}:
            raise ValueError("integration_mode must be svf or direct")
        if len(self.inshape) != 3 or any(value % max(self.pyramid_factors) for value in self.inshape):
            raise ValueError("inshape must contain three values divisible by the largest pyramid factor")
        spacing = torch.tensor(tuple(float(value) for value in spacing_dhw), dtype=torch.float32)
        if spacing.numel() != 3 or float(spacing.min()) <= 0.0:
            raise ValueError("spacing_dhw must contain three positive values")
        self.register_buffer("spacing_dhw", spacing, persistent=True)
        self.decomposer = HierarchicalGaussianDecomposer(
            root_grid_shape=root_grid_shape,
            children_per_parent=children_per_parent,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            pyramid_factors=pyramid_factors,
            samples_per_axis=samples_per_axis,
            raster_chunk=raster_chunk,
            covariance_mode=covariance_mode,
        )
        self.encoder = HierarchicalGaussianEncoder(
            feature_dim=feature_dim,
            heads=graph_heads,
            neighbors=graph_neighbors,
            blocks_per_level=graph_blocks_per_level,
        )
        self.correspondence = HierarchicalGaussianCorrespondence(
            feature_dim=feature_dim,
            temperature=match_temperature,
            position_weight=position_weight,
            scale_weight=scale_weight,
            dustbin_mass=dustbin_mass,
            sinkhorn_iterations=sinkhorn_iterations,
            parent_candidates=parent_candidates,
            children_per_parent=children_per_parent,
        )
        self.velocity_head = GaussianVelocityHead(
            feature_dim=feature_dim,
            hidden_dim=velocity_hidden_dim,
            children_per_parent=children_per_parent,
            motion_mode=motion_mode,
        )
        self.velocity_synthesis = HierarchicalGaussianVelocitySynthesis(
            node_chunk=raster_chunk,
            cutoff_sigma=cutoff_sigma,
        )
        self.integration = ScalingAndSquaring(steps=integration_steps)

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        return_aux: bool = False,
    ):
        if moving.shape != fixed.shape or moving.ndim != 5 or moving.shape[1] != 1:
            raise AssertionError("moving/fixed must share shape [B,1,D,H,W]")
        if tuple(moving.shape[2:]) != self.inshape:
            raise AssertionError(
                "input shape %s does not match configured shape %s"
                % (tuple(moving.shape[2:]), self.inshape)
            )
        spacing = self.spacing_dhw.to(device=moving.device).view(1, 3).expand(
            moving.shape[0], -1
        )
        with _autocast_disabled(moving.device):
            moving_decomposition = self.decomposer(
                moving.float(),
                spacing,
                compute_reconstruction=return_aux,
            )
            fixed_decomposition = self.decomposer(
                fixed.float(),
                spacing,
                compute_reconstruction=return_aux,
            )
        moving_levels = self.encoder(
            moving_decomposition["levels"],
            moving_decomposition["extent_mm"],
        )
        fixed_levels = self.encoder(
            fixed_decomposition["levels"],
            fixed_decomposition["extent_mm"],
        )
        matches = self.correspondence(
            fixed_levels,
            moving_levels,
            fixed_decomposition["extent_mm"],
        )
        local_velocities = self.velocity_head(fixed_levels, matches)
        synthesis_shape = fixed_decomposition["pyramid_images"][-1].shape[2:]
        with _autocast_disabled(moving.device):
            synthesis = self.velocity_synthesis(
                fixed_levels,
                local_velocities,
                synthesis_shape,
                fixed_decomposition["extent_mm"],
            )
            velocity_mm = F.interpolate(
                synthesis["velocity_mm"],
                size=self.inshape,
                mode="trilinear",
                align_corners=True,
            )
            velocity_vox = velocity_mm / spacing.view(moving.shape[0], 3, 1, 1, 1)
            if self.integration_mode == "svf":
                flow = self.integration(velocity_vox)
                inverse_flow = self.integration(-velocity_vox)
            else:
                flow = velocity_vox
                inverse_flow = -velocity_vox
            warped = warp_tensor(moving.float(), flow, padding_mode="zeros")
            inverse_warped = warp_tensor(fixed.float(), inverse_flow, padding_mode="zeros")
        if not return_aux:
            return warped, flow
        moving_decomposition["levels"] = moving_levels
        fixed_decomposition["levels"] = fixed_levels
        return {
            "warped": warped,
            "flow": flow,
            "inverse_warped": inverse_warped,
            "inverse_flow": inverse_flow,
            "velocity_mm": velocity_mm,
            "velocity_vox": velocity_vox,
            "moving_decomposition": moving_decomposition,
            "fixed_decomposition": fixed_decomposition,
            "correspondence": matches,
            "local_velocities": local_velocities,
            "level_velocity_mm": synthesis["level_velocity_mm"],
            "level_velocity_coverage": synthesis["level_coverage"],
        }


__all__ = ["GaussianNativeRegistration"]
