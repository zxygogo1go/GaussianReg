"""End-to-end Gaussian-native diffeomorphic registration model."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn

from .correspondence import HierarchicalGaussianCorrespondence
from .decomposition import HierarchicalGaussianDecomposer
from .encoding import HierarchicalGaussianEncoder
from .integration import ScalingAndSquaring, warp_tensor
from .refinement import GaussianGuidedResidualPyramid
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

    architecture_revision = "gaussian_native_v3"

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
        geometry_mode: str = "adaptive",
        architecture_revision: str = "gaussian_native_v3",
        appearance_weight: float = 0.0,
        transport_mode: str = "sinkhorn",
        correspondence_score_mode: str = "convex",
        feature_residual_weight: float = 0.0,
        max_feature_residual_logit: float = 2.0,
        pair_score_hidden_dim: int = 32,
        pair_context_dim: Optional[int] = None,
        pair_score_heads: int = 4,
        pair_fusion_hidden_dim: Optional[int] = None,
        pair_context_temperature: float = 0.20,
        refinement_factors: Sequence[int] = (8, 4, 2),
        refinement_channels: Sequence[int] = (48, 40, 32),
        refinement_blocks_per_stage: Union[int, Sequence[int]] = 3,
        refinement_maximum_residual_vox: Sequence[float] = (
            1.5,
            1.0,
            0.75,
        ),
        refinement_use_gradient_features: bool = False,
        include_identity_candidate: Optional[bool] = None,
        match_evidence_power: float = 1.0,
        direct_displacement_fractions: Sequence[float] = (1.0, 1.0, 1.0),
        direct_displacement_limit: float = 1.5,
        direct_displacement_limits_mm: Optional[Sequence[float]] = (12.0, 6.0, 3.0),
        learned_translation_fractions: Optional[Sequence[float]] = (0.20, 0.12, 0.08),
        max_rotation_radians: Optional[float] = None,
        max_strain: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.architecture_revision = str(architecture_revision).strip().lower()
        if self.architecture_revision not in {
            "gaussian_native_v1",
            "gaussian_native_v2",
            "gaussian_native_v3",
            "gaussian_native_v4",
            "gaussian_native_v5",
            "gaussian_native_v6",
            "gaussian_native_v7",
            "gaussian_native_v8",
            "gaussian_native_v9",
            "gaussian_native_v10",
            "gaussian_native_v11",
        }:
            raise ValueError("unsupported Gaussian-native architecture revision")
        use_calibrated_motion = self.architecture_revision in {
            "gaussian_native_v2",
            "gaussian_native_v3",
            "gaussian_native_v4",
            "gaussian_native_v5",
            "gaussian_native_v6",
            "gaussian_native_v7",
            "gaussian_native_v8",
            "gaussian_native_v9",
            "gaussian_native_v10",
            "gaussian_native_v11",
        }
        use_stable_motion_basis = self.architecture_revision in {
            "gaussian_native_v3",
            "gaussian_native_v4",
            "gaussian_native_v5",
            "gaussian_native_v6",
            "gaussian_native_v7",
            "gaussian_native_v8",
            "gaussian_native_v9",
            "gaussian_native_v10",
            "gaussian_native_v11",
        }
        use_v3_motion = self.architecture_revision == "gaussian_native_v3"
        use_anatomical_motion = self.architecture_revision in {
            "gaussian_native_v4",
            "gaussian_native_v5",
            "gaussian_native_v6",
            "gaussian_native_v7",
            "gaussian_native_v8",
            "gaussian_native_v9",
            "gaussian_native_v10",
            "gaussian_native_v11",
        }
        use_sparse_appearance_motion = self.architecture_revision in {
            "gaussian_native_v5",
            "gaussian_native_v6",
            "gaussian_native_v7",
            "gaussian_native_v8",
            "gaussian_native_v9",
            "gaussian_native_v10",
            "gaussian_native_v11",
        }
        use_v5_motion = self.architecture_revision == "gaussian_native_v5"
        use_residual_pair_motion = self.architecture_revision in {
            "gaussian_native_v7",
            "gaussian_native_v8",
            "gaussian_native_v9",
            "gaussian_native_v10",
            "gaussian_native_v11",
        }
        use_pyramid_refinement = (
            self.architecture_revision
            in {"gaussian_native_v10", "gaussian_native_v11"}
        )
        if include_identity_candidate is None:
            include_identity_candidate = use_v5_motion
        rotation_limit = (
            0.08
            if max_rotation_radians is None and use_stable_motion_basis
            else 0.20
            if max_rotation_radians is None
            else float(max_rotation_radians)
        )
        strain_limit = (
            0.04
            if max_strain is None and use_stable_motion_basis
            else 0.08
            if max_strain is None
            else float(max_strain)
        )
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
            geometry_mode=(
                geometry_mode if use_residual_pair_motion else "adaptive"
            ),
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
            identity_calibration=use_calibrated_motion,
            calibration_gradient=use_v3_motion,
            # V4 matches and measures motion in the learned anatomical Gaussian
            # frame.  The stopped fixed-to-fixed reference removes entropic
            # barycentre bias while preserving moving-to-fixed centre drift.
            coordinate_mode=(
                "canonical"
                if use_v3_motion or use_residual_pair_motion
                else "learned"
            ),
            mutual_transport=use_v3_motion,
            detach_geometry_cost=use_anatomical_motion,
            appearance_weight=(
                appearance_weight if use_sparse_appearance_motion else 0.0
            ),
            transport_mode=(
                transport_mode if use_sparse_appearance_motion else "sinkhorn"
            ),
            shared_calibration_candidates=use_sparse_appearance_motion,
            include_identity_candidate=(
                bool(include_identity_candidate)
                if use_sparse_appearance_motion
                else False
            ),
            score_mode=(
                correspondence_score_mode
                if use_residual_pair_motion
                else "convex"
            ),
            feature_residual_weight=(
                feature_residual_weight
                if use_residual_pair_motion
                else 0.0
            ),
            max_feature_residual_logit=max_feature_residual_logit,
            pair_score_hidden_dim=pair_score_hidden_dim,
            pair_context_dim=pair_context_dim,
            pair_score_heads=pair_score_heads,
            pair_fusion_hidden_dim=pair_fusion_hidden_dim,
            pair_context_temperature=pair_context_temperature,
        )
        self.velocity_head = GaussianVelocityHead(
            feature_dim=feature_dim,
            hidden_dim=velocity_hidden_dim,
            children_per_parent=children_per_parent,
            motion_mode=motion_mode,
            hierarchy_mode=(
                "soft_residual" if use_calibrated_motion else "hard_centered"
            ),
            direct_displacement_fractions=direct_displacement_fractions,
            direct_displacement_limit=direct_displacement_limit,
            direct_displacement_limits_mm=(
                direct_displacement_limits_mm if use_stable_motion_basis else None
            ),
            learned_translation_fractions=(
                learned_translation_fractions if use_stable_motion_basis else None
            ),
            max_rotation_radians=rotation_limit,
            max_strain=strain_limit,
            use_match_evidence=use_sparse_appearance_motion,
            match_evidence_power=match_evidence_power,
        )
        self.velocity_synthesis = HierarchicalGaussianVelocitySynthesis(
            node_chunk=raster_chunk,
            cutoff_sigma=cutoff_sigma,
            use_canonical_basis=use_stable_motion_basis,
        )
        self.integration = ScalingAndSquaring(steps=integration_steps)
        self.residual_pyramid = (
            GaussianGuidedResidualPyramid(
                factors=refinement_factors,
                channels=refinement_channels,
                blocks_per_stage=refinement_blocks_per_stage,
                maximum_residual_vox=(
                    refinement_maximum_residual_vox
                ),
                integration_steps=integration_steps,
                use_gradient_features=(
                    refinement_use_gradient_features
                ),
            )
            if use_pyramid_refinement
            else None
        )

    def set_correspondence_temperature(self, temperature: float) -> None:
        self.correspondence.set_temperature(temperature)

    def set_correspondence_appearance_weight(
        self,
        appearance_weight: float,
    ) -> None:
        self.correspondence.set_appearance_weight(appearance_weight)

    def set_correspondence_feature_residual_weight(
        self,
        feature_residual_weight: float,
    ) -> None:
        self.correspondence.set_feature_residual_weight(
            feature_residual_weight
        )

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
        refinement = None
        if self.residual_pyramid is not None:
            refinement = self.residual_pyramid(
                moving,
                fixed,
                synthesis["level_velocity_mm"],
                spacing,
            )
            velocity_vox = refinement["velocity_vox"]
            flow = refinement["flow"]
            inverse_flow = refinement["inverse_flow"]
            velocity_mm = velocity_vox * spacing.view(
                moving.shape[0],
                3,
                1,
                1,
                1,
            )
        else:
            with _autocast_disabled(moving.device):
                velocity_mm = F.interpolate(
                    synthesis["velocity_mm"],
                    size=self.inshape,
                    mode="trilinear",
                    align_corners=True,
                )
                velocity_vox = velocity_mm / spacing.view(
                    moving.shape[0],
                    3,
                    1,
                    1,
                    1,
                )
                if self.integration_mode == "svf":
                    flow = self.integration(velocity_vox)
                    inverse_flow = self.integration(-velocity_vox)
                else:
                    flow = velocity_vox
                    inverse_flow = -velocity_vox
        with _autocast_disabled(moving.device):
            warped = warp_tensor(moving.float(), flow, padding_mode="zeros")
            inverse_warped = warp_tensor(fixed.float(), inverse_flow, padding_mode="zeros")
        if not return_aux:
            return warped, flow
        moving_decomposition["levels"] = moving_levels
        fixed_decomposition["levels"] = fixed_levels
        result = {
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
        if refinement is not None:
            result.update(
                {
                    "pyramid_factors": refinement["pyramid_factors"],
                    "pyramid_velocity_vox": refinement[
                        "pyramid_velocity_vox"
                    ],
                    "pyramid_residual_velocity_vox": refinement[
                        "pyramid_residual_velocity_vox"
                    ],
                    "pyramid_flow": refinement["pyramid_flow"],
                    "pyramid_warped": refinement["pyramid_warped"],
                }
            )
        return result


__all__ = ["GaussianNativeRegistration"]
