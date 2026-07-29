"""True coarse-to-fine residual refinement for Gaussian registration."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .integration import ScalingAndSquaring, warp_tensor


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    """Small pre-activation 3D residual block with a dilated middle field."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        groups = _group_count(int(channels))
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        update = self.conv1(F.gelu(self.norm1(features)))
        update = self.conv2(F.gelu(self.norm2(update)))
        return features + update


class ResidualVelocityStage(nn.Module):
    """Predict a bounded SVF residual after warping at one pyramid scale."""

    def __init__(
        self,
        channels: int,
        blocks: int,
        maximum_residual_vox: float,
    ) -> None:
        super().__init__()
        if channels <= 0 or blocks <= 0 or maximum_residual_vox <= 0.0:
            raise ValueError("invalid residual velocity stage configuration")
        self.maximum_residual_vox = float(maximum_residual_vox)
        self.stem = nn.Conv3d(6, channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[
                ResidualConvBlock(
                    channels,
                    dilation=2 if index % 2 else 1,
                )
                for index in range(blocks)
            ]
        )
        self.output_norm = nn.GroupNorm(
            _group_count(channels),
            channels,
        )
        self.output = nn.Conv3d(
            channels,
            3,
            kernel_size=3,
            padding=1,
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        current_flow: torch.Tensor,
    ) -> torch.Tensor:
        warped = warp_tensor(
            moving.float(),
            current_flow.float(),
            padding_mode="zeros",
        )
        inputs = torch.cat(
            (
                fixed.float(),
                warped,
                fixed.float() - warped,
                current_flow.float(),
            ),
            dim=1,
        )
        features = self.blocks(self.stem(inputs))
        raw = self.output(F.gelu(self.output_norm(features)))
        return self.maximum_residual_vox * torch.tanh(raw)


class GaussianGuidedResidualPyramid(nn.Module):
    """Inject Gaussian velocities level-by-level and refine after each warp.

    Unlike the earlier one-shot hierarchy, every stage observes the moving
    image already warped by all coarser velocity components. Residuals are
    accumulated as a stationary velocity and integrated at every scale.
    """

    def __init__(
        self,
        factors: Sequence[int] = (8, 4, 2),
        channels: Sequence[int] = (48, 40, 32),
        blocks_per_stage: int = 3,
        maximum_residual_vox: Sequence[float] = (1.5, 1.0, 0.75),
        integration_steps: int = 7,
    ) -> None:
        super().__init__()
        self.factors = tuple(int(value) for value in factors)
        channels = tuple(int(value) for value in channels)
        maximum_residual_vox = tuple(
            float(value) for value in maximum_residual_vox
        )
        if (
            not self.factors
            or len(self.factors) != 3
            or len(channels) != len(self.factors)
            or len(maximum_residual_vox) != len(self.factors)
            or any(value <= 0 for value in self.factors)
            or any(
                coarse <= fine
                for coarse, fine in zip(
                    self.factors,
                    self.factors[1:],
                )
            )
        ):
            raise ValueError(
                "residual pyramid requires three strictly decreasing factors"
            )
        self.stages = nn.ModuleList(
            [
                ResidualVelocityStage(
                    stage_channels,
                    int(blocks_per_stage),
                    stage_limit,
                )
                for stage_channels, stage_limit in zip(
                    channels,
                    maximum_residual_vox,
                )
            ]
        )
        self.integration = ScalingAndSquaring(
            steps=int(integration_steps)
        )

    @staticmethod
    def _scaled_size(
        full_shape: Sequence[int],
        factor: int,
    ) -> tuple[int, int, int]:
        if any(int(value) % int(factor) for value in full_shape):
            raise ValueError(
                "input shape must be divisible by every pyramid factor"
            )
        return tuple(int(value) // int(factor) for value in full_shape)

    def forward(
        self,
        moving: torch.Tensor,
        fixed: torch.Tensor,
        gaussian_level_velocity_mm: Sequence[torch.Tensor],
        spacing_dhw: torch.Tensor,
    ) -> dict:
        if len(gaussian_level_velocity_mm) != len(self.factors):
            raise AssertionError(
                "one Gaussian velocity field is required per pyramid stage"
            )
        batch = int(moving.shape[0])
        spacing = spacing_dhw.to(
            device=moving.device,
            dtype=torch.float32,
        ).reshape(batch, 3, 1, 1, 1)
        current_velocity = None
        previous_factor = None
        stage_velocities = []
        stage_residuals = []
        stage_flows = []
        stage_warped = []
        for factor, stage, gaussian_velocity_mm in zip(
            self.factors,
            self.stages,
            gaussian_level_velocity_mm,
        ):
            size = self._scaled_size(moving.shape[2:], factor)
            moving_scale = F.interpolate(
                moving.float(),
                size=size,
                mode="trilinear",
                align_corners=True,
            )
            fixed_scale = F.interpolate(
                fixed.float(),
                size=size,
                mode="trilinear",
                align_corners=True,
            )
            gaussian_velocity = F.interpolate(
                gaussian_velocity_mm.float(),
                size=size,
                mode="trilinear",
                align_corners=True,
            )
            gaussian_velocity = gaussian_velocity / spacing / float(
                factor
            )
            if current_velocity is None:
                current_velocity = gaussian_velocity
            else:
                ratio = float(previous_factor) / float(factor)
                current_velocity = (
                    F.interpolate(
                        current_velocity,
                        size=size,
                        mode="trilinear",
                        align_corners=True,
                    )
                    * ratio
                    + gaussian_velocity
                )
            current_flow = self.integration(current_velocity)
            residual = stage(
                moving_scale,
                fixed_scale,
                current_flow,
            )
            current_velocity = current_velocity + residual.float()
            current_flow = self.integration(current_velocity)
            stage_velocities.append(current_velocity)
            stage_residuals.append(residual)
            stage_flows.append(current_flow)
            stage_warped.append(
                warp_tensor(
                    moving_scale,
                    current_flow,
                    padding_mode="zeros",
                )
            )
            previous_factor = factor

        final_velocity = F.interpolate(
            current_velocity,
            size=moving.shape[2:],
            mode="trilinear",
            align_corners=True,
        ) * float(previous_factor)
        flow = self.integration(final_velocity)
        inverse_flow = self.integration(-final_velocity)
        return {
            "velocity_vox": final_velocity,
            "flow": flow,
            "inverse_flow": inverse_flow,
            "pyramid_factors": self.factors,
            "pyramid_velocity_vox": stage_velocities,
            "pyramid_residual_velocity_vox": stage_residuals,
            "pyramid_flow": stage_flows,
            "pyramid_warped": stage_warped,
        }


__all__ = [
    "GaussianGuidedResidualPyramid",
    "ResidualConvBlock",
    "ResidualVelocityStage",
]
