"""True coarse-to-fine residual refinement for Gaussian registration."""

from __future__ import annotations

from typing import Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn

from .integration import ScalingAndSquaring, warp_tensor


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def _gradient_magnitude(volume: torch.Tensor) -> torch.Tensor:
    """Differentiable forward-difference magnitude in DHW order."""
    derivative_d = F.pad(
        volume[:, :, 1:, :, :] - volume[:, :, :-1, :, :],
        (0, 0, 0, 0, 0, 1),
    )
    derivative_h = F.pad(
        volume[:, :, :, 1:, :] - volume[:, :, :, :-1, :],
        (0, 0, 0, 1, 0, 0),
    )
    derivative_w = F.pad(
        volume[:, :, :, :, 1:] - volume[:, :, :, :, :-1],
        (0, 1, 0, 0, 0, 0),
    )
    return torch.sqrt(
        derivative_d.square()
        + derivative_h.square()
        + derivative_w.square()
        + 1.0e-12
    )


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
        use_gradient_features: bool = False,
    ) -> None:
        super().__init__()
        if channels <= 0 or blocks <= 0 or maximum_residual_vox <= 0.0:
            raise ValueError("invalid residual velocity stage configuration")
        self.maximum_residual_vox = float(maximum_residual_vox)
        self.use_gradient_features = bool(use_gradient_features)
        input_channels = 8 if self.use_gradient_features else 6
        self.stem = nn.Conv3d(
            input_channels,
            channels,
            kernel_size=3,
            padding=1,
        )
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
        input_tensors = [
            fixed.float(),
            warped,
            fixed.float() - warped,
            current_flow.float(),
        ]
        if self.use_gradient_features:
            input_tensors.extend(
                (
                    _gradient_magnitude(fixed.float()),
                    _gradient_magnitude(warped),
                )
            )
        inputs = torch.cat(input_tensors, dim=1)
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
        blocks_per_stage: Union[int, Sequence[int]] = 3,
        maximum_residual_vox: Sequence[float] = (1.5, 1.0, 0.75),
        integration_steps: int = 7,
        use_gradient_features: bool = False,
    ) -> None:
        super().__init__()
        self.factors = tuple(int(value) for value in factors)
        channels = tuple(int(value) for value in channels)
        maximum_residual_vox = tuple(
            float(value) for value in maximum_residual_vox
        )
        if isinstance(blocks_per_stage, int):
            stage_blocks = (int(blocks_per_stage),) * len(self.factors)
        else:
            stage_blocks = tuple(
                int(value) for value in blocks_per_stage
            )
        if (
            not self.factors
            or len(self.factors) < 3
            or len(channels) != len(self.factors)
            or len(stage_blocks) != len(self.factors)
            or len(maximum_residual_vox) != len(self.factors)
            or any(value <= 0 for value in self.factors)
            or any(value <= 0 for value in stage_blocks)
            or any(
                coarse <= fine
                for coarse, fine in zip(
                    self.factors,
                    self.factors[1:],
                )
            )
        ):
            raise ValueError(
                "residual pyramid requires at least three strictly decreasing factors"
            )
        self.stages = nn.ModuleList(
            [
                ResidualVelocityStage(
                    stage_channels,
                    blocks,
                    stage_limit,
                    use_gradient_features=use_gradient_features,
                )
                for stage_channels, blocks, stage_limit in zip(
                    channels,
                    stage_blocks,
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
        if len(gaussian_level_velocity_mm) != 3:
            raise AssertionError(
                "the residual pyramid requires three Gaussian velocity fields"
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
        for stage_index, (factor, stage) in enumerate(zip(
            self.factors,
            self.stages,
        )):
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
            if current_velocity is None:
                current_velocity = moving_scale.new_zeros(
                    batch,
                    3,
                    *size,
                    dtype=torch.float32,
                )
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
                )
            if stage_index < len(gaussian_level_velocity_mm):
                gaussian_velocity = F.interpolate(
                    gaussian_level_velocity_mm[stage_index].float(),
                    size=size,
                    mode="trilinear",
                    align_corners=True,
                )
                current_velocity = (
                    current_velocity
                    + gaussian_velocity / spacing / float(factor)
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
