"""Geometry-Conditioned Dense Registration Module (GCDR)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )


class GeometryConditionedDenseRegistrationBlock(nn.Module):
    """Fuse local dense correspondence with a Gaussian geometry prior.

    The zero-initialized heads make the block start close to the original
    SACB-Net dense flow while preserving direct gradients to the Gaussian flow.
    """

    def __init__(
        self,
        feat_ch: int,
        context_ch: int = 11,
        hidden_ch: int = 64,
        max_residual: float = 1.0,
    ) -> None:
        super().__init__()
        if min(feat_ch, context_ch, hidden_ch) <= 0:
            raise ValueError("channel counts must be positive")
        if max_residual <= 0.0:
            raise ValueError("max_residual must be positive")
        self.feat_ch = int(feat_ch)
        self.context_ch = int(context_ch)
        self.hidden_ch = int(hidden_ch)
        self.max_residual = float(max_residual)

        self.dense_embed = _ConvBlock(2 * feat_ch + 3, hidden_ch)
        self.gaussian_embed = _ConvBlock(3 + context_ch, hidden_ch)
        gate_in = 2 * hidden_ch + 3 + 1
        residual_in = 2 * feat_ch + 2 * hidden_ch + 6 + context_ch
        head_ch = max(hidden_ch // 2, 8)
        self.gate_head = nn.Sequential(
            nn.Conv3d(gate_in, head_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(head_ch, 1, kernel_size=1),
        )
        self.residual_head = nn.Sequential(
            nn.Conv3d(residual_in, hidden_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(hidden_ch, 3, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -4.0)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        moving_feat: torch.Tensor,
        fixed_feat: torch.Tensor,
        dense_flow: torch.Tensor,
        gaussian_flow: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if moving_feat.shape != fixed_feat.shape:
            raise AssertionError("moving and fixed feature shapes must match")
        spatial_shape = moving_feat.shape[2:]
        if dense_flow.shape != (moving_feat.shape[0], 3) + spatial_shape:
            raise AssertionError("dense_flow shape mismatch")
        if gaussian_flow.shape != dense_flow.shape:
            raise AssertionError("gaussian_flow shape mismatch")
        if context.shape != (moving_feat.shape[0], self.context_ch) + spatial_shape:
            raise AssertionError("context shape mismatch")

        dense_embedding = self.dense_embed(torch.cat((fixed_feat, moving_feat, dense_flow), dim=1))
        gaussian_embedding = self.gaussian_embed(torch.cat((gaussian_flow, context), dim=1))
        disagreement = torch.abs(dense_flow - gaussian_flow)
        gate_features = torch.cat(
            (dense_embedding, gaussian_embedding, disagreement, context[:, 3:4]),
            dim=1,
        )
        gate = torch.sigmoid(self.gate_head(gate_features))
        base_flow = gate * gaussian_flow + (1.0 - gate) * dense_flow
        residual_features = torch.cat(
            (
                fixed_feat,
                moving_feat,
                dense_embedding,
                gaussian_embedding,
                dense_flow,
                gaussian_flow,
                context,
            ),
            dim=1,
        )
        residual = F.softsign(self.residual_head(residual_features)) * self.max_residual
        return base_flow + residual, gate


# Backward-compatible implementation name used by the original draft spec.
GaussianDenseFusionBlock = GeometryConditionedDenseRegistrationBlock


__all__ = [
    "GeometryConditionedDenseRegistrationBlock",
    "GaussianDenseFusionBlock",
]
