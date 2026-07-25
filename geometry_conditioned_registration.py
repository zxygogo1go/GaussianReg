"""Lightweight geometry-conditioned residual correction."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class GeometryConditionedResidualCorrector(nn.Module):
    """Apply a bounded context-conditioned correction to a dense flow.

    There is intentionally no Gaussian/dense gate and no parallel flow
    embedding.  The original SACB dense correspondence is always the base
    prediction; Gaussian correspondence contributes only through a small
    residual head.
    """

    def __init__(
        self,
        feat_ch: int,
        context_ch: int = 8,
        hidden_ch: int = 32,
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
        input_ch = 2 * self.feat_ch + 3 + self.context_ch
        self.residual_head = nn.Sequential(
            nn.Conv3d(input_ch, self.hidden_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(self.hidden_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(self.hidden_ch, 3, kernel_size=3, padding=1),
        )
        # A tiny nonzero initialization keeps the first prediction effectively
        # identical to the SACB dense flow while allowing registration loss to
        # reach the Gaussian context from the first optimizer step.
        nn.init.normal_(self.residual_head[-1].weight, mean=0.0, std=1.0e-5)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        moving_feat: torch.Tensor,
        fixed_feat: torch.Tensor,
        dense_flow: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if moving_feat.shape != fixed_feat.shape:
            raise AssertionError("moving and fixed feature shapes must match")
        spatial_shape = moving_feat.shape[2:]
        expected_flow = (moving_feat.shape[0], 3) + spatial_shape
        if dense_flow.shape != expected_flow:
            raise AssertionError("dense_flow shape mismatch")
        expected_context = (
            moving_feat.shape[0],
            self.context_ch,
        ) + spatial_shape
        if context.shape != expected_context:
            raise AssertionError("context shape mismatch")
        features = torch.cat(
            (fixed_feat, moving_feat, dense_flow, context),
            dim=1,
        )
        residual = (
            torch.tanh(self.residual_head(features))
            * self.max_residual
        )
        return dense_flow + residual, residual


__all__ = ["GeometryConditionedResidualCorrector"]
