"""Compact Gaussian anatomy correspondence for 3-D registration.

The module deliberately models only what is required by the registration
network: spatial Gaussian tokens, feature correspondence, and a rasterized
correspondence residual.  It does not predict visibility, confidence,
anatomical classes, or a standalone Gaussian displacement field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _factor_grid(num_tokens: int, spatial_size: Sequence[int]) -> Tuple[int, int, int]:
    """Find an exact token-grid factorization close to the feature aspect ratio."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if len(spatial_size) != 3 or min(int(value) for value in spatial_size) <= 0:
        raise ValueError("spatial_size must contain three positive values")
    target = torch.tensor([float(value) for value in spatial_size])
    target = target / target.prod().pow(1.0 / 3.0)
    best: Optional[Tuple[int, int, int]] = None
    best_score = float("inf")
    for depth in range(1, num_tokens + 1):
        if num_tokens % depth:
            continue
        remaining = num_tokens // depth
        for height in range(1, remaining + 1):
            if remaining % height:
                continue
            width = remaining // height
            grid = torch.tensor([float(depth), float(height), float(width)])
            grid = grid / grid.prod().pow(1.0 / 3.0)
            score = float((grid.log() - target.log()).square().sum())
            if score < best_score:
                best_score = score
                best = (depth, height, width)
    if best is None:
        raise RuntimeError("failed to factor token count")
    return best


def _normalized_grid(
    spatial_size: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    axes = [
        torch.linspace(-1.0, 1.0, int(size), device=device, dtype=dtype)
        for size in spatial_size
    ]
    depth, height, width = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((depth, height, width), dim=-1).reshape(-1, 3)


def _initial_anchor_logits(num_tokens: int, spatial_size: Sequence[int]) -> torch.Tensor:
    depth_count, height_count, width_count = _factor_grid(num_tokens, spatial_size)

    def axis(count: int) -> torch.Tensor:
        if count == 1:
            return torch.zeros(1)
        return torch.linspace(-0.90, 0.90, count)

    depth, height, width = torch.meshgrid(
        axis(depth_count),
        axis(height_count),
        axis(width_count),
        indexing="ij",
    )
    anchors = torch.stack((depth, height, width), dim=-1).reshape(num_tokens, 3)
    return torch.atanh(anchors.clamp(-0.999, 0.999))


@dataclass
class GaussianTokenSet:
    """Compact diagonal-Gaussian token representation."""

    mu: torch.Tensor
    variance: torch.Tensor
    feat: torch.Tensor
    attention: Optional[torch.Tensor] = None

    def validate(self) -> None:
        if self.mu.ndim != 3 or self.mu.shape[-1] != 3:
            raise AssertionError("mu must have shape [B,N,3]")
        if self.variance.shape != self.mu.shape:
            raise AssertionError("variance must have shape [B,N,3]")
        if self.feat.shape[:2] != self.mu.shape[:2]:
            raise AssertionError("feat must share batch and token dimensions")
        if self.attention is not None and self.attention.shape[:2] != self.mu.shape[:2]:
            raise AssertionError("attention must have shape [B,N,V]")


class CompactGaussianTokenizer3D(nn.Module):
    """Shared attention tokenizer with diagonal spatial moments."""

    def __init__(
        self,
        in_ch: int,
        spatial_size: Sequence[int],
        token_dim: int = 32,
        num_tokens: int = 96,
        attention_temperature: float = 0.20,
        min_variance: float = 1.0e-3,
        max_variance: float = 0.50,
    ) -> None:
        super().__init__()
        if min(in_ch, token_dim, num_tokens) <= 0:
            raise ValueError("channel and token counts must be positive")
        if attention_temperature <= 0.0:
            raise ValueError("attention_temperature must be positive")
        if not 0.0 < min_variance < max_variance:
            raise ValueError("variance bounds must satisfy 0 < min < max")
        self.spatial_size = tuple(int(value) for value in spatial_size)
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.attention_temperature = float(attention_temperature)
        self.min_variance = float(min_variance)
        self.max_variance = float(max_variance)

        self.key_proj = nn.Conv3d(in_ch, token_dim, kernel_size=1)
        self.value_proj = nn.Conv3d(in_ch, token_dim, kernel_size=1)
        self.queries = nn.Parameter(torch.empty(num_tokens, token_dim))
        self.anchor_logits = nn.Parameter(
            _initial_anchor_logits(num_tokens, self.spatial_size)
        )
        self.log_radius = nn.Parameter(torch.full((num_tokens, 3), -1.05))
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(
        self,
        feature: torch.Tensor,
        return_attention: bool = False,
    ) -> GaussianTokenSet:
        if feature.ndim != 5:
            raise AssertionError("feature must have shape [B,C,D,H,W]")
        batch, _, depth, height, width = feature.shape
        keys = self.key_proj(feature).flatten(2).transpose(1, 2)
        values = self.value_proj(feature).flatten(2).transpose(1, 2)
        coords = _normalized_grid(
            (depth, height, width),
            feature.device,
            feature.dtype,
        )

        query = F.normalize(
            self.queries.to(dtype=feature.dtype),
            dim=-1,
            eps=1.0e-6,
        )
        key = F.normalize(keys, dim=-1, eps=1.0e-6)
        content_logits = torch.einsum("ne,bve->bnv", query, key)
        content_logits = content_logits / math.sqrt(float(self.token_dim))

        anchors = torch.tanh(self.anchor_logits).to(dtype=feature.dtype)
        radius = (F.softplus(self.log_radius) + 0.05).to(dtype=feature.dtype)
        difference = coords[None, None, :, :] - anchors[None, :, None, :]
        spatial_logits = -0.5 * (
            difference.square() / radius[None, :, None, :].square()
        ).sum(dim=-1)
        attention = torch.softmax(
            (content_logits + spatial_logits) / self.attention_temperature,
            dim=-1,
        )

        mu = torch.einsum("bnv,vc->bnc", attention, coords)
        token_feature = torch.einsum("bnv,bve->bne", attention, values)
        centered = coords[None, None, :, :] - mu[:, :, None, :]
        variance = torch.einsum(
            "bnv,bnvc->bnc",
            attention.float(),
            centered.float().square(),
        )
        variance = variance.clamp(
            min=self.min_variance,
            max=self.max_variance,
        ).to(dtype=feature.dtype)
        tokens = GaussianTokenSet(
            mu=mu,
            variance=variance,
            feat=token_feature,
            attention=attention if return_attention else None,
        )
        tokens.validate()
        if tokens.mu.shape[0] != batch:
            raise AssertionError("token batch size changed unexpectedly")
        return tokens


class GaussianAnatomyCorrespondenceModule(nn.Module):
    """Rasterize matched token-feature residuals into a spatial context map."""

    def __init__(
        self,
        in_ch: int,
        spatial_size: Sequence[int],
        num_tokens: int = 96,
        token_dim: int = 32,
        context_ch: int = 8,
        match_temperature: float = 0.10,
        position_cost_weight: float = 0.05,
        raster_voxel_chunk: int = 4096,
    ) -> None:
        super().__init__()
        if context_ch <= 0 or match_temperature <= 0.0:
            raise ValueError("context_ch and match_temperature must be positive")
        if position_cost_weight < 0.0 or raster_voxel_chunk <= 0:
            raise ValueError("position weight must be nonnegative and chunk size positive")
        self.context_ch = int(context_ch)
        self.match_temperature = float(match_temperature)
        self.position_cost_weight = float(position_cost_weight)
        self.raster_voxel_chunk = int(raster_voxel_chunk)
        self.tokenizer = CompactGaussianTokenizer3D(
            in_ch=in_ch,
            spatial_size=spatial_size,
            token_dim=token_dim,
            num_tokens=num_tokens,
        )
        self.residual_norm = nn.LayerNorm(token_dim)
        self.residual_proj = nn.Linear(token_dim, context_ch, bias=False)

    def _rasterize(
        self,
        fixed: GaussianTokenSet,
        token_context: torch.Tensor,
        spatial_size: Sequence[int],
    ) -> torch.Tensor:
        batch = fixed.mu.shape[0]
        depth, height, width = (int(value) for value in spatial_size)
        coords = _normalized_grid(
            (depth, height, width),
            fixed.mu.device,
            fixed.mu.dtype,
        )
        context_flat = token_context.new_empty(
            (batch, self.context_ch, coords.shape[0])
        )
        for start in range(0, coords.shape[0], self.raster_voxel_chunk):
            end = min(start + self.raster_voxel_chunk, coords.shape[0])
            difference = (
                coords[None, start:end, None, :]
                - fixed.mu[:, None, :, :]
            )
            mahalanobis = (
                difference.square()
                / fixed.variance[:, None, :, :].clamp_min(1.0e-5)
            ).sum(dim=-1)
            interpolation = torch.softmax(-0.5 * mahalanobis, dim=-1)
            context_chunk = torch.einsum(
                "bvn,bnc->bvc",
                interpolation,
                token_context,
            )
            context_flat[:, :, start:end] = context_chunk.transpose(1, 2)
        return context_flat.reshape(
            batch,
            self.context_ch,
            depth,
            height,
            width,
        )

    def forward(
        self,
        moving_feat: torch.Tensor,
        fixed_feat: torch.Tensor,
        return_aux: bool = False,
    ) -> Dict[str, object]:
        if moving_feat.shape != fixed_feat.shape:
            raise AssertionError("moving and fixed features must have identical shapes")
        moving = self.tokenizer(moving_feat, return_attention=return_aux)
        fixed = self.tokenizer(fixed_feat, return_attention=return_aux)

        fixed_feature = F.normalize(fixed.feat, dim=-1, eps=1.0e-6)
        moving_feature = F.normalize(moving.feat, dim=-1, eps=1.0e-6)
        feature_cost = 1.0 - torch.einsum(
            "bne,bme->bnm",
            fixed_feature,
            moving_feature,
        )
        position_cost = (
            fixed.mu[:, :, None, :] - moving.mu[:, None, :, :]
        ).square().sum(dim=-1)
        match_cost = feature_cost + self.position_cost_weight * position_cost
        correspondence = torch.softmax(
            -match_cost / self.match_temperature,
            dim=-1,
        )
        matched_moving_feature = torch.einsum(
            "bnm,bme->bne",
            correspondence,
            moving.feat,
        )
        token_residual = fixed.feat - matched_moving_feature
        token_context = torch.tanh(
            self.residual_proj(self.residual_norm(token_residual))
        )
        context = self._rasterize(
            fixed,
            token_context,
            fixed_feat.shape[2:],
        )
        return {
            "context": context,
            "moving_tokens": moving,
            "fixed_tokens": fixed,
            "correspondence": correspondence,
            "match_cost": match_cost,
            "feature_cost": feature_cost,
            "position_cost": position_cost,
            "token_residual": token_residual,
        }


__all__ = [
    "CompactGaussianTokenizer3D",
    "GaussianAnatomyCorrespondenceModule",
    "GaussianTokenSet",
]
