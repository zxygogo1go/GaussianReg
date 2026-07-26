"""Hierarchical Gaussian graph feature encoding."""

from __future__ import annotations

import math
from typing import List

import torch
from torch import nn

from .geometry import GaussianLevel, gather_nodes


class GaussianGraphBlock(nn.Module):
    """Local self-attention over geometry-defined Gaussian neighbours."""

    def __init__(
        self,
        feature_dim: int,
        heads: int = 4,
        neighbors: int = 16,
        expansion: int = 3,
    ) -> None:
        super().__init__()
        if feature_dim % heads:
            raise ValueError("feature_dim must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = feature_dim // heads
        self.neighbors = int(neighbors)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.qkv = nn.Linear(feature_dim, feature_dim * 3)
        self.edge_bias = nn.Sequential(
            nn.Linear(7, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, heads),
        )
        self.projection = nn.Linear(feature_dim, feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * expansion),
            nn.GELU(),
            nn.Linear(feature_dim * expansion, feature_dim),
        )

    def forward(self, level: GaussianLevel, features: torch.Tensor) -> torch.Tensor:
        batch, nodes, channels = features.shape
        neighbor_count = min(self.neighbors + 1, nodes)
        global_scale = level.scales_mm.mean(dim=(1, 2), keepdim=True).clamp_min(1.0e-3)
        normalized_center = level.centers_mm / global_scale
        distances = torch.cdist(normalized_center.float(), normalized_center.float())
        indices = distances.topk(neighbor_count, dim=-1, largest=False).indices
        if neighbor_count > 1:
            indices = indices[..., 1:]
        normalized = self.norm1(features)
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        q = q.reshape(batch, nodes, self.heads, self.head_dim)
        k = k.reshape(batch, nodes, self.heads, self.head_dim)
        v = v.reshape(batch, nodes, self.heads, self.head_dim)
        neighbor_k = gather_nodes(k, indices)
        neighbor_v = gather_nodes(v, indices)
        query = q.unsqueeze(2)
        logits = (query * neighbor_k).sum(dim=-1) / math.sqrt(float(self.head_dim))

        neighbor_centers = gather_nodes(level.centers_mm, indices)
        delta = neighbor_centers - level.centers_mm.unsqueeze(2)
        local_delta = torch.einsum(
            "bkji,bkni->bknj",
            level.rotations,
            delta,
        ) / level.scales_mm.unsqueeze(2).clamp_min(1.0e-3)
        neighbor_scales = gather_nodes(level.scales_mm, indices)
        log_scale_ratio = torch.log(
            neighbor_scales.clamp_min(1.0e-3)
            / level.scales_mm.unsqueeze(2).clamp_min(1.0e-3)
        )
        local_distance = torch.linalg.vector_norm(local_delta, dim=-1, keepdim=True)
        edge = torch.cat((local_delta, log_scale_ratio, local_distance), dim=-1)
        logits = logits + self.edge_bias(edge)
        attention = torch.softmax(logits.float(), dim=2).to(neighbor_v.dtype)
        update = (attention.unsqueeze(-1) * neighbor_v).sum(dim=2)
        update = update.reshape(batch, nodes, channels)
        features = features + self.projection(update)
        return features + self.ffn(self.norm2(features))


class HierarchicalGaussianEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 96,
        heads: int = 4,
        neighbors: int = 16,
        blocks_per_level: int = 2,
    ) -> None:
        super().__init__()
        self.level_embeddings = nn.Parameter(torch.zeros(3, feature_dim))
        nn.init.normal_(self.level_embeddings, std=0.02)
        self.geometry_projection = nn.Sequential(
            nn.Linear(7, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.parent_projection = nn.ModuleList(
            [nn.Linear(feature_dim, feature_dim) for _ in range(2)]
        )
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        GaussianGraphBlock(
                            feature_dim=feature_dim,
                            heads=heads,
                            neighbors=neighbors,
                        )
                        for _ in range(blocks_per_level)
                    ]
                )
                for _ in range(3)
            ]
        )
        self.output_norms = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in range(3)])

    def forward(
        self,
        levels: List[GaussianLevel],
        extent_mm: torch.Tensor,
    ) -> List[GaussianLevel]:
        if len(levels) != 3:
            raise AssertionError("the encoder expects three Gaussian levels")
        encoded = []
        for level_index, level in enumerate(levels):
            center = 2.0 * level.centers_mm / extent_mm.unsqueeze(1).clamp_min(1.0e-6) - 1.0
            scale = level.scales_mm / extent_mm.unsqueeze(1).clamp_min(1.0e-6)
            mass = torch.log(level.mass.clamp_min(1.0e-8)).unsqueeze(-1)
            geometry = torch.cat((center, scale, mass), dim=-1)
            features = level.features + self.geometry_projection(geometry)
            features = features + self.level_embeddings[level_index].view(1, 1, -1)
            if level_index:
                if level.parent_index is None:
                    raise AssertionError("child level must provide parent indices")
                parent_features = encoded[-1].features[:, level.parent_index]
                features = features + self.parent_projection[level_index - 1](parent_features)
            for block in self.blocks[level_index]:
                features = block(level, features)
            encoded.append(level.with_features(self.output_norms[level_index](features)))
        return encoded


__all__ = ["GaussianGraphBlock", "HierarchicalGaussianEncoder"]
