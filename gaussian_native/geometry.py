"""Geometry primitives shared by Gaussian decomposition, matching, and motion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class GaussianLevel:
    """One level of a hierarchical Gaussian anatomy representation."""

    centers_mm: torch.Tensor
    covariance_mm2: torch.Tensor
    precision_mm2: torch.Tensor
    scales_mm: torch.Tensor
    rotations: torch.Tensor
    mass: torch.Tensor
    features: torch.Tensor
    appearance: torch.Tensor
    parent_index: Optional[torch.Tensor]
    anchor_centers_mm: Optional[torch.Tensor] = None
    anchor_scales_mm: Optional[torch.Tensor] = None
    reconstruction: Optional[torch.Tensor] = None
    coverage: Optional[torch.Tensor] = None

    def with_features(self, features: torch.Tensor) -> "GaussianLevel":
        return replace(self, features=features)


def rotation_6d_to_matrix(value: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Continuous 6D rotation representation with orthonormal matrix output."""
    if value.shape[-1] != 6:
        raise AssertionError("rotation representation must end in six values")
    first = F.normalize(value[..., 0:3], dim=-1, eps=eps)
    second_raw = value[..., 3:6]
    second = F.normalize(
        second_raw - (first * second_raw).sum(dim=-1, keepdim=True) * first,
        dim=-1,
        eps=eps,
    )
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def identity_rotation_6d(
    *leading_shape: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    base = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        device=device,
        dtype=dtype,
    )
    return base.view(*([1] * len(leading_shape)), 6).expand(*leading_shape, 6)


def covariance_from_scale_rotation(
    scales_mm: torch.Tensor,
    rotations: torch.Tensor,
    epsilon_mm2: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a full SPD covariance and its precision in float32."""
    if scales_mm.shape[-1] != 3 or rotations.shape[-2:] != (3, 3):
        raise AssertionError("invalid scale/rotation shapes")
    scales = scales_mm.float().clamp_min(1.0e-3)
    rotation = rotations.float()
    diagonal = torch.diag_embed(scales.square())
    covariance = rotation @ diagonal @ rotation.transpose(-1, -2)
    eye = torch.eye(3, device=covariance.device, dtype=covariance.dtype)
    covariance = covariance + float(epsilon_mm2) * eye
    precision = torch.linalg.inv(covariance)
    return covariance, precision


def normalized_lattice(shape: Sequence[int]) -> torch.Tensor:
    """Cell-centred lattice in normalized DHW coordinates."""
    if len(shape) != 3 or min(int(value) for value in shape) <= 0:
        raise ValueError("lattice shape must contain three positive values")
    axes = [
        (torch.arange(int(size), dtype=torch.float32) + 0.5) * (2.0 / float(size)) - 1.0
        for size in shape
    ]
    dd, hh, ww = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((dd, hh, ww), dim=-1).reshape(-1, 3)


def volume_extent_mm(
    spatial_shape: Sequence[int],
    spacing_dhw: torch.Tensor,
) -> torch.Tensor:
    if len(spatial_shape) != 3 or spacing_dhw.ndim != 2 or spacing_dhw.shape[-1] != 3:
        raise AssertionError("invalid spatial shape or spacing")
    sizes = spacing_dhw.new_tensor([max(int(size) - 1, 1) for size in spatial_shape])
    return spacing_dhw * sizes.view(1, 3)


def normalized_to_physical(
    normalized_dhw: torch.Tensor,
    extent_mm: torch.Tensor,
) -> torch.Tensor:
    while extent_mm.ndim < normalized_dhw.ndim:
        extent_mm = extent_mm.unsqueeze(1)
    return 0.5 * (normalized_dhw + 1.0) * extent_mm


def physical_to_normalized(
    physical_dhw_mm: torch.Tensor,
    extent_mm: torch.Tensor,
) -> torch.Tensor:
    while extent_mm.ndim < physical_dhw_mm.ndim:
        extent_mm = extent_mm.unsqueeze(1)
    return 2.0 * physical_dhw_mm / extent_mm.clamp_min(1.0e-6) - 1.0


def physical_grid(
    shape_dhw: Sequence[int],
    extent_mm: torch.Tensor,
) -> torch.Tensor:
    """Return a batch-aware flattened physical grid in DHW order."""
    if len(shape_dhw) != 3:
        raise ValueError("shape_dhw must contain three values")
    device, dtype = extent_mm.device, extent_mm.dtype
    unit_axes = [
        torch.linspace(0.0, 1.0, int(size), device=device, dtype=dtype)
        for size in shape_dhw
    ]
    dd, hh, ww = torch.meshgrid(*unit_axes, indexing="ij")
    unit = torch.stack((dd, hh, ww), dim=-1).reshape(1, -1, 3)
    return unit * extent_mm.unsqueeze(1)


def gather_nodes(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Batch-aware gather along the node dimension."""
    if values.ndim < 3 or indices.ndim != 3 or values.shape[0] != indices.shape[0]:
        raise AssertionError("values must be [B,K,...] and indices [B,Q,N]")
    batch = torch.arange(values.shape[0], device=values.device).view(-1, 1, 1)
    return values[batch, indices]


def skew_matrix(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1] != 3:
        raise AssertionError("vector must end in three values")
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def symmetric_matrix(parameters: torch.Tensor) -> torch.Tensor:
    """Map six values to a symmetric 3x3 matrix."""
    if parameters.shape[-1] != 6:
        raise AssertionError("symmetric parameters must end in six values")
    d0, d1, d2, o01, o02, o12 = parameters.unbind(dim=-1)
    return torch.stack(
        (
            d0,
            o01,
            o02,
            o01,
            d1,
            o12,
            o02,
            o12,
            d2,
        ),
        dim=-1,
    ).reshape(*parameters.shape[:-1], 3, 3)


__all__ = [
    "GaussianLevel",
    "covariance_from_scale_rotation",
    "gather_nodes",
    "identity_rotation_6d",
    "normalized_lattice",
    "normalized_to_physical",
    "physical_grid",
    "physical_to_normalized",
    "rotation_6d_to_matrix",
    "skew_matrix",
    "symmetric_matrix",
    "volume_extent_mm",
]
