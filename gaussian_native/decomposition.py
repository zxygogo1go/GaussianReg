"""Image-conditioned hierarchical anisotropic Gaussian decomposition."""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .geometry import (
    GaussianLevel,
    covariance_from_scale_rotation,
    identity_rotation_6d,
    normalized_lattice,
    normalized_to_physical,
    physical_grid,
    physical_to_normalized,
    rotation_6d_to_matrix,
    volume_extent_mm,
)


def _odd_gaussian_kernel(sigma: float, truncate: float = 2.5) -> torch.Tensor:
    radius = max(1, int(math.ceil(float(sigma) * float(truncate))))
    coordinate = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (coordinate / float(sigma)).square())
    return kernel / kernel.sum()


class FixedGaussianBlur3d(nn.Module):
    """Separable fixed Gaussian convolution."""

    def __init__(self, sigma: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("kernel", _odd_gaussian_kernel(sigma), persistent=False)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        if volume.ndim != 5:
            raise AssertionError("volume must have shape [B,C,D,H,W]")
        channels = int(volume.shape[1])
        kernel = self.kernel.to(device=volume.device, dtype=volume.dtype)
        radius = kernel.numel() // 2
        kd = kernel.view(1, 1, -1, 1, 1).expand(channels, 1, -1, 1, 1)
        kh = kernel.view(1, 1, 1, -1, 1).expand(channels, 1, 1, -1, 1)
        kw = kernel.view(1, 1, 1, 1, -1).expand(channels, 1, 1, 1, -1)
        value = F.conv3d(volume, kd, padding=(radius, 0, 0), groups=channels)
        value = F.conv3d(value, kh, padding=(0, radius, 0), groups=channels)
        return F.conv3d(value, kw, padding=(0, 0, radius), groups=channels)


class GaussianScaleSpace(nn.Module):
    """Fixed Gaussian pyramid and first/second derivative measurements."""

    def __init__(self, factors: Sequence[int] = (16, 8, 4), sigma: float = 1.0) -> None:
        super().__init__()
        factors = tuple(int(value) for value in factors)
        if not factors or any(value < 2 or value & (value - 1) for value in factors):
            raise ValueError("pyramid factors must be powers of two greater than one")
        self.factors = factors
        self.blur = FixedGaussianBlur3d(sigma=sigma)

    @staticmethod
    def _central_derivative(volume: torch.Tensor, axis: int) -> torch.Tensor:
        kernel = volume.new_tensor([-0.5, 0.0, 0.5])
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = 3
        weight = kernel.view(*shape)
        padding = [0, 0, 0]
        padding[axis] = 1
        return F.conv3d(volume, weight, padding=tuple(padding))

    @staticmethod
    def _second_derivative(volume: torch.Tensor, axis: int) -> torch.Tensor:
        kernel = volume.new_tensor([1.0, -2.0, 1.0])
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = 3
        weight = kernel.view(*shape)
        padding = [0, 0, 0]
        padding[axis] = 1
        return F.conv3d(volume, weight, padding=tuple(padding))

    def _measurements(
        self,
        volume: torch.Tensor,
        effective_spacing: torch.Tensor,
    ) -> torch.Tensor:
        derivatives = []
        laplacian = torch.zeros_like(volume)
        for axis in range(3):
            spacing = effective_spacing[:, axis].view(-1, 1, 1, 1, 1)
            derivatives.append(self._central_derivative(volume, axis) / spacing)
            laplacian = laplacian + self._second_derivative(volume, axis) / spacing.square()
        gradient_magnitude = torch.sqrt(
            sum(derivative.square() for derivative in derivatives) + 1.0e-6
        )
        local_mean = self.blur(volume)
        local_variance = (self.blur(volume.square()) - local_mean.square()).clamp_min(0.0)
        return torch.cat(
            (
                volume,
                derivatives[0],
                derivatives[1],
                derivatives[2],
                gradient_magnitude,
                laplacian,
                local_variance,
            ),
            dim=1,
        )

    def forward(
        self,
        volume: torch.Tensor,
        spacing_dhw: torch.Tensor,
    ) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        if volume.ndim != 5 or volume.shape[1] != 1:
            raise AssertionError("Gaussian scale-space expects [B,1,D,H,W]")
        requested = set(self.factors)
        maximum_steps = int(math.log2(max(self.factors)))
        current = volume.float()
        images = {}
        for step in range(1, maximum_steps + 1):
            current = self.blur(current)
            current = F.avg_pool3d(current, kernel_size=2, stride=2)
            factor = 2 ** step
            if factor in requested:
                images[factor] = current
        ordered_images = [images[factor] for factor in self.factors]
        measurements = [
            self._measurements(image, spacing_dhw.float() * float(factor))
            for factor, image in zip(self.factors, ordered_images)
        ]
        return ordered_images, measurements


class LocalGaussianSampler(nn.Module):
    """Sample local scale-space measurements in a Gaussian coordinate frame."""

    def __init__(self, samples_per_axis: int = 3, radius: float = 1.5) -> None:
        super().__init__()
        if samples_per_axis < 2:
            raise ValueError("samples_per_axis must be at least two")
        axis = torch.linspace(-float(radius), float(radius), int(samples_per_axis))
        dd, hh, ww = torch.meshgrid(axis, axis, axis, indexing="ij")
        offsets = torch.stack((dd, hh, ww), dim=-1).reshape(-1, 3)
        weights = torch.exp(-0.5 * offsets.square().sum(dim=-1))
        self.register_buffer("offsets", offsets, persistent=False)
        self.register_buffer("weights", weights / weights.sum(), persistent=False)

    @property
    def descriptor_multiplier(self) -> int:
        return 2

    def forward(
        self,
        measurements: torch.Tensor,
        centers_mm: torch.Tensor,
        scales_mm: torch.Tensor,
        rotations: torch.Tensor,
        extent_mm: torch.Tensor,
    ) -> torch.Tensor:
        batch, nodes = centers_mm.shape[:2]
        offsets = self.offsets.to(device=centers_mm.device, dtype=centers_mm.dtype)
        local = offsets.view(1, 1, -1, 3) * scales_mm.unsqueeze(2)
        world = torch.einsum("bkij,bkpj->bkpi", rotations, local)
        points_mm = centers_mm.unsqueeze(2) + world
        normalized = physical_to_normalized(points_mm, extent_mm)
        grid = normalized[..., [2, 1, 0]].reshape(batch, nodes, offsets.shape[0], 1, 3)
        sampled = F.grid_sample(
            measurements,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.squeeze(-1).permute(0, 2, 3, 1)
        weights = self.weights.to(device=sampled.device, dtype=sampled.dtype).view(1, 1, -1, 1)
        mean = (weights * sampled).sum(dim=2)
        variance = (weights * (sampled - mean.unsqueeze(2)).square()).sum(dim=2)
        return torch.cat((mean, torch.sqrt(variance + 1.0e-6)), dim=-1)


class GaussianScalarRasterizer(nn.Module):
    """Chunked normalized Gaussian interpolation for decomposition supervision."""

    def __init__(self, node_chunk: int = 64, cutoff_sigma: float = 3.5) -> None:
        super().__init__()
        self.node_chunk = int(node_chunk)
        self.cutoff_squared = float(cutoff_sigma) ** 2

    def forward(
        self,
        level: GaussianLevel,
        output_shape: Sequence[int],
        extent_mm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = physical_grid(output_shape, extent_mm.float())
        batch, voxels = grid.shape[:2]
        numerator = grid.new_zeros(batch, voxels)
        denominator = grid.new_zeros(batch, voxels)
        nodes = int(level.centers_mm.shape[1])
        amplitude = level.appearance[..., 0].float()
        scaled_mass = level.mass.float() * float(nodes)
        for start in range(0, nodes, self.node_chunk):
            stop = min(start + self.node_chunk, nodes)
            delta = grid.unsqueeze(1) - level.centers_mm[:, start:stop].float().unsqueeze(2)
            mahalanobis = torch.einsum(
                "bcvi,bcij,bcvj->bcv",
                delta,
                level.precision_mm2[:, start:stop].float(),
                delta,
            )
            weights = torch.exp(-0.5 * mahalanobis)
            weights = torch.where(
                mahalanobis <= self.cutoff_squared,
                weights,
                torch.zeros_like(weights),
            )
            weights = weights * scaled_mass[:, start:stop].unsqueeze(-1)
            denominator = denominator + weights.sum(dim=1)
            numerator = numerator + (
                weights * amplitude[:, start:stop].unsqueeze(-1)
            ).sum(dim=1)
        reconstruction = (numerator / denominator.clamp_min(1.0e-6)).reshape(
            batch, 1, *output_shape
        )
        coverage = (1.0 - torch.exp(-denominator)).reshape(batch, 1, *output_shape)
        return reconstruction, coverage


class _GeometryPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.geometry = nn.Linear(hidden_dim, 13)
        nn.init.zeros_(self.geometry.weight)
        nn.init.zeros_(self.geometry.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.geometry(self.body(value))


class HierarchicalGaussianDecomposer(nn.Module):
    """Shared moving/fixed volume-to-Gaussian decomposition."""

    measurement_channels = 7

    def __init__(
        self,
        root_grid_shape: Sequence[int] = (4, 4, 4),
        children_per_parent: int = 4,
        feature_dim: int = 96,
        hidden_dim: int = 128,
        pyramid_factors: Sequence[int] = (16, 8, 4),
        samples_per_axis: int = 3,
        raster_chunk: int = 64,
        covariance_mode: str = "full",
    ) -> None:
        super().__init__()
        if int(children_per_parent) != 4:
            raise ValueError("the current anatomy-adaptive split uses exactly four children")
        if len(pyramid_factors) != 3:
            raise ValueError("the Gaussian hierarchy requires three pyramid factors")
        self.root_grid_shape = tuple(int(value) for value in root_grid_shape)
        self.children_per_parent = int(children_per_parent)
        self.feature_dim = int(feature_dim)
        self.covariance_mode = str(covariance_mode).strip().lower()
        if self.covariance_mode not in {"diagonal", "full"}:
            raise ValueError("covariance_mode must be diagonal or full")
        self.pyramid = GaussianScaleSpace(factors=pyramid_factors)
        self.sampler = LocalGaussianSampler(samples_per_axis=samples_per_axis)
        descriptor_dim = (
            self.measurement_channels * self.sampler.descriptor_multiplier
        )
        self.root_geometry = _GeometryPredictor(descriptor_dim, hidden_dim)
        self.child_geometry = nn.ModuleList(
            [
                _GeometryPredictor(descriptor_dim + feature_dim, hidden_dim),
                _GeometryPredictor(descriptor_dim + feature_dim, hidden_dim),
            ]
        )
        feature_input_root = descriptor_dim + 6
        feature_input_child = descriptor_dim + feature_dim + 6
        self.root_feature = nn.Sequential(
            nn.Linear(feature_input_root, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.child_feature = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_input_child, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, feature_dim),
                )
                for _ in range(2)
            ]
        )
        self.rasterizer = GaussianScalarRasterizer(node_chunk=raster_chunk)
        self.register_buffer(
            "root_lattice",
            normalized_lattice(self.root_grid_shape),
            persistent=False,
        )
        tetrahedron = torch.tensor(
            [
                [-0.55, -0.55, -0.55],
                [-0.55, 0.55, 0.55],
                [0.55, -0.55, 0.55],
                [0.55, 0.55, -0.55],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("child_offsets", tetrahedron, persistent=False)

    def _root_level(
        self,
        measurements: torch.Tensor,
        extent_mm: torch.Tensor,
        spacing_dhw: torch.Tensor,
    ) -> GaussianLevel:
        batch = measurements.shape[0]
        lattice = self.root_lattice.to(device=measurements.device, dtype=torch.float32)
        lattice = lattice.unsqueeze(0).expand(batch, -1, -1)
        anchor_centers = normalized_to_physical(lattice, extent_mm)
        cell = extent_mm / extent_mm.new_tensor(self.root_grid_shape).view(1, 3)
        anchor_scales = 0.55 * cell.unsqueeze(1).expand(
            -1,
            anchor_centers.shape[1],
            -1,
        )
        centers = anchor_centers
        scales = anchor_scales
        rotations = torch.eye(3, device=centers.device, dtype=centers.dtype).view(
            1, 1, 3, 3
        ).expand(batch, centers.shape[1], -1, -1)
        initial = self.sampler(measurements, centers, scales, rotations, extent_mm)
        parameters = self.root_geometry(initial)
        local_offset = torch.tanh(parameters[..., 0:3]) * scales * 0.35
        centers = centers + local_offset
        scales = scales * torch.exp(0.40 * torch.tanh(parameters[..., 3:6]))
        minimum = spacing_dhw.unsqueeze(1) * 1.25
        scales = torch.maximum(scales, minimum)
        rotation_6d = identity_rotation_6d(
            batch,
            centers.shape[1],
            device=centers.device,
            dtype=centers.dtype,
        ) + 0.25 * parameters[..., 6:12]
        rotations = rotation_6d_to_matrix(rotation_6d)
        if self.covariance_mode == "diagonal":
            rotations = torch.eye(
                3,
                device=centers.device,
                dtype=centers.dtype,
            ).view(1, 1, 3, 3).expand(batch, centers.shape[1], -1, -1)
        mass = torch.softmax(parameters[..., 12], dim=1)
        refined = self.sampler(measurements, centers, scales, rotations, extent_mm)
        normalized_center = physical_to_normalized(centers, extent_mm)
        normalized_scale = scales / extent_mm.unsqueeze(1).clamp_min(1.0e-6)
        features = self.root_feature(
            torch.cat((refined, normalized_center, normalized_scale), dim=-1)
        )
        covariance, precision = covariance_from_scale_rotation(scales, rotations)
        return GaussianLevel(
            centers_mm=centers,
            covariance_mm2=covariance,
            precision_mm2=precision,
            scales_mm=scales,
            rotations=rotations,
            mass=mass,
            features=features,
            appearance=refined,
            parent_index=None,
            anchor_centers_mm=anchor_centers,
            anchor_scales_mm=anchor_scales,
        )

    def _child_level(
        self,
        parent: GaussianLevel,
        measurements: torch.Tensor,
        extent_mm: torch.Tensor,
        spacing_dhw: torch.Tensor,
        child_stage: int,
    ) -> GaussianLevel:
        batch, parent_nodes = parent.centers_mm.shape[:2]
        children = self.children_per_parent
        parent_index = torch.arange(parent_nodes, device=parent.centers_mm.device).repeat_interleave(
            children
        )
        parent_center = parent.centers_mm[:, parent_index]
        parent_scale = parent.scales_mm[:, parent_index]
        parent_rotation = parent.rotations[:, parent_index]
        parent_feature = parent.features[:, parent_index]
        if parent.anchor_centers_mm is None or parent.anchor_scales_mm is None:
            raise AssertionError("canonical parent anchors are required")
        anchor_parent_center = parent.anchor_centers_mm[:, parent_index]
        anchor_parent_scale = parent.anchor_scales_mm[:, parent_index]
        offsets = self.child_offsets.to(
            device=parent_center.device,
            dtype=parent_center.dtype,
        ).repeat(parent_nodes, 1)
        anchor_offset = offsets.view(1, parent_nodes * children, 3)
        anchor_offset = anchor_offset * anchor_parent_scale * 0.72
        anchor_centers = anchor_parent_center + anchor_offset
        anchor_scales = anchor_parent_scale * 0.56
        local_offset = offsets.view(1, parent_nodes * children, 3) * parent_scale * 0.72
        world_offset = torch.einsum(
            "bkij,bkj->bki",
            parent_rotation,
            local_offset,
        )
        centers = parent_center + world_offset
        scales = parent_scale * 0.56
        rotations = parent_rotation
        initial = self.sampler(measurements, centers, scales, rotations, extent_mm)
        parameters = self.child_geometry[child_stage](
            torch.cat((initial, parent_feature), dim=-1)
        )
        residual_local = torch.tanh(parameters[..., 0:3]) * parent_scale * 0.22
        centers = centers + torch.einsum(
            "bkij,bkj->bki",
            parent_rotation,
            residual_local,
        )
        scales = scales * torch.exp(0.35 * torch.tanh(parameters[..., 3:6]))
        scales = torch.minimum(scales, parent_scale * 0.82)
        scales = torch.maximum(scales, spacing_dhw.unsqueeze(1) * 1.10)
        delta_rotation = rotation_6d_to_matrix(
            identity_rotation_6d(
                batch,
                parent_nodes * children,
                device=centers.device,
                dtype=centers.dtype,
            )
            + 0.20 * parameters[..., 6:12]
        )
        rotations = parent_rotation @ delta_rotation
        if self.covariance_mode == "diagonal":
            rotations = torch.eye(
                3,
                device=centers.device,
                dtype=centers.dtype,
            ).view(1, 1, 3, 3).expand(batch, parent_nodes * children, -1, -1)
        child_logits = parameters[..., 12].reshape(batch, parent_nodes, children)
        child_fraction = torch.softmax(child_logits, dim=-1)
        mass = (
            parent.mass.unsqueeze(-1) * child_fraction
        ).reshape(batch, parent_nodes * children)
        refined = self.sampler(measurements, centers, scales, rotations, extent_mm)
        normalized_center = physical_to_normalized(centers, extent_mm)
        normalized_scale = scales / extent_mm.unsqueeze(1).clamp_min(1.0e-6)
        features = self.child_feature[child_stage](
            torch.cat(
                (
                    refined,
                    parent_feature,
                    normalized_center,
                    normalized_scale,
                ),
                dim=-1,
            )
        )
        covariance, precision = covariance_from_scale_rotation(scales, rotations)
        return GaussianLevel(
            centers_mm=centers,
            covariance_mm2=covariance,
            precision_mm2=precision,
            scales_mm=scales,
            rotations=rotations,
            mass=mass,
            features=features,
            appearance=refined,
            parent_index=parent_index,
            anchor_centers_mm=anchor_centers,
            anchor_scales_mm=anchor_scales,
        )

    def forward(
        self,
        volume: torch.Tensor,
        spacing_dhw: torch.Tensor,
        compute_reconstruction: bool = True,
    ) -> dict:
        if spacing_dhw.ndim == 1:
            spacing_dhw = spacing_dhw.unsqueeze(0).expand(volume.shape[0], -1)
        spacing = spacing_dhw.to(device=volume.device, dtype=torch.float32)
        extent = volume_extent_mm(volume.shape[2:], spacing)
        pyramid_images, measurements = self.pyramid(volume, spacing)
        levels = [self._root_level(measurements[0], extent, spacing)]
        levels.append(
            self._child_level(levels[-1], measurements[1], extent, spacing, child_stage=0)
        )
        levels.append(
            self._child_level(levels[-1], measurements[2], extent, spacing, child_stage=1)
        )
        if compute_reconstruction:
            reconstructed = []
            for level, target in zip(levels, pyramid_images):
                reconstruction, coverage = self.rasterizer(
                    level,
                    target.shape[2:],
                    extent,
                )
                level.reconstruction = reconstruction
                level.coverage = coverage
                reconstructed.append(reconstruction)
        return {
            "levels": levels,
            "pyramid_images": pyramid_images,
            "extent_mm": extent,
        }


__all__ = [
    "FixedGaussianBlur3d",
    "GaussianScaleSpace",
    "GaussianScalarRasterizer",
    "HierarchicalGaussianDecomposer",
    "LocalGaussianSampler",
]
