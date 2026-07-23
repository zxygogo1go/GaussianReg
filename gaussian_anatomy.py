"""Gaussian Anatomy Correspondence Module (GACM).

The module extracts a shared set of anisotropic Gaussian anatomy tokens from
moving and fixed feature volumes, matches them with visibility-aware
unbalanced optimal transport, and rasterizes sparse token correspondences into
a dense displacement prior and geometry context.

Coordinates stored by tokens use normalized ``(D, H, W)`` order. Dense flow
uses voxel displacement in the same order, matching ``utils.SpatialTransformer``.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _fp32_context(tensor: torch.Tensor):
    if tensor.is_cuda:
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", enabled=False)
        return torch.cuda.amp.autocast(enabled=False)
    return contextlib.nullcontext()


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(float(value)))


def _factor_grid(num_tokens: int, spatial_size: Sequence[int]) -> Tuple[int, int, int]:
    """Find an exact factorization whose aspect ratio follows the feature grid."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if len(spatial_size) != 3 or min(int(v) for v in spatial_size) <= 0:
        raise ValueError("spatial_size must contain three positive values")
    target = torch.tensor([float(v) for v in spatial_size])
    target = target / target.prod().pow(1.0 / 3.0)
    best = None
    best_score = float("inf")
    for d in range(1, num_tokens + 1):
        if num_tokens % d:
            continue
        remaining = num_tokens // d
        for h in range(1, remaining + 1):
            if remaining % h:
                continue
            w = remaining // h
            grid = torch.tensor([float(d), float(h), float(w)])
            grid = grid / grid.prod().pow(1.0 / 3.0)
            score = float((grid.log() - target.log()).square().sum())
            if score < best_score:
                best_score = score
                best = (d, h, w)
    if best is None:
        raise RuntimeError("failed to factor token count")
    return best


def _make_normalized_grid(
    spatial_size: Sequence[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    axes = [torch.linspace(-1.0, 1.0, int(size), device=device, dtype=dtype) for size in spatial_size]
    dd, hh, ww = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((dd, hh, ww), dim=-1).reshape(-1, 3)


def _make_anchor_grid(num_tokens: int, spatial_size: Sequence[int]) -> torch.Tensor:
    gd, gh, gw = _factor_grid(num_tokens, spatial_size)

    def axis(count: int) -> torch.Tensor:
        if count == 1:
            return torch.zeros(1)
        return torch.linspace(-0.95, 0.95, count)

    dd, hh, ww = torch.meshgrid(axis(gd), axis(gh), axis(gw), indexing="ij")
    constrained = torch.stack((dd, hh, ww), dim=-1).reshape(num_tokens, 3)
    return torch.atanh(constrained.clamp(-0.999, 0.999))


def _sqrtm_spd(matrix: torch.Tensor, eps: float = 1.0e-6, iterations: int = 8) -> torch.Tensor:
    """Differentiable SPD square root using a scaled Newton-Schulz iteration.

    Backpropagating through eigenvectors is undefined at repeated eigenvalues,
    which are common for near-isotropic anatomy tokens. Newton-Schulz avoids
    that failure while retaining a fully differentiable float32 square root.
    """
    matrix_f = 0.5 * (matrix.float() + matrix.float().transpose(-1, -2))
    eye = torch.eye(3, device=matrix.device, dtype=torch.float32)
    eye = eye.expand(matrix_f.shape)
    matrix_f = matrix_f + float(eps) * eye
    norm = torch.linalg.matrix_norm(matrix_f, ord="fro", dim=(-2, -1), keepdim=True)
    norm = norm.clamp_min(float(eps))
    y = matrix_f / norm
    z = eye.clone()
    for _ in range(int(iterations)):
        update = 0.5 * (3.0 * eye - z @ y)
        y = y @ update
        z = update @ z
    return y * norm.sqrt()


@dataclass
class GaussianTokenSet:
    mu: torch.Tensor
    cov: torch.Tensor
    feat: torch.Tensor
    anatomy: torch.Tensor
    visibility: torch.Tensor
    attention: Optional[torch.Tensor] = None

    def validate(self) -> None:
        if self.mu.ndim != 3 or self.mu.shape[-1] != 3:
            raise AssertionError("mu must have shape [B,N,3]")
        if self.cov.shape != self.mu.shape[:2] + (3, 3):
            raise AssertionError("cov must have shape [B,N,3,3]")
        if self.feat.shape[:2] != self.mu.shape[:2]:
            raise AssertionError("feat must share batch/token dimensions")
        if self.anatomy.shape[:2] != self.mu.shape[:2]:
            raise AssertionError("anatomy must share batch/token dimensions")
        if self.visibility.shape != self.mu.shape[:2] + (1,):
            raise AssertionError("visibility must have shape [B,N,1]")
        if self.attention is not None and self.attention.shape[:2] != self.mu.shape[:2]:
            raise AssertionError("attention must have shape [B,N,V]")


class AnisotropicGaussianTokenizer3D(nn.Module):
    """Shared attention-moment tokenizer for 3D feature volumes."""

    def __init__(
        self,
        in_ch: int,
        spatial_size: Sequence[int],
        token_dim: int = 64,
        num_tokens: int = 128,
        num_types: int = 8,
        temperature: float = 0.10,
        sigma_min: float = 0.015,
        sigma_max: float = 0.35,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or num_tokens <= 0 or num_types <= 0:
            raise ValueError("token_dim, num_tokens and num_types must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError("sigma bounds must satisfy 0 < sigma_min < sigma_max")
        self.spatial_size = tuple(int(v) for v in spatial_size)
        self.token_dim = int(token_dim)
        self.num_tokens = int(num_tokens)
        self.num_types = int(num_types)
        self.temperature = float(temperature)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

        self.key_proj = nn.Conv3d(in_ch, token_dim, kernel_size=1)
        self.value_proj = nn.Conv3d(in_ch, token_dim, kernel_size=1)
        self.queries = nn.Parameter(torch.empty(num_tokens, token_dim))
        self.anchors = nn.Parameter(_make_anchor_grid(num_tokens, self.spatial_size))
        radius_raw = _inverse_softplus(0.35 - 0.05)
        self.log_radius = nn.Parameter(torch.full((num_tokens, 3), radius_raw))
        self.type_head = nn.Linear(token_dim, num_types)
        self.vis_head = nn.Linear(token_dim + 1, 1)
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, feature: torch.Tensor, return_attention: bool = False) -> GaussianTokenSet:
        if feature.ndim != 5:
            raise AssertionError("feature must have shape [B,C,D,H,W]")
        b, _, d, h, w = feature.shape
        keys = self.key_proj(feature).permute(0, 2, 3, 4, 1).reshape(b, -1, self.token_dim)
        values = self.value_proj(feature).permute(0, 2, 3, 4, 1).reshape(b, -1, self.token_dim)
        coords = _make_normalized_grid((d, h, w), feature.device, feature.dtype)

        query = F.normalize(self.queries.to(dtype=feature.dtype), dim=-1, eps=1.0e-6)
        key = F.normalize(keys, dim=-1, eps=1.0e-6)
        content_logits = torch.einsum("ne,bve->bnv", query, key) / math.sqrt(self.token_dim)
        centers = torch.tanh(self.anchors).to(dtype=feature.dtype)
        radius = (F.softplus(self.log_radius) + 0.05).to(dtype=feature.dtype)
        diff = coords[None, None] - centers[None, :, None]
        spatial_bias = -0.5 * (diff.square() / radius[None, :, None].square()).sum(dim=-1)
        attention = torch.softmax((content_logits + spatial_bias) / self.temperature, dim=-1)

        mu = torch.einsum("bnv,vc->bnc", attention, coords)
        token_feat = torch.einsum("bnv,bve->bne", attention, values)

        with _fp32_context(feature):
            attention_f = attention.float()
            coords_f = coords.float()
            mu_f = mu.float()
            second = torch.einsum(
                "bnv,vij->bnij",
                attention_f,
                torch.einsum("vi,vj->vij", coords_f, coords_f),
            )
            cov = second - torch.einsum("bni,bnj->bnij", mu_f, mu_f)
            eye = torch.eye(3, device=feature.device, dtype=torch.float32).view(1, 1, 3, 3)
            cov = 0.5 * (cov + cov.transpose(-1, -2)) + 1.0e-5 * eye
            # Spectral bounds are computed without eigenvector gradients. Near-
            # isotropic covariances have repeated eigenvalues, for which the
            # derivative of an eigendecomposition is not finite.
            with torch.no_grad():
                eigvals = torch.linalg.eigvalsh(cov)
                shift = (self.sigma_min ** 2 - eigvals[..., 0]).clamp_min(0.0)
                maximum = eigvals[..., -1] + shift
                scale = (self.sigma_max ** 2 / maximum.clamp_min(1.0e-8)).clamp_max(1.0)
            cov = (cov + shift[..., None, None] * eye) * scale[..., None, None]

        anatomy = torch.softmax(self.type_head(token_feat), dim=-1)
        entropy = -(attention.float() * attention.float().clamp_min(1.0e-8).log()).sum(dim=-1)
        entropy = entropy / max(math.log(float(attention.shape[-1])), 1.0)
        visibility_input = torch.cat((token_feat, entropy.to(token_feat.dtype).unsqueeze(-1)), dim=-1)
        visibility = 0.2 + 0.8 * torch.sigmoid(self.vis_head(visibility_input))
        tokens = GaussianTokenSet(
            mu=mu,
            cov=cov.to(feature.dtype),
            feat=token_feat,
            anatomy=anatomy,
            visibility=visibility,
            attention=attention if return_attention else None,
        )
        tokens.validate()
        return tokens


def pairwise_bures_wasserstein(
    mu_fixed: torch.Tensor,
    cov_fixed: torch.Tensor,
    mu_moving: torch.Tensor,
    cov_moving: torch.Tensor,
    chunk_size: int = 32,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return center, covariance-only Bures, and full Gaussian W2 costs."""
    if mu_fixed.ndim != 3 or mu_moving.ndim != 3:
        raise AssertionError("mu tensors must have shape [B,N,3]")
    if mu_fixed.shape[0] != mu_moving.shape[0] or mu_fixed.shape[-1] != 3 or mu_moving.shape[-1] != 3:
        raise AssertionError("mu tensors have incompatible shapes")
    if cov_fixed.shape != mu_fixed.shape[:2] + (3, 3):
        raise AssertionError("fixed covariance shape mismatch")
    if cov_moving.shape != mu_moving.shape[:2] + (3, 3):
        raise AssertionError("moving covariance shape mismatch")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    out_dtype = mu_fixed.dtype
    center_chunks = []
    covariance_chunks = []
    with _fp32_context(mu_fixed):
        mf = mu_fixed.float()
        mm = mu_moving.float()
        sf = cov_fixed.float()
        sm = cov_moving.float()
        trace_m = sm.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[:, None, :]
        for start in range(0, mf.shape[1], int(chunk_size)):
            end = min(start + int(chunk_size), mf.shape[1])
            mf_chunk = mf[:, start:end]
            sf_chunk = sf[:, start:end]
            center = (mf_chunk[:, :, None] - mm[:, None]).square().sum(dim=-1)
            sqrt_f = _sqrtm_spd(sf_chunk, eps=eps)
            inner = sqrt_f[:, :, None] @ sm[:, None] @ sqrt_f[:, :, None]
            sqrt_inner = _sqrtm_spd(inner, eps=eps)
            trace_f = sf_chunk.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[:, :, None]
            trace_inner = sqrt_inner.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            covariance = (trace_f + trace_m - 2.0 * trace_inner).clamp_min(0.0)
            center_chunks.append(center)
            covariance_chunks.append(covariance)
        center_cost = torch.cat(center_chunks, dim=1)
        covariance_cost = torch.cat(covariance_chunks, dim=1)
        full_cost = (center_cost + covariance_cost).clamp_min(0.0)
    return center_cost.to(out_dtype), covariance_cost.to(out_dtype), full_cost.to(out_dtype)


class UnbalancedSinkhorn(nn.Module):
    """Log-domain entropy-regularized unbalanced optimal transport."""

    def __init__(self, epsilon: float = 0.05, rho: float = 0.5, iterations: int = 25) -> None:
        super().__init__()
        if epsilon <= 0.0 or rho <= 0.0 or iterations <= 0:
            raise ValueError("epsilon, rho and iterations must be positive")
        self.epsilon = float(epsilon)
        self.rho = float(rho)
        self.iterations = int(iterations)

    def forward(self, cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if cost.ndim != 3 or a.shape != cost.shape[:2] or b.shape != (cost.shape[0], cost.shape[2]):
            raise AssertionError("cost/a/b shapes are incompatible")
        out_dtype = cost.dtype
        with _fp32_context(cost):
            cost_f = cost.float()
            log_a = a.float().clamp_min(1.0e-8).log()
            log_b = b.float().clamp_min(1.0e-8).log()
            tau = self.rho / (self.rho + self.epsilon)
            log_k = (-cost_f / self.epsilon).clamp_min(-1.0e4)
            u = torch.zeros_like(log_a)
            v = torch.zeros_like(log_b)
            for _ in range(self.iterations):
                u = tau * (log_a - torch.logsumexp(log_k + v[:, None, :], dim=2))
                v = tau * (log_b - torch.logsumexp(log_k + u[:, :, None], dim=1))
            log_plan = log_k + u[:, :, None] + v[:, None, :]
            plan = torch.exp(log_plan.clamp(min=-80.0, max=30.0))
        return plan.to(out_dtype)


class GaussianAnatomyMatcher3D(nn.Module):
    """Shared-tokenizer GACM with UOT matching and dense rasterization."""

    def __init__(
        self,
        in_ch: int,
        spatial_size: Sequence[int],
        num_tokens: int,
        token_dim: int = 64,
        num_types: int = 8,
        cost_feat: float = 1.0,
        cost_pos: float = 0.15,
        cost_cov: float = 0.25,
        cost_anatomy: float = 0.10,
        cost_visibility: float = 0.05,
        sinkhorn_epsilon: float = 0.05,
        sinkhorn_rho: float = 0.5,
        sinkhorn_iterations: int = 25,
        bures_chunk: int = 32,
        raster_voxel_chunk: int = 32768,
    ) -> None:
        super().__init__()
        self.spatial_size = tuple(int(v) for v in spatial_size)
        self.tokenizer = AnisotropicGaussianTokenizer3D(
            in_ch=in_ch,
            spatial_size=self.spatial_size,
            token_dim=token_dim,
            num_tokens=num_tokens,
            num_types=num_types,
        )
        self.sinkhorn = UnbalancedSinkhorn(
            epsilon=sinkhorn_epsilon,
            rho=sinkhorn_rho,
            iterations=sinkhorn_iterations,
        )
        self.cost_feat = float(cost_feat)
        self.cost_pos = float(cost_pos)
        self.cost_cov = float(cost_cov)
        self.cost_anatomy = float(cost_anatomy)
        self.cost_visibility = float(cost_visibility)
        self.bures_chunk = int(bures_chunk)
        self.raster_voxel_chunk = int(raster_voxel_chunk)

    def _rasterize(
        self,
        fixed: GaussianTokenSet,
        anchor_disp_voxel: torch.Tensor,
        anchor_conf: torch.Tensor,
        spatial_size: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d, h, w = (int(v) for v in spatial_size)
        b, n, _ = fixed.mu.shape
        out_dtype = fixed.mu.dtype
        coords = _make_normalized_grid((d, h, w), fixed.mu.device, torch.float32)
        flow_flat = torch.zeros((b, 3, coords.shape[0]), device=fixed.mu.device, dtype=torch.float32)
        conf_flat = torch.zeros((b, 1, coords.shape[0]), device=fixed.mu.device, dtype=torch.float32)
        cov_flat = torch.zeros((b, 6, coords.shape[0]), device=fixed.mu.device, dtype=torch.float32)
        aniso_flat = torch.zeros((b, 1, coords.shape[0]), device=fixed.mu.device, dtype=torch.float32)

        with _fp32_context(fixed.mu):
            mu = fixed.mu.float()
            cov = fixed.cov.float()
            cov_inv = torch.linalg.inv(cov)
            trace = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            cov_normalized = cov / trace.unsqueeze(-1)
            with torch.no_grad():
                eigvals = torch.linalg.eigvalsh(cov)
                anisotropy = (eigvals[..., -1] - eigvals[..., 0]) / eigvals[..., -1].clamp_min(1.0e-6)
            disp = anchor_disp_voxel.float()
            token_conf = anchor_conf.float().squeeze(-1).clamp(0.0, 1.0)
            for start in range(0, coords.shape[0], self.raster_voxel_chunk):
                end = min(start + self.raster_voxel_chunk, coords.shape[0])
                diff = coords[None, start:end, None, :] - mu[:, None, :, :]
                mahal = torch.einsum("bvni,bnij,bvnj->bvn", diff, cov_inv, diff).clamp(0.0, 60.0)
                weights = torch.exp(-0.5 * mahal) * token_conf[:, None, :]
                den = weights.sum(dim=-1, keepdim=True)
                alpha = weights / (den + 1.0e-8)
                flow_chunk = torch.einsum("bvn,bnc->bvc", alpha, disp)
                cov_chunk = torch.einsum("bvn,bnij->bvij", alpha, cov_normalized)
                aniso_chunk = torch.einsum("bvn,bn->bv", alpha, anisotropy)
                flow_flat[:, :, start:end] = flow_chunk.transpose(1, 2)
                conf_flat[:, :, start:end] = (den / (1.0 + den)).transpose(1, 2)
                components = torch.stack(
                    (
                        cov_chunk[..., 0, 0],
                        cov_chunk[..., 1, 1],
                        cov_chunk[..., 2, 2],
                        cov_chunk[..., 0, 1],
                        cov_chunk[..., 0, 2],
                        cov_chunk[..., 1, 2],
                    ),
                    dim=-1,
                )
                cov_flat[:, :, start:end] = components.transpose(1, 2)
                aniso_flat[:, :, start:end] = aniso_chunk[:, None, :]

        flow = flow_flat.reshape(b, 3, d, h, w).to(out_dtype)
        confidence = conf_flat.reshape(b, 1, d, h, w).to(out_dtype)
        scale = flow.new_tensor(
            [2.0 / max(d - 1, 1), 2.0 / max(h - 1, 1), 2.0 / max(w - 1, 1)]
        ).view(1, 3, 1, 1, 1)
        flow_normalized = flow * scale
        context = torch.cat(
            (
                flow_normalized,
                confidence,
                cov_flat.reshape(b, 6, d, h, w).to(out_dtype),
                aniso_flat.reshape(b, 1, d, h, w).to(out_dtype),
            ),
            dim=1,
        )
        return flow, confidence, context

    def forward(
        self,
        moving_feat: torch.Tensor,
        fixed_feat: torch.Tensor,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if moving_feat.shape != fixed_feat.shape:
            raise AssertionError("moving and fixed features must have identical shapes")
        spatial_size = moving_feat.shape[2:]
        moving = self.tokenizer(moving_feat, return_attention=return_aux)
        fixed = self.tokenizer(fixed_feat, return_attention=return_aux)

        fixed_feat_n = F.normalize(fixed.feat, dim=-1, eps=1.0e-6)
        moving_feat_n = F.normalize(moving.feat, dim=-1, eps=1.0e-6)
        c_feat = 1.0 - torch.einsum("bfe,bme->bfm", fixed_feat_n, moving_feat_n)
        c_pos, c_cov, _ = pairwise_bures_wasserstein(
            fixed.mu,
            fixed.cov,
            moving.mu,
            moving.cov,
            chunk_size=self.bures_chunk,
        )
        c_anatomy = 1.0 - torch.einsum("bfk,bmk->bfm", fixed.anatomy, moving.anatomy)
        c_visibility = -torch.log(
            fixed.visibility * moving.visibility.transpose(1, 2) + 1.0e-6
        )
        cost = (
            self.cost_feat * c_feat
            + self.cost_pos * c_pos
            + self.cost_cov * c_cov
            + self.cost_anatomy * c_anatomy
            + self.cost_visibility * c_visibility
        )

        mass_fixed = fixed.visibility.squeeze(-1)
        mass_moving = moving.visibility.squeeze(-1)
        mass_fixed = mass_fixed / mass_fixed.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        mass_moving = mass_moving / mass_moving.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        transport = self.sinkhorn(cost, mass_fixed, mass_moving)
        row_mass = transport.sum(dim=-1)
        row_probability = transport / (row_mass.unsqueeze(-1) + 1.0e-8)
        matched_moving_mu = torch.einsum("bfm,bmc->bfc", row_probability, moving.mu)
        anchor_disp_normalized = matched_moving_mu - fixed.mu
        voxel_scale = fixed.mu.new_tensor(
            [max(spatial_size[0] - 1, 1) / 2.0, max(spatial_size[1] - 1, 1) / 2.0, max(spatial_size[2] - 1, 1) / 2.0]
        )
        anchor_disp_voxel = anchor_disp_normalized * voxel_scale.view(1, 1, 3)
        entropy = -(
            row_probability * row_probability.clamp_min(1.0e-8).log()
        ).sum(dim=-1) / max(math.log(float(row_probability.shape[-1])), 1.0)
        relative_mass = (row_mass / (mass_fixed + 1.0e-8)).clamp(0.0, 1.0)
        anchor_conf = (
            relative_mass * (1.0 - entropy).clamp(0.0, 1.0) * fixed.visibility.squeeze(-1)
        ).unsqueeze(-1)
        flow, confidence, context = self._rasterize(
            fixed,
            anchor_disp_voxel,
            anchor_conf,
            spatial_size,
        )
        return {
            "flow": flow,
            "confidence": confidence,
            "context": context,
            "moving_tokens": moving,
            "fixed_tokens": fixed,
            "transport": transport,
            "cost": cost,
            "cost_feature": c_feat,
            "cost_position": c_pos,
            "cost_covariance": c_cov,
            "cost_anatomy": c_anatomy,
            "anchor_disp": anchor_disp_voxel,
            "anchor_conf": anchor_conf,
        }


GaussianAnatomyCorrespondenceModule = GaussianAnatomyMatcher3D


__all__ = [
    "GaussianTokenSet",
    "AnisotropicGaussianTokenizer3D",
    "pairwise_bures_wasserstein",
    "UnbalancedSinkhorn",
    "GaussianAnatomyMatcher3D",
    "GaussianAnatomyCorrespondenceModule",
]
