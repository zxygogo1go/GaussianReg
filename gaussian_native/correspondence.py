"""Coarse-to-fine partial Gaussian correspondence."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .geometry import GaussianLevel


class PartialSinkhornMatcher(nn.Module):
    """Mass-aware entropic transport with an explicit unmatched dustbin."""

    def __init__(
        self,
        feature_dim: int = 96,
        temperature: float = 0.08,
        position_weight: float = 0.12,
        scale_weight: float = 0.04,
        dustbin_mass: float = 0.15,
        iterations: int = 12,
    ) -> None:
        super().__init__()
        if not 0.0 <= dustbin_mass < 0.5:
            raise ValueError("dustbin_mass must lie in [0, 0.5)")
        if temperature <= 0.0 or iterations <= 0:
            raise ValueError("temperature and iterations must be positive")
        self.temperature = float(temperature)
        self.position_weight = float(position_weight)
        self.scale_weight = float(scale_weight)
        self.dustbin_mass = float(dustbin_mass)
        self.iterations = int(iterations)
        self.feature_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.dustbin_score = nn.Parameter(torch.tensor(-1.0))

    def _log_transport(
        self,
        logits: torch.Tensor,
        fixed_mass: torch.Tensor,
        moving_mass: torch.Tensor,
    ) -> torch.Tensor:
        batch, fixed_nodes, moving_nodes = logits.shape
        augmented = self.dustbin_score.to(
            device=logits.device,
            dtype=logits.dtype,
        ).expand(batch, fixed_nodes + 1, moving_nodes + 1).clone()
        augmented[:, :fixed_nodes, :moving_nodes] = logits
        augmented[:, -1, -1] = 0.0
        real_fraction = 1.0 - self.dustbin_mass
        fixed_marginal = torch.cat(
            (
                real_fraction * fixed_mass,
                fixed_mass.new_full((batch, 1), self.dustbin_mass),
            ),
            dim=1,
        )
        moving_marginal = torch.cat(
            (
                real_fraction * moving_mass,
                moving_mass.new_full((batch, 1), self.dustbin_mass),
            ),
            dim=1,
        )
        log_a = torch.log(fixed_marginal.clamp_min(1.0e-8))
        log_b = torch.log(moving_marginal.clamp_min(1.0e-8))
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        for _ in range(self.iterations):
            log_u = log_a - torch.logsumexp(augmented + log_v.unsqueeze(1), dim=2)
            log_v = log_b - torch.logsumexp(augmented + log_u.unsqueeze(2), dim=1)
        return augmented + log_u.unsqueeze(2) + log_v.unsqueeze(1)

    def forward(
        self,
        fixed: GaussianLevel,
        moving: GaussianLevel,
        extent_mm: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        fixed_feature = F.normalize(self.feature_projection(fixed.features), dim=-1)
        moving_feature = F.normalize(self.feature_projection(moving.features), dim=-1)
        feature_similarity = torch.einsum("bif,bjf->bij", fixed_feature, moving_feature)
        center_delta = (
            fixed.centers_mm.unsqueeze(2) - moving.centers_mm.unsqueeze(1)
        ) / extent_mm.unsqueeze(1).unsqueeze(1).clamp_min(1.0e-6)
        position_cost = center_delta.square().sum(dim=-1)
        log_scale_delta = torch.log(fixed.scales_mm.clamp_min(1.0e-3)).unsqueeze(2) - torch.log(
            moving.scales_mm.clamp_min(1.0e-3)
        ).unsqueeze(1)
        scale_cost = log_scale_delta.square().mean(dim=-1)
        cost = 1.0 - feature_similarity + self.position_weight * position_cost
        cost = cost + self.scale_weight * scale_cost
        logits = -cost / self.temperature
        if candidate_mask is not None:
            if candidate_mask.shape != logits.shape:
                raise AssertionError("candidate mask shape must match correspondence logits")
            logits = logits.masked_fill(~candidate_mask, -1.0e4)
        log_plan = self._log_transport(
            logits.float(),
            fixed.mass.float(),
            moving.mass.float(),
        )
        full_plan = torch.exp(log_plan)
        plan = full_plan[:, :-1, :-1]
        row_sum = plan.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
        column_sum = plan.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        row_plan = plan / row_sum
        column_plan = plan / column_sum
        matched_center = torch.einsum("bij,bjd->bid", row_plan, moving.centers_mm)
        matched_scale = torch.einsum("bij,bjd->bid", row_plan, moving.scales_mm)
        matched_feature = torch.einsum("bij,bjf->bif", row_plan, moving.features)
        matched_covariance = torch.einsum(
            "bij,bjmn->bimn",
            row_plan,
            moving.covariance_mm2,
        )
        reverse_center = torch.einsum(
            "bij,bid->bjd",
            column_plan,
            fixed.centers_mm,
        )
        cycle_center = torch.einsum("bij,bjd->bid", row_plan, reverse_center)
        cycle_error = (
            (cycle_center - fixed.centers_mm)
            / extent_mm.unsqueeze(1).clamp_min(1.0e-6)
        ).square().sum(dim=-1)
        transport_cost = (plan * cost.float()).sum() / plan.sum().clamp_min(1.0e-8)
        return {
            "plan": plan,
            "full_plan": full_plan,
            "cost": cost,
            "matched_center_mm": matched_center,
            "matched_scale_mm": matched_scale,
            "matched_covariance_mm2": matched_covariance,
            "matched_feature": matched_feature,
            "cycle_error": cycle_error.mean(),
            "transport_cost": transport_cost,
            "matched_mass_fraction": (
                plan.sum(dim=2)
                / ((1.0 - self.dustbin_mass) * fixed.mass.float()).clamp_min(1.0e-8)
            ).clamp(0.0, 1.0),
        }


class HierarchicalGaussianCorrespondence(nn.Module):
    def __init__(
        self,
        feature_dim: int = 96,
        temperature: float = 0.08,
        position_weight: float = 0.12,
        scale_weight: float = 0.04,
        dustbin_mass: float = 0.15,
        sinkhorn_iterations: int = 12,
        parent_candidates: int = 4,
        children_per_parent: int = 4,
        identity_calibration: bool = True,
    ) -> None:
        super().__init__()
        self.parent_candidates = int(parent_candidates)
        self.children_per_parent = int(children_per_parent)
        self.identity_calibration = bool(identity_calibration)
        self.matchers = nn.ModuleList(
            [
                PartialSinkhornMatcher(
                    feature_dim=feature_dim,
                    temperature=temperature,
                    position_weight=position_weight,
                    scale_weight=scale_weight,
                    dustbin_mass=dustbin_mass,
                    iterations=sinkhorn_iterations,
                )
                for _ in range(3)
            ]
        )

    def _candidate_mask(
        self,
        parent_plan: torch.Tensor,
        fixed_parent: torch.Tensor,
        moving_parent: torch.Tensor,
    ) -> torch.Tensor:
        batch, fixed_parents, moving_parents = parent_plan.shape
        count = min(self.parent_candidates, moving_parents)
        normalized = parent_plan / parent_plan.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
        top = normalized.topk(count, dim=2).indices
        allowed = torch.zeros(
            batch,
            fixed_parents,
            moving_parents,
            dtype=torch.bool,
            device=parent_plan.device,
        )
        allowed.scatter_(2, top, True)
        child_rows = allowed[:, fixed_parent, :]
        return child_rows[:, :, moving_parent]

    def _hierarchy_error(
        self,
        parent_plan: torch.Tensor,
        child_plan: torch.Tensor,
    ) -> torch.Tensor:
        children = self.children_per_parent
        batch, fixed_children, moving_children = child_plan.shape
        fixed_parents = fixed_children // children
        moving_parents = moving_children // children
        aggregated = child_plan.reshape(
            batch,
            fixed_parents,
            children,
            moving_parents,
            children,
        ).sum(dim=(2, 4))
        aggregated = aggregated / aggregated.sum(dim=(1, 2), keepdim=True).clamp_min(1.0e-8)
        parent = parent_plan / parent_plan.sum(dim=(1, 2), keepdim=True).clamp_min(1.0e-8)
        return F.smooth_l1_loss(aggregated, parent)

    def forward(
        self,
        fixed_levels: List[GaussianLevel],
        moving_levels: List[GaussianLevel],
        extent_mm: torch.Tensor,
    ) -> List[dict]:
        if len(fixed_levels) != 3 or len(moving_levels) != 3:
            raise AssertionError("correspondence expects three levels")
        results = []
        reference_parent_plan = None
        for index, (fixed, moving, matcher) in enumerate(
            zip(fixed_levels, moving_levels, self.matchers)
        ):
            reference_center = fixed.centers_mm
            if self.identity_calibration:
                reference_mask = None
                if index:
                    if fixed.parent_index is None or reference_parent_plan is None:
                        raise AssertionError("self-calibrated child matching requires parents")
                    reference_mask = self._candidate_mask(
                        reference_parent_plan,
                        fixed.parent_index,
                        fixed.parent_index,
                    )
                # The fixed-to-fixed transport is a non-learned calibration
                # reference. Subtracting its barycentre removes entropic
                # transport bias and guarantees zero direct displacement for
                # identical inputs without introducing a confidence gate.
                with torch.no_grad():
                    reference = matcher(
                        fixed,
                        fixed,
                        extent_mm,
                        candidate_mask=reference_mask,
                    )
                reference_center = reference["matched_center_mm"]
                reference_parent_plan = reference["plan"]
            mask = None
            if index:
                if fixed.parent_index is None or moving.parent_index is None:
                    raise AssertionError("child correspondence requires parent indices")
                mask = self._candidate_mask(
                    results[-1]["plan"].detach(),
                    fixed.parent_index,
                    moving.parent_index,
                )
            result = matcher(fixed, moving, extent_mm, candidate_mask=mask)
            result["candidate_mask"] = mask
            result["identity_reference_center_mm"] = reference_center
            result["transport_delta_mm"] = (
                result["matched_center_mm"] - reference_center
            )
            result["hierarchy_error"] = (
                result["transport_cost"].new_zeros(())
                if index == 0
                else self._hierarchy_error(results[-1]["plan"], result["plan"])
            )
            results.append(result)
        return results


__all__ = [
    "HierarchicalGaussianCorrespondence",
    "PartialSinkhornMatcher",
]
