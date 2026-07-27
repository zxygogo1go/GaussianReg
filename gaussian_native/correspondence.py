"""Coarse-to-fine partial Gaussian correspondence."""

from __future__ import annotations

from contextlib import nullcontext
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
        coordinate_mode: str = "learned",
        mutual_transport: bool = False,
        detach_geometry_cost: bool = False,
        appearance_weight: float = 0.0,
        transport_mode: str = "sinkhorn",
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
        self.coordinate_mode = str(coordinate_mode).strip().lower()
        if self.coordinate_mode not in {"learned", "canonical"}:
            raise ValueError("coordinate_mode must be learned or canonical")
        self.mutual_transport = bool(mutual_transport)
        self.detach_geometry_cost = bool(detach_geometry_cost)
        self.appearance_weight = float(appearance_weight)
        if not 0.0 <= self.appearance_weight <= 1.0:
            raise ValueError("appearance_weight must lie in [0, 1]")
        self.transport_mode = str(transport_mode).strip().lower()
        if self.transport_mode not in {"sinkhorn", "row_softmax"}:
            raise ValueError("transport_mode must be sinkhorn or row_softmax")
        if self.transport_mode == "row_softmax" and self.dustbin_mass:
            raise ValueError("row_softmax requires dustbin_mass=0")
        self.feature_projection = nn.Linear(feature_dim, feature_dim, bias=False)
        self.dustbin_score = nn.Parameter(torch.tensor(-1.0))

    def set_temperature(self, temperature: float) -> None:
        """Update the entropic matching temperature without changing weights."""
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def _coordinates(self, level: GaussianLevel) -> torch.Tensor:
        if self.coordinate_mode == "learned":
            return level.centers_mm
        if level.anchor_centers_mm is None:
            raise AssertionError("canonical correspondence requires anchor centers")
        return level.anchor_centers_mm

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

    @staticmethod
    def _standardized_appearance(level: GaussianLevel) -> torch.Tensor:
        """Return a fixed, per-volume normalized Gaussian appearance descriptor."""
        appearance = level.appearance.detach().float()
        mean = appearance.mean(dim=1, keepdim=True)
        scale = appearance.std(dim=1, keepdim=True, unbiased=False).clamp_min(
            1.0e-4
        )
        standardized = ((appearance - mean) / scale).clamp(-5.0, 5.0)
        return F.normalize(standardized, dim=-1)

    def _transport(
        self,
        logits: torch.Tensor,
        fixed_mass: torch.Tensor,
        moving_mass: torch.Tensor,
        candidate_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.transport_mode == "sinkhorn":
            return torch.exp(
                self._log_transport(
                    logits.float(),
                    fixed_mass.float(),
                    moving_mass.float(),
                )
            )
        row_probability = torch.softmax(logits.float(), dim=2)
        if candidate_mask is not None:
            row_probability = row_probability * candidate_mask.float()
            row_probability = row_probability / row_probability.sum(
                dim=2,
                keepdim=True,
            ).clamp_min(1.0e-8)
        plan = row_probability * fixed_mass.float().unsqueeze(2)
        return F.pad(plan, (0, 1, 0, 1))

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
        if self.appearance_weight:
            fixed_appearance = self._standardized_appearance(fixed)
            moving_appearance = self._standardized_appearance(moving)
            if fixed_appearance.shape[-1] != moving_appearance.shape[-1]:
                raise AssertionError("moving/fixed appearance dimensions must match")
            appearance_similarity = torch.einsum(
                "bif,bjf->bij",
                fixed_appearance,
                moving_appearance,
            )
        else:
            appearance_similarity = torch.zeros_like(feature_similarity)
        fixed_coordinate = self._coordinates(fixed)
        moving_coordinate = self._coordinates(moving)
        fixed_cost_coordinate = (
            fixed_coordinate.detach()
            if self.detach_geometry_cost
            else fixed_coordinate
        )
        moving_cost_coordinate = (
            moving_coordinate.detach()
            if self.detach_geometry_cost
            else moving_coordinate
        )
        center_delta = (
            fixed_cost_coordinate.unsqueeze(2)
            - moving_cost_coordinate.unsqueeze(1)
        ) / extent_mm.unsqueeze(1).unsqueeze(1).clamp_min(1.0e-6)
        position_cost = center_delta.square().sum(dim=-1)
        fixed_scale = (
            fixed.scales_mm.detach()
            if self.detach_geometry_cost
            else fixed.scales_mm
        )
        moving_scale = (
            moving.scales_mm.detach()
            if self.detach_geometry_cost
            else moving.scales_mm
        )
        log_scale_delta = torch.log(fixed_scale.clamp_min(1.0e-3)).unsqueeze(
            2
        ) - torch.log(moving_scale.clamp_min(1.0e-3)).unsqueeze(1)
        scale_cost = log_scale_delta.square().mean(dim=-1)
        feature_cost = 1.0 - feature_similarity
        appearance_cost = 1.0 - appearance_similarity
        cost = (
            (1.0 - self.appearance_weight) * feature_cost
            + self.appearance_weight * appearance_cost
            + self.position_weight * position_cost
        )
        cost = cost + self.scale_weight * scale_cost
        logits = -cost / self.temperature
        if candidate_mask is not None:
            if candidate_mask.shape != logits.shape:
                raise AssertionError("candidate mask shape must match correspondence logits")
            logits = logits.masked_fill(~candidate_mask, -1.0e4)
        full_plan = self._transport(
            logits,
            fixed.mass,
            moving.mass,
            candidate_mask,
        )
        plan = full_plan[:, :-1, :-1]
        row_sum = plan.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
        column_sum = plan.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
        row_plan = plan / row_sum
        column_plan = plan / column_sum
        mutual_affinity = row_plan * column_plan
        mutual_row_plan = mutual_affinity / mutual_affinity.sum(
            dim=2,
            keepdim=True,
        ).clamp_min(1.0e-8)
        mutual_column_plan = mutual_affinity / mutual_affinity.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-8)
        selected_row_plan = mutual_row_plan if self.mutual_transport else row_plan
        selected_column_plan = (
            mutual_column_plan if self.mutual_transport else column_plan
        )
        row_entropy = -(
            selected_row_plan
            * selected_row_plan.clamp_min(1.0e-8).log()
        ).sum(dim=2)
        if candidate_mask is None:
            support_size = row_entropy.new_full(
                row_entropy.shape,
                moving.centers_mm.shape[1],
            )
        else:
            support_size = candidate_mask.sum(dim=2).to(row_entropy.dtype)
        maximum_entropy = support_size.clamp_min(2.0).log()
        support_entropy = row_entropy / maximum_entropy
        support_entropy = torch.where(
            support_size > 1.0,
            support_entropy,
            torch.zeros_like(support_entropy),
        ).clamp(0.0, 1.0)
        match_evidence = (1.0 - support_entropy).clamp(0.0, 1.0)
        matched_center = torch.einsum(
            "bij,bjd->bid",
            selected_row_plan,
            moving.centers_mm,
        )
        matched_anchor_center = torch.einsum(
            "bij,bjd->bid",
            selected_row_plan,
            moving_coordinate,
        )
        matched_scale = torch.einsum(
            "bij,bjd->bid",
            selected_row_plan,
            moving.scales_mm,
        )
        matched_feature = torch.einsum(
            "bij,bjf->bif",
            selected_row_plan,
            moving.features,
        )
        matched_covariance = torch.einsum(
            "bij,bjmn->bimn",
            selected_row_plan,
            moving.covariance_mm2,
        )
        reverse_center = torch.einsum(
            "bij,bid->bjd",
            selected_column_plan,
            fixed_coordinate,
        )
        cycle_center = torch.einsum(
            "bij,bjd->bid",
            selected_row_plan,
            reverse_center,
        )
        cycle_error = (
            (cycle_center - fixed_coordinate)
            / extent_mm.unsqueeze(1).clamp_min(1.0e-6)
        ).square().sum(dim=-1)
        transport_cost = (plan * cost.float()).sum() / plan.sum().clamp_min(1.0e-8)
        return {
            "plan": plan,
            "motion_plan": mutual_affinity if self.mutual_transport else plan,
            "full_plan": full_plan,
            "cost": cost,
            "matched_center_mm": matched_center,
            "matched_anchor_center_mm": matched_anchor_center,
            "matched_scale_mm": matched_scale,
            "matched_covariance_mm2": matched_covariance,
            "matched_feature": matched_feature,
            "cycle_error": cycle_error.mean(),
            "transport_cost": transport_cost,
            "mutual_concentration": selected_row_plan.max(dim=2).values.mean(),
            "support_entropy": support_entropy,
            "match_evidence": match_evidence,
            "support_size": support_size,
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
        calibration_gradient: bool = False,
        coordinate_mode: str = "learned",
        mutual_transport: bool = False,
        detach_geometry_cost: bool = False,
        appearance_weight: float = 0.0,
        transport_mode: str = "sinkhorn",
        shared_calibration_candidates: bool = False,
        include_identity_candidate: bool = False,
    ) -> None:
        super().__init__()
        self.parent_candidates = int(parent_candidates)
        self.children_per_parent = int(children_per_parent)
        self.identity_calibration = bool(identity_calibration)
        self.calibration_gradient = bool(calibration_gradient)
        self.coordinate_mode = str(coordinate_mode).strip().lower()
        self.shared_calibration_candidates = bool(
            shared_calibration_candidates
        )
        self.include_identity_candidate = bool(include_identity_candidate)
        self.matchers = nn.ModuleList(
            [
                PartialSinkhornMatcher(
                    feature_dim=feature_dim,
                    temperature=temperature,
                    position_weight=position_weight,
                    scale_weight=scale_weight,
                    dustbin_mass=dustbin_mass,
                    iterations=sinkhorn_iterations,
                    coordinate_mode=coordinate_mode,
                    mutual_transport=mutual_transport,
                    detach_geometry_cost=detach_geometry_cost,
                    appearance_weight=appearance_weight,
                    transport_mode=transport_mode,
                )
                for _ in range(3)
            ]
        )

    @property
    def temperature(self) -> float:
        temperatures = {matcher.temperature for matcher in self.matchers}
        if len(temperatures) != 1:
            raise RuntimeError("correspondence levels have inconsistent temperatures")
        return temperatures.pop()

    def set_temperature(self, temperature: float) -> None:
        for matcher in self.matchers:
            matcher.set_temperature(temperature)

    def _candidate_mask(
        self,
        parent_plan: torch.Tensor,
        fixed_parent: torch.Tensor,
        moving_parent: torch.Tensor,
    ) -> torch.Tensor:
        batch, fixed_parents, moving_parents = parent_plan.shape
        count = min(self.parent_candidates, moving_parents)
        normalized = parent_plan / parent_plan.sum(dim=2, keepdim=True).clamp_min(1.0e-8)
        allowed = torch.zeros(
            batch,
            fixed_parents,
            moving_parents,
            dtype=torch.bool,
            device=parent_plan.device,
        )
        add_identity = (
            self.include_identity_candidate
            and fixed_parents == moving_parents
        )
        top_count = max(count - 1, 0) if add_identity else count
        if top_count:
            top = normalized.topk(top_count, dim=2).indices
            allowed.scatter_(2, top, True)
        if add_identity:
            identity = torch.arange(
                fixed_parents,
                device=parent_plan.device,
            ).view(1, fixed_parents, 1).expand(batch, -1, -1)
            allowed.scatter_(2, identity, True)
        child_rows = allowed[:, fixed_parent, :]
        child_mask = child_rows[:, :, moving_parent]
        if not bool(child_mask.any(dim=2).all()):
            raise AssertionError("every fixed child requires a moving candidate")
        return child_mask

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
            mask = None
            if index:
                if fixed.parent_index is None or moving.parent_index is None:
                    raise AssertionError("child correspondence requires parent indices")
                mask = self._candidate_mask(
                    results[-1]["motion_plan"].detach(),
                    fixed.parent_index,
                    moving.parent_index,
                )
            reference_center = fixed.centers_mm
            reference_mask = None
            if self.identity_calibration:
                if index:
                    if fixed.parent_index is None:
                        raise AssertionError("self-calibrated child matching requires parents")
                    if self.shared_calibration_candidates:
                        if moving.parent_index is None or not torch.equal(
                            fixed.parent_index,
                            moving.parent_index,
                        ):
                            raise AssertionError(
                                "shared calibration requires aligned hierarchy indices"
                            )
                        reference_mask = mask
                    else:
                        if reference_parent_plan is None:
                            raise AssertionError(
                                "legacy self-calibration requires a parent plan"
                            )
                        reference_mask = self._candidate_mask(
                            reference_parent_plan,
                            fixed.parent_index,
                            fixed.parent_index,
                        )
                # Subtract the fixed-to-fixed transport barycentre to remove
                # entropic matching bias. V3 keeps this path differentiable so
                # identical cross/self transports cancel in both value and
                # gradient; V2 retains its historical stopped reference.
                calibration_context = (
                    nullcontext()
                    if self.calibration_gradient
                    else torch.no_grad()
                )
                with calibration_context:
                    reference = matcher(
                        fixed,
                        fixed,
                        extent_mm,
                        candidate_mask=reference_mask,
                    )
                reference_center = (
                    reference["matched_anchor_center_mm"]
                    if self.coordinate_mode == "canonical"
                    else reference["matched_center_mm"]
                )
                reference_parent_plan = reference["motion_plan"]
            result = matcher(fixed, moving, extent_mm, candidate_mask=mask)
            result["candidate_mask"] = mask
            result["identity_candidate_mask"] = reference_mask
            matched_coordinate = (
                result["matched_anchor_center_mm"]
                if self.coordinate_mode == "canonical"
                else result["matched_center_mm"]
            )
            result["identity_reference_center_mm"] = reference_center
            result["transport_delta_mm"] = (
                matched_coordinate - reference_center
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
