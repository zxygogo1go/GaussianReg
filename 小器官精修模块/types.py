"""Public data contracts for plug-in local refinement."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class LocalRefinementConfig:
    """Non-learned controller settings for the local refiner."""

    input_mode: str = "full"
    roi_radius_min: int = 3
    roi_radius_max: int = 8
    roi_smooth_steps: int = 2
    residual_scale_min: float = 0.25
    residual_scale_max: float = 1.0
    lambda_small_min: float = 1.0
    lambda_small_max: float = 3.0
    lambda_smooth_base: float = 0.10
    lambda_smooth_extra: float = 0.20
    anatomy_difficulty_weights: Tuple[float, float, float, float] = (0.45, 0.20, 0.25, 0.10)
    ct_difficulty_weights: Tuple[float, float, float] = (0.35, 0.45, 0.20)
    flow_scale: float = 20.0

    def __post_init__(self) -> None:
        from .functional import INPUT_MODES

        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"Unsupported input mode {self.input_mode!r}; choices={INPUT_MODES}")
        if self.roi_radius_min < 0 or self.roi_radius_max < self.roi_radius_min:
            raise ValueError(f"Invalid ROI radius range: {self.roi_radius_min}..{self.roi_radius_max}")
        if self.residual_scale_max < self.residual_scale_min:
            raise ValueError("residual_scale_max must be greater than or equal to residual_scale_min")
        if self.lambda_small_max < self.lambda_small_min:
            raise ValueError("lambda_small_max must be greater than or equal to lambda_small_min")
        if len(self.anatomy_difficulty_weights) != 4 or len(self.ct_difficulty_weights) != 3:
            raise ValueError("Difficulty weights must contain four anatomy values and three CT-only values")
        if self.flow_scale <= 0:
            raise ValueError("flow_scale must be positive")


@dataclass
class RefinementInput:
    """Base-registrar output and anatomy signals consumed by the refiner.

    All tensors use ``(B,C,D,H,W)`` layout. ``base_dvf`` must be an additive
    three-channel displacement field in the same voxel/grid convention used by
    the caller's warp operator.
    """

    fixed_image: torch.Tensor
    warped_moving_image: torch.Tensor
    base_dvf: torch.Tensor
    fixed_small_mask: torch.Tensor
    warped_small_mask: torch.Tensor
    fixed_bone_mask: torch.Tensor
    warped_bone_mask: torch.Tensor
    moving_image: Optional[torch.Tensor] = None
    difficulty: Optional[torch.Tensor] = None
    roi_source: Optional[torch.Tensor] = None


@dataclass
class RefinementOutput:
    """Complete, inspectable result of anatomy-aware local refinement."""

    refined_dvf: torch.Tensor
    gated_residual_dvf: torch.Tensor
    scaled_residual_dvf: torch.Tensor
    raw_residual_dvf: torch.Tensor
    difficulty: torch.Tensor
    roi_radius: torch.Tensor
    roi_gate: torch.Tensor
    residual_scale: torch.Tensor
    lambda_small: torch.Tensor
    lambda_smooth: torch.Tensor
    feature_tensor: torch.Tensor
    anatomy_maps: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class PlugInRegistrationOutput:
    """Output returned by :class:`RegistrationWithLocalRefinement`."""

    base_warped_moving: torch.Tensor
    base_dvf: torch.Tensor
    refinement: RefinementOutput
    refined_warped_moving: torch.Tensor
    base_warped_small_mask: torch.Tensor
    refined_warped_small_mask: torch.Tensor
    base_warped_bone_mask: torch.Tensor
    refined_warped_bone_mask: torch.Tensor

    @property
    def refined_dvf(self) -> torch.Tensor:
        return self.refinement.refined_dvf
