"""End-to-end anatomy-aware local residual refiner."""

from typing import Optional, Sequence, Union

import torch

from .functional import (
    build_anatomy_maps,
    build_feature_tensor,
    build_roi_gate_per_batch,
    conditioning_masks,
    difficulty_to_radius,
    difficulty_to_value,
    estimate_anatomy_difficulty,
    estimate_ct_only_difficulty,
)
from .network import LocalResidualUNet
from .types import LocalRefinementConfig, RefinementInput, RefinementOutput


class AnatomyAwareLocalRefiner(LocalResidualUNet):
    """Plug-in anatomy-aware additive DVF refinement module.

    Calling the module with a tensor preserves the legacy U-Net behavior and
    returns a raw residual DVF. Calling it with :class:`RefinementInput` runs
    the complete dynamic controller, ROI gate, residual scaling, and DVF sum.

    The trainable layer names are identical to ``LocalResidualUNet`` so old
    Stage-3 checkpoints load without key conversion.
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 3,
        filters: Sequence[int] = (8, 16, 32),
        instance_norm: bool = True,
        init_std: float = 1e-5,
        config: Optional[LocalRefinementConfig] = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            filters=filters,
            instance_norm=instance_norm,
            init_std=init_std,
        )
        self.config = config or LocalRefinementConfig()

    def forward(self, value: Union[torch.Tensor, RefinementInput]):
        if isinstance(value, torch.Tensor):
            return super().forward(value)
        if isinstance(value, RefinementInput):
            return self.refine(value)
        raise TypeError(f"Expected torch.Tensor or RefinementInput, got {type(value)}")

    def refine(self, inputs: RefinementInput) -> RefinementOutput:
        """Refine a base registration result without knowing its backbone."""

        self._validate_inputs(inputs)
        conditioned = conditioning_masks(
            input_mode=self.config.input_mode,
            fixed_small_mask=inputs.fixed_small_mask,
            warped_small_mask=inputs.warped_small_mask,
            fixed_bone_mask=inputs.fixed_bone_mask,
        )
        difficulty = self._resolve_difficulty(inputs)
        roi_radius = difficulty_to_radius(
            difficulty,
            radius_min=self.config.roi_radius_min,
            radius_max=self.config.roi_radius_max,
        )
        roi_source = inputs.roi_source if inputs.roi_source is not None else conditioned["roi_source"]
        roi_gate = build_roi_gate_per_batch(
            roi_source,
            radii=roi_radius,
            smooth_steps=self.config.roi_smooth_steps,
        )
        feature_tensor = build_feature_tensor(
            fixed_image=inputs.fixed_image,
            warped_moving_image=inputs.warped_moving_image,
            fixed_small_mask=conditioned["fixed_small_feature"],
            warped_small_mask=inputs.warped_small_mask,
            base_dvf=inputs.base_dvf,
            fixed_bone_mask=conditioned["fixed_bone_feature"],
            warped_bone_mask=inputs.warped_bone_mask,
        )
        expected_channels = self.enc0.block[0].in_channels
        if feature_tensor.shape[1] != expected_channels:
            raise ValueError(
                f"Refinement feature tensor has {feature_tensor.shape[1]} channels, "
                f"but the network expects {expected_channels}"
            )

        raw_residual = super().forward(feature_tensor)
        residual_scale = difficulty_to_value(
            difficulty,
            self.config.residual_scale_min,
            self.config.residual_scale_max,
        ).view(-1, 1, 1, 1, 1)
        scaled_residual = raw_residual * residual_scale
        gated_residual = scaled_residual * roi_gate
        refined_dvf = inputs.base_dvf + gated_residual
        anatomy_maps = build_anatomy_maps(
            fixed_bone_mask=conditioned["anatomy_bone"],
            roi_gate=roi_gate,
            difficulty=difficulty,
        )
        lambda_small = difficulty_to_value(
            difficulty,
            self.config.lambda_small_min,
            self.config.lambda_small_max,
        )
        lambda_smooth = self.config.lambda_smooth_base + self.config.lambda_smooth_extra * difficulty

        return RefinementOutput(
            refined_dvf=refined_dvf,
            gated_residual_dvf=gated_residual,
            scaled_residual_dvf=scaled_residual,
            raw_residual_dvf=raw_residual,
            difficulty=difficulty,
            roi_radius=roi_radius,
            roi_gate=roi_gate,
            residual_scale=residual_scale,
            lambda_small=lambda_small,
            lambda_smooth=lambda_smooth,
            feature_tensor=feature_tensor,
            anatomy_maps=anatomy_maps,
        )

    def _resolve_difficulty(self, inputs: RefinementInput) -> torch.Tensor:
        if inputs.difficulty is not None:
            difficulty = inputs.difficulty.to(device=inputs.base_dvf.device, dtype=inputs.base_dvf.dtype).flatten()
            if difficulty.numel() == 1:
                difficulty = difficulty.expand(inputs.base_dvf.shape[0])
            if difficulty.numel() != inputs.base_dvf.shape[0]:
                raise ValueError(
                    f"Expected one difficulty value per batch item, got {difficulty.numel()} "
                    f"for batch size {inputs.base_dvf.shape[0]}"
                )
            return difficulty.clamp(0.0, 1.0)

        if self.config.input_mode == "no-fixed-seg":
            if inputs.moving_image is None:
                raise ValueError("moving_image is required for automatic difficulty in no-fixed-seg mode")
            return estimate_ct_only_difficulty(
                moving_image=inputs.moving_image,
                fixed_image=inputs.fixed_image,
                warped_moving_image=inputs.warped_moving_image,
                base_dvf=inputs.base_dvf,
                weights=self.config.ct_difficulty_weights,
                flow_scale=self.config.flow_scale,
            )

        small_roi = torch.maximum(inputs.fixed_small_mask, inputs.warped_small_mask.detach())
        return estimate_anatomy_difficulty(
            fixed_image=inputs.fixed_image,
            warped_moving_image=inputs.warped_moving_image,
            base_dvf=inputs.base_dvf,
            warped_small_mask=inputs.warped_small_mask,
            fixed_small_mask=inputs.fixed_small_mask,
            warped_bone_mask=inputs.warped_bone_mask,
            fixed_bone_mask=inputs.fixed_bone_mask,
            image_mask=small_roi,
            weights=self.config.anatomy_difficulty_weights,
            flow_scale=self.config.flow_scale,
        )

    @staticmethod
    def _validate_inputs(inputs: RefinementInput) -> None:
        if inputs.base_dvf.ndim != 5:
            raise ValueError(f"base_dvf must use (B,3,D,H,W), got {tuple(inputs.base_dvf.shape)}")
        named_tensors = {
            "fixed_image": inputs.fixed_image,
            "warped_moving_image": inputs.warped_moving_image,
            "base_dvf": inputs.base_dvf,
            "fixed_small_mask": inputs.fixed_small_mask,
            "warped_small_mask": inputs.warped_small_mask,
            "fixed_bone_mask": inputs.fixed_bone_mask,
            "warped_bone_mask": inputs.warped_bone_mask,
        }
        batch_size = inputs.base_dvf.shape[0]
        spatial_shape = inputs.base_dvf.shape[2:]
        for name, tensor in named_tensors.items():
            if tensor.ndim != 5:
                raise ValueError(f"{name} must use (B,C,D,H,W), got {tuple(tensor.shape)}")
            if tensor.shape[0] != batch_size or tensor.shape[2:] != spatial_shape:
                raise ValueError(
                    f"{name} shape {tuple(tensor.shape)} is incompatible with "
                    f"base_dvf {tuple(inputs.base_dvf.shape)}"
                )
            if tensor.device != inputs.base_dvf.device:
                raise ValueError(f"{name} and base_dvf must be on the same device")
        if inputs.base_dvf.shape[1] != 3:
            raise ValueError(f"base_dvf must have three channels, got {inputs.base_dvf.shape[1]}")
        if inputs.fixed_image.shape[1] != inputs.warped_moving_image.shape[1]:
            raise ValueError("fixed_image and warped_moving_image must have the same channel count")
        for name in ("fixed_small_mask", "warped_small_mask", "fixed_bone_mask", "warped_bone_mask"):
            if named_tensors[name].shape[1] != 1:
                raise ValueError(f"{name} must have one channel")
        if inputs.moving_image is not None:
            if inputs.moving_image.shape != inputs.fixed_image.shape:
                raise ValueError("moving_image and fixed_image must have identical shapes")
            if inputs.moving_image.device != inputs.base_dvf.device:
                raise ValueError("moving_image and base_dvf must be on the same device")
        if inputs.roi_source is not None:
            if inputs.roi_source.shape != inputs.fixed_small_mask.shape:
                raise ValueError("roi_source and fixed_small_mask must have identical shapes")
            if inputs.roi_source.device != inputs.base_dvf.device:
                raise ValueError("roi_source and base_dvf must be on the same device")
