"""Adapter that attaches local refinement to an arbitrary registrar."""

from contextlib import nullcontext
from typing import Any, Callable, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .refiner import AnatomyAwareLocalRefiner
from .types import PlugInRegistrationOutput, RefinementInput


BaseOutputAdapter = Callable[[Any], Tuple[torch.Tensor, torch.Tensor]]


class RegistrationWithLocalRefinement(nn.Module):
    """Compose a registration backbone, warp operator, and local refiner.

    By default the backbone must return ``(warped_moving, dvf)``. Set
    ``base_output_order='dvf-warped'`` or provide ``base_output_adapter`` for a
    different API. The warp operator must support ``warp(src, dvf, mode=...)``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        refiner: AnatomyAwareLocalRefiner,
        warp: nn.Module,
        freeze_backbone: bool = True,
        base_output_order: str = "warped-dvf",
        base_output_adapter: Optional[BaseOutputAdapter] = None,
        image_warp_mode: str = "bilinear",
        mask_warp_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if base_output_order not in ("warped-dvf", "dvf-warped"):
            raise ValueError("base_output_order must be 'warped-dvf' or 'dvf-warped'")
        self.backbone = backbone
        self.refiner = refiner
        self.warp = warp
        self.freeze_backbone = bool(freeze_backbone)
        self.base_output_order = base_output_order
        self.base_output_adapter = base_output_adapter
        self.image_warp_mode = image_warp_mode
        self.mask_warp_mode = mask_warp_mode
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(
        self,
        moving_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moving_small_mask: torch.Tensor,
        fixed_small_mask: torch.Tensor,
        moving_bone_mask: torch.Tensor,
        fixed_bone_mask: torch.Tensor,
        difficulty: Optional[torch.Tensor] = None,
        roi_source: Optional[torch.Tensor] = None,
    ) -> PlugInRegistrationOutput:
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            base_output = self.backbone(moving_image, fixed_image)
            base_warped, base_dvf = self._unpack_base_output(base_output)
        if self.freeze_backbone:
            base_warped = base_warped.detach()
            base_dvf = base_dvf.detach()

        warped_small = self.warp(moving_small_mask.float(), base_dvf, mode=self.mask_warp_mode).clamp(0.0, 1.0)
        warped_bone = self.warp(moving_bone_mask.float(), base_dvf, mode=self.mask_warp_mode).clamp(0.0, 1.0)
        refinement = self.refiner.refine(
            RefinementInput(
                fixed_image=fixed_image,
                warped_moving_image=base_warped,
                base_dvf=base_dvf,
                fixed_small_mask=fixed_small_mask.float(),
                warped_small_mask=warped_small,
                fixed_bone_mask=fixed_bone_mask.float(),
                warped_bone_mask=warped_bone,
                moving_image=moving_image,
                difficulty=difficulty,
                roi_source=roi_source,
            )
        )
        refined_warped = self.warp(moving_image, refinement.refined_dvf, mode=self.image_warp_mode)
        refined_small = self.warp(
            moving_small_mask.float(), refinement.refined_dvf, mode=self.mask_warp_mode
        ).clamp(0.0, 1.0)
        refined_bone = self.warp(
            moving_bone_mask.float(), refinement.refined_dvf, mode=self.mask_warp_mode
        ).clamp(0.0, 1.0)
        return PlugInRegistrationOutput(
            base_warped_moving=base_warped,
            base_dvf=base_dvf,
            refinement=refinement,
            refined_warped_moving=refined_warped,
            base_warped_small_mask=warped_small,
            refined_warped_small_mask=refined_small,
            base_warped_bone_mask=warped_bone,
            refined_warped_bone_mask=refined_bone,
        )

    def _unpack_base_output(self, output: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.base_output_adapter is not None:
            return self.base_output_adapter(output)
        if isinstance(output, Mapping):
            warped = output.get("warped_moving", output.get("deformed"))
            dvf = output.get("dvf", output.get("flow"))
            if warped is None or dvf is None:
                raise ValueError("Mapping backbone output needs warped_moving/deformed and dvf/flow keys")
            return warped, dvf
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise ValueError("Backbone output must be a mapping or a tuple/list with at least two tensors")
        if self.base_output_order == "warped-dvf":
            return output[0], output[1]
        return output[1], output[0]
