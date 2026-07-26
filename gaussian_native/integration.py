"""Dense warping and stationary-velocity scaling-and-squaring."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def warp_tensor(
    source: torch.Tensor,
    flow_dhw: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """Warp a tensor with fixed-grid to source sampling flow in DHW voxels."""
    if source.ndim != 5 or flow_dhw.ndim != 5 or flow_dhw.shape[1] != 3:
        raise AssertionError("source and flow must be [B,C,D,H,W] and [B,3,D,H,W]")
    if source.shape[0] != flow_dhw.shape[0] or source.shape[2:] != flow_dhw.shape[2:]:
        raise AssertionError("source and flow batch/spatial shapes must match")
    spatial = flow_dhw.shape[2:]
    axes = [
        torch.linspace(-1.0, 1.0, int(size), device=source.device, dtype=flow_dhw.dtype)
        for size in spatial
    ]
    dd, hh, ww = torch.meshgrid(*axes, indexing="ij")
    identity = torch.stack((dd, hh, ww), dim=0).unsqueeze(0)
    scale = flow_dhw.new_tensor(
        [
            2.0 / max(int(spatial[0]) - 1, 1),
            2.0 / max(int(spatial[1]) - 1, 1),
            2.0 / max(int(spatial[2]) - 1, 1),
        ]
    ).view(1, 3, 1, 1, 1)
    sampling_dhw = identity + flow_dhw * scale
    sampling_whd = sampling_dhw[:, [2, 1, 0]].permute(0, 2, 3, 4, 1)
    return F.grid_sample(
        source,
        sampling_whd,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


def compose_displacements(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compose ``second ∘ first`` for sampling displacements."""
    return first + warp_tensor(second, first, padding_mode="border")


class ScalingAndSquaring(nn.Module):
    def __init__(self, steps: int = 7) -> None:
        super().__init__()
        if steps <= 0:
            raise ValueError("scaling-and-squaring steps must be positive")
        self.steps = int(steps)

    def forward(self, velocity_vox: torch.Tensor) -> torch.Tensor:
        flow = velocity_vox.float() / float(2 ** self.steps)
        for _ in range(self.steps):
            flow = compose_displacements(flow, flow)
        return flow


__all__ = ["ScalingAndSquaring", "compose_displacements", "warp_tensor"]
