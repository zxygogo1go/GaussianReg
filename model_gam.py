"""SACB-Net backbone with GACM and GCDR research modules."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

import utils
from SACB1 import SACB, cross_Sim
from gaussian_anatomy import GaussianAnatomyMatcher3D
from geometry_conditioned_registration import GeometryConditionedDenseRegistrationBlock
from model import Encoder, double_conv, tuple_
from nn_util import conv


def resize_context(context: torch.Tensor, target_size: Sequence[int]) -> torch.Tensor:
    return F.interpolate(context, size=tuple(int(v) for v in target_size), mode="trilinear", align_corners=True)


class GAM_SACB_Net(nn.Module):
    """Gaussian anatomy correspondence conditioned SACB-Net.

    Flow tensors use voxel displacement in ``(D,H,W)`` order and map a fixed
    output grid to moving-image sampling locations.
    """

    def __init__(
        self,
        inshape: Sequence[int] = (128, 160, 160),
        in_c: int = 1,
        ch_scale: int = 4,
        num_k: Union[int, Tuple[int, int, int, int]] = 7,
        scale: float = 1.0,
        mean_type: str = "s",
        token_dim: int = 64,
        token_num_l5: int = 128,
        token_num_l4: int = 192,
        num_types: int = 8,
        context_ch: int = 11,
        fusion_hidden_ch: int = 64,
        fix_kmeans_rng: bool = True,
    ) -> None:
        super().__init__()
        self.inshape = tuple(int(v) for v in inshape)
        if len(self.inshape) != 3 or min(self.inshape) < 16:
            raise ValueError("inshape must contain three values of at least 16")
        self.ch_scale = int(ch_scale)
        self.scale = float(scale)
        self.mt = mean_type
        self.context_ch = int(context_ch)
        c = self.ch_scale
        self.num_k = tuple_(num_k, length=4) if not isinstance(num_k, tuple) else num_k
        if len(self.num_k) != 4:
            raise ValueError("num_k must be an int or a four-element tuple")

        # Shared names and tensor sizes intentionally match the baseline model.
        self.encoder = Encoder(in_c=in_c, c=c)
        act = ("leakyrelu", {"negative_slope": 0.1})
        self.up_tri = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv1 = double_conv(2 * c, 2 * c, act=act)
        self.cross_sim = cross_Sim()
        sacb_common = {"act": act, "residual": True, "cond_ch": context_ch, "fix_rng": bool(fix_kmeans_rng)}
        self.sacb_proj2 = SACB(4 * c, 4 * c, in_proj_n=1, ks=3, mean_type=self.mt, num_k=self.num_k[0], **sacb_common)
        self.sacb_proj3 = SACB(8 * c, 8 * c, in_proj_n=1, ks=3, mean_type=self.mt, num_k=self.num_k[1], **sacb_common)
        self.sacb_proj4 = SACB(16 * c, 16 * c, in_proj_n=1, ks=3, mean_type=self.mt, num_k=self.num_k[2], **sacb_common)
        self.sacb_proj5 = SACB(16 * c, 16 * c, in_proj_n=1, ks=3, mean_type=self.mt, num_k=self.num_k[3], **sacb_common)
        self.conv1_out = double_conv(4 * c, 2 * c, act=act, append_fn=conv(2 * c, 3, 3, 1, 1, act=None))

        self.transformer = nn.ModuleList(
            [utils.SpatialTransformer([size // (2 ** level) for size in self.inshape]) for level in range(4)]
        )

        size_l5 = tuple(size // 16 for size in self.inshape)
        size_l4 = tuple(size // 8 for size in self.inshape)
        self.gacm5 = GaussianAnatomyMatcher3D(
            16 * c,
            size_l5,
            num_tokens=token_num_l5,
            token_dim=token_dim,
            num_types=num_types,
        )
        self.gacm4 = GaussianAnatomyMatcher3D(
            16 * c,
            size_l4,
            num_tokens=token_num_l4,
            token_dim=token_dim,
            num_types=num_types,
        )
        self.gcdr5 = GeometryConditionedDenseRegistrationBlock(
            16 * c,
            context_ch=context_ch,
            hidden_ch=fusion_hidden_ch,
            max_residual=1.0,
        )
        self.gcdr4 = GeometryConditionedDenseRegistrationBlock(
            16 * c,
            context_ch=context_ch,
            hidden_ch=fusion_hidden_ch,
            max_residual=1.0,
        )
        self.context_refiner = nn.Sequential(
            nn.Conv3d(4 * c + 3 + context_ch, 2 * c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(2 * c, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(2 * c, 2 * c, kernel_size=3, padding=1),
            nn.InstanceNorm3d(2 * c, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(2 * c, 3, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.context_refiner[-1].weight)
        nn.init.zeros_(self.context_refiner[-1].bias)

    @property
    def gam5(self):
        """Compatibility alias for the original draft specification."""
        return self.gacm5

    @property
    def gam4(self):
        return self.gacm4

    @property
    def fusion5(self):
        return self.gcdr5

    @property
    def fusion4(self):
        return self.gcdr4

    def set_k(self, k: Union[int, Tuple[int, int, int, int]]) -> None:
        values = tuple_(k, length=4) if not isinstance(k, tuple) else k
        self.sacb_proj5.set_num_k(values[0])
        self.sacb_proj4.set_num_k(values[1])
        self.sacb_proj3.set_num_k(values[2])
        self.sacb_proj2.set_num_k(values[3])

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        softsign_last: bool = False,
        return_aux: bool = False,
    ):
        moving1, moving2, moving3, moving4, moving5 = self.encoder(x)
        fixed1, fixed2, fixed3, fixed4, fixed5 = self.encoder(y)

        gacm5 = self.gacm5(moving5, fixed5, return_aux=return_aux or self.training)
        moving5_projected = self.sacb_proj5(moving5, gacm5["context"])
        fixed5_projected = self.sacb_proj5(fixed5, gacm5["context"])
        dense5 = self.cross_sim(moving5_projected, fixed5_projected)
        phi5_native, gate5 = self.gcdr5(
            moving5_projected,
            fixed5_projected,
            dense5,
            gacm5["flow"],
            gacm5["context"],
        )
        phi5 = self.up_tri(2.0 * phi5_native)

        moving4_warped = self.transformer[3](moving4, phi5)
        gacm4 = self.gacm4(moving4_warped, fixed4, return_aux=return_aux or self.training)
        moving4_projected = self.sacb_proj4(moving4_warped, gacm4["context"])
        fixed4_projected = self.sacb_proj4(fixed4, gacm4["context"])
        dense4 = self.cross_sim(moving4_projected, fixed4_projected)
        delta4, gate4 = self.gcdr4(
            moving4_projected,
            fixed4_projected,
            dense4,
            gacm4["flow"],
            gacm4["context"],
        )
        phi4_native = self.transformer[3](phi5, delta4) + delta4
        phi4 = self.up_tri(2.0 * phi4_native)

        context3 = resize_context(gacm4["context"], moving3.shape[2:])
        moving3_warped = self.transformer[2](moving3, phi4)
        moving3_projected = self.sacb_proj3(moving3_warped, context3)
        fixed3_projected = self.sacb_proj3(fixed3, context3)
        delta3 = self.cross_sim(moving3_projected, fixed3_projected)
        phi3_native = self.transformer[2](phi4, delta3) + delta3
        phi3 = self.up_tri(2.0 * phi3_native)

        context2 = resize_context(gacm4["context"], moving2.shape[2:])
        moving2_warped = self.transformer[1](moving2, phi3)
        moving2_projected = self.sacb_proj2(moving2_warped, context2)
        fixed2_projected = self.sacb_proj2(fixed2, context2)
        delta2 = self.cross_sim(moving2_projected, fixed2_projected)
        phi2_native = self.transformer[1](phi3, delta2) + delta2
        phi2 = self.up_tri(2.0 * phi2_native)

        context1 = resize_context(gacm4["context"], moving1.shape[2:])
        moving1_warped = self.transformer[0](moving1, phi2)
        moving1_projected = self.conv1(moving1_warped)
        fixed1_projected = self.conv1(fixed1)
        delta1_base = self.conv1_out(torch.cat((fixed1_projected, moving1_projected), dim=1))
        delta1_context = self.context_refiner(
            torch.cat((fixed1_projected, moving1_projected, delta1_base, context1), dim=1)
        )
        delta1 = delta1_base + F.softsign(delta1_context)
        if softsign_last:
            delta1 = F.softsign(delta1)
        flow = self.transformer[0](phi2, delta1) + delta1
        warped = self.transformer[0](x, flow)

        if not return_aux:
            return warped, flow
        return {
            "warped": warped,
            "flow": flow,
            "context_full": context1,
            "gacm5": gacm5,
            "gacm4": gacm4,
            "gam5": gacm5,
            "gam4": gacm4,
            "gate5": gate5,
            "gate4": gate4,
            "phi5_native": phi5_native,
            "phi4_native": phi4_native,
            "phi3_native": phi3_native,
            "phi2_native": phi2_native,
            "delta4": delta4,
        }

    def load_sacb_checkpoint(self, path: Union[str, Path]):
        state = torch.load(str(path), map_location="cpu")
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))
        if not isinstance(state, dict):
            raise TypeError("checkpoint must contain a state dictionary")
        state = {key.removeprefix("module."): value for key, value in state.items()}
        return self.load_state_dict(state, strict=False)


__all__ = ["GAM_SACB_Net", "resize_context"]
