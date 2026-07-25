"""Minimal Gaussian-correspondence extension of SACB-Net."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

import utils
from SACB1 import SACB, cross_Sim
from gaussian_anatomy import GaussianAnatomyCorrespondenceModule
from geometry_conditioned_registration import (
    GeometryConditionedResidualCorrector,
)
from model import Encoder, double_conv, tuple_
from nn_util import conv


class GAM_SACB_Net(nn.Module):
    """SACB-Net with one L4 Gaussian correspondence residual.

    The L5, L3, L2, and full-resolution paths are the original SACB-Net
    hierarchy.  At L4, compact Gaussian tokens match fixed and coarse-warped
    moving features.  Their feature residual is rasterized as context for one
    bounded correction of the original SACB dense flow.
    """

    architecture_revision = "minimal_v2"

    def __init__(
        self,
        inshape: Sequence[int] = (128, 160, 160),
        in_c: int = 1,
        ch_scale: int = 4,
        num_k: Union[int, Tuple[int, int, int, int]] = 7,
        scale: float = 1.0,
        mean_type: str = "s",
        token_dim: int = 32,
        token_num_l4: int = 96,
        context_ch: int = 8,
        residual_hidden_ch: int = 32,
        match_temperature: float = 0.10,
        position_cost_weight: float = 0.05,
        max_residual: float = 1.0,
        fix_kmeans_rng: bool = True,
        kmeans_max_iter: int = 20,
        kmeans_tolerance: float = 1.0e-4,
    ) -> None:
        super().__init__()
        self.inshape = tuple(int(value) for value in inshape)
        if len(self.inshape) != 3 or min(self.inshape) < 16:
            raise ValueError("inshape must contain three values of at least 16")
        if any(value % 16 for value in self.inshape):
            raise ValueError("inshape values must be divisible by 16")
        if int(kmeans_max_iter) <= 0 or float(kmeans_tolerance) <= 0.0:
            raise ValueError("KMeans iterations and tolerance must be positive")
        self.ch_scale = int(ch_scale)
        self.scale = float(scale)
        self.mt = str(mean_type)
        self.context_ch = int(context_ch)
        channels = self.ch_scale
        self.num_k = (
            tuple_(num_k, length=4)
            if not isinstance(num_k, tuple)
            else num_k
        )
        if len(self.num_k) != 4:
            raise ValueError("num_k must be an int or a four-element tuple")

        # These modules and names intentionally match the original backbone so
        # a SACB-Net checkpoint can initialize every shared parameter.
        self.encoder = Encoder(in_c=in_c, c=channels)
        activation = ("leakyrelu", {"negative_slope": 0.1})
        self.up_tri = nn.Upsample(
            scale_factor=2,
            mode="trilinear",
            align_corners=True,
        )
        self.conv1 = double_conv(
            2 * channels,
            2 * channels,
            act=activation,
        )
        self.cross_sim = cross_Sim()
        sacb_common = {
            "act": activation,
            "residual": True,
            "fix_rng": bool(fix_kmeans_rng),
            "m_iter": int(kmeans_max_iter),
            "tol": float(kmeans_tolerance),
        }
        self.sacb_proj2 = SACB(
            4 * channels,
            4 * channels,
            in_proj_n=1,
            ks=3,
            mean_type=self.mt,
            num_k=self.num_k[0],
            **sacb_common,
        )
        self.sacb_proj3 = SACB(
            8 * channels,
            8 * channels,
            in_proj_n=1,
            ks=3,
            mean_type=self.mt,
            num_k=self.num_k[1],
            **sacb_common,
        )
        self.sacb_proj4 = SACB(
            16 * channels,
            16 * channels,
            in_proj_n=1,
            ks=3,
            mean_type=self.mt,
            num_k=self.num_k[2],
            **sacb_common,
        )
        self.sacb_proj5 = SACB(
            16 * channels,
            16 * channels,
            in_proj_n=1,
            ks=3,
            mean_type=self.mt,
            num_k=self.num_k[3],
            **sacb_common,
        )
        self.conv1_out = double_conv(
            4 * channels,
            2 * channels,
            act=activation,
            append_fn=conv(
                2 * channels,
                3,
                3,
                1,
                1,
                act=None,
            ),
        )
        self.transformer = nn.ModuleList(
            [
                utils.SpatialTransformer(
                    [size // (2 ** level) for size in self.inshape]
                )
                for level in range(4)
            ]
        )

        size_l4 = tuple(size // 8 for size in self.inshape)
        self.gacm4 = GaussianAnatomyCorrespondenceModule(
            in_ch=16 * channels,
            spatial_size=size_l4,
            num_tokens=int(token_num_l4),
            token_dim=int(token_dim),
            context_ch=self.context_ch,
            match_temperature=float(match_temperature),
            position_cost_weight=float(position_cost_weight),
        )
        self.geometry_corrector4 = GeometryConditionedResidualCorrector(
            feat_ch=16 * channels,
            context_ch=self.context_ch,
            hidden_ch=int(residual_hidden_ch),
            max_residual=float(max_residual),
        )

    def set_k(self, k: Union[int, Tuple[int, int, int, int]]) -> None:
        values = tuple_(k, length=4) if not isinstance(k, tuple) else k
        if len(values) != 4:
            raise ValueError("k must be an int or a four-element tuple")
        self.sacb_proj2.set_num_k(values[0])
        self.sacb_proj3.set_num_k(values[1])
        self.sacb_proj4.set_num_k(values[2])
        self.sacb_proj5.set_num_k(values[3])

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        softsign_last: bool = False,
        return_aux: bool = False,
    ):
        moving1, moving2, moving3, moving4, moving5 = self.encoder(x)
        fixed1, fixed2, fixed3, fixed4, fixed5 = self.encoder(y)

        fixed5_projected = self.sacb_proj5(fixed5)
        moving5_projected = self.sacb_proj5(moving5)
        phi5_native = self.cross_sim(
            moving5_projected,
            fixed5_projected,
        )
        phi5 = self.up_tri(2.0 * phi5_native)

        moving4_warped = self.transformer[3](moving4, phi5)
        gacm4 = self.gacm4(
            moving4_warped,
            fixed4,
            return_aux=False,
        )
        fixed4_projected = self.sacb_proj4(fixed4)
        moving4_projected = self.sacb_proj4(moving4_warped)
        dense4 = self.cross_sim(
            moving4_projected,
            fixed4_projected,
        )
        delta4, residual4 = self.geometry_corrector4(
            moving4_projected,
            fixed4_projected,
            dense4,
            gacm4["context"],
        )
        phi4_native = self.transformer[3](phi5, delta4) + delta4
        phi4 = self.up_tri(2.0 * phi4_native)

        moving3_warped = self.transformer[2](moving3, phi4)
        fixed3_projected = self.sacb_proj3(fixed3)
        moving3_projected = self.sacb_proj3(moving3_warped)
        delta3 = self.cross_sim(
            moving3_projected,
            fixed3_projected,
        )
        phi3_native = self.transformer[2](phi4, delta3) + delta3
        phi3 = self.up_tri(2.0 * phi3_native)

        moving2_warped = self.transformer[1](moving2, phi3)
        fixed2_projected = self.sacb_proj2(fixed2)
        moving2_projected = self.sacb_proj2(moving2_warped)
        delta2 = self.cross_sim(
            moving2_projected,
            fixed2_projected,
        )
        phi2_native = self.transformer[1](phi3, delta2) + delta2
        phi2 = self.up_tri(2.0 * phi2_native)

        moving1_warped = self.transformer[0](moving1, phi2)
        fixed1_projected = self.conv1(fixed1)
        moving1_projected = self.conv1(moving1_warped)
        delta1 = self.conv1_out(
            torch.cat(
                (fixed1_projected, moving1_projected),
                dim=1,
            )
        )
        if softsign_last:
            delta1 = F.softsign(delta1)
        flow = self.transformer[0](phi2, delta1) + delta1
        warped = self.transformer[0](x, flow)

        if not return_aux:
            return warped, flow
        return {
            "warped": warped,
            "flow": flow,
            "gacm4": gacm4,
            "phi5_native": phi5_native,
            "phi4_native": phi4_native,
            "phi3_native": phi3_native,
            "phi2_native": phi2_native,
            "dense4": dense4,
            "delta4": delta4,
            "residual4": residual4,
        }

    def load_sacb_checkpoint(self, path: Union[str, Path]):
        state = torch.load(str(path), map_location="cpu")
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))
        if not isinstance(state, dict):
            raise TypeError("checkpoint must contain a state dictionary")
        state = {
            key.removeprefix("module."): value
            for key, value in state.items()
        }
        return self.load_state_dict(state, strict=False)


__all__ = ["GAM_SACB_Net"]
