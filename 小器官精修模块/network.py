"""Backbone-agnostic network used by local residual refinement."""

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 3-D convolutions with optional instance normalization."""

    def __init__(self, in_channels: int, out_channels: int, instance_norm: bool = True) -> None:
        super().__init__()
        layers = [nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)]
        if instance_norm:
            layers.append(nn.InstanceNorm3d(out_channels, affine=True))
        layers.extend(
            [
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            ]
        )
        if instance_norm:
            layers.append(nn.InstanceNorm3d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LocalResidualUNet(nn.Module):
    """Small 3-D U-Net that predicts an additive residual DVF.

    The default seven channels are assembled by :func:`build_feature_tensor`.
    The class intentionally contains no registration-backbone assumptions.
    """

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 3,
        filters: Sequence[int] = (8, 16, 32),
        instance_norm: bool = True,
        init_std: float = 1e-5,
    ) -> None:
        super().__init__()
        if len(filters) != 3:
            raise ValueError(f"Expected exactly three filter levels, got {filters}")

        f0, f1, f2 = tuple(int(value) for value in filters)
        self.enc0 = ConvBlock(in_channels, f0, instance_norm=instance_norm)
        self.enc1 = ConvBlock(f0, f1, instance_norm=instance_norm)
        self.enc2 = ConvBlock(f1, f2, instance_norm=instance_norm)

        self.up1 = nn.ConvTranspose3d(f2, f1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f1 + f1, f1, instance_norm=instance_norm)
        self.up0 = nn.ConvTranspose3d(f1, f0, kernel_size=2, stride=2)
        self.dec0 = ConvBlock(f0 + f0, f0, instance_norm=instance_norm)

        self.out = nn.Conv3d(f0, out_channels, kernel_size=3, padding=1)
        self.initialize_weights(init_std=init_std)

    def initialize_weights(self, init_std: float = 1e-5) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, a=0.2)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.normal_(self.out.weight, mean=0.0, std=init_std)
        if self.out.bias is not None:
            nn.init.constant_(self.out.bias, 0)

    @staticmethod
    def _match_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] == reference.shape[2:]:
            return x
        return F.interpolate(x, size=reference.shape[2:], mode="trilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip0 = self.enc0(x)
        x = F.avg_pool3d(skip0, kernel_size=2, stride=2)
        skip1 = self.enc1(x)
        x = F.avg_pool3d(skip1, kernel_size=2, stride=2)
        x = self.enc2(x)

        x = self.up1(x)
        x = self._match_size(x, skip1)
        x = self.dec1(torch.cat((x, skip1), dim=1))
        x = self.up0(x)
        x = self._match_size(x, skip0)
        x = self.dec0(torch.cat((x, skip0), dim=1))
        return self.out(x)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return total and trainable parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
