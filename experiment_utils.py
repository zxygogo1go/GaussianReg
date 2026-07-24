"""Shared experiment utilities for GAM-SACB-Net training and evaluation."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from losses import (
    AnchorFlowConsistencyLoss,
    GaussianTokenRegularization,
    Grad3d,
    NCC_vxm,
    TransportCostLoss,
)
from model import SACB_Net
from model_gam import GAM_SACB_Net


def load_json(path: str) -> Dict[str, object]:
    with Path(path).open("r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("experiment config must be a JSON object")
    return value


def to_json_safe(value):
    """Convert tensors/NumPy values and non-finite floats to strict JSON."""
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if torch.is_tensor(value):
        return to_json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return to_json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(to_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(str(temporary), str(path))


def set_reproducibility(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def cuda_autocast(enabled: bool):
    """Version-compatible CUDA autocast context (PyTorch 1.13 through 2.x)."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=bool(enabled))
    return torch.cuda.amp.autocast(enabled=bool(enabled))


def make_grad_scaler(
    enabled: bool,
    initial_scale: float = 1024.0,
    growth_interval: int = 2000,
):
    """Version-compatible GradScaler for the original PyTorch 1.13 pin."""
    if initial_scale <= 0.0 or growth_interval <= 0:
        raise ValueError("GradScaler initial scale and growth interval must be positive")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(
            "cuda",
            init_scale=float(initial_scale),
            growth_interval=int(growth_interval),
            enabled=bool(enabled),
        )
    return torch.cuda.amp.GradScaler(
        init_scale=float(initial_scale),
        growth_interval=int(growth_interval),
        enabled=bool(enabled),
    )


class BaselineSACBNet(SACB_Net):
    """Training-interface adapter around the unchanged SACB-Net architecture."""

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        softsign_last: bool = False,
        return_aux: bool = False,
    ):
        warped, flow = super().forward(x, y, softsign_last=softsign_last)
        if return_aux:
            return {"warped": warped, "flow": flow}
        return warped, flow


def config_architecture(config: Mapping[str, object]) -> str:
    """Return the canonical architecture name stored in an experiment config."""
    model = dict(config.get("model", {}))
    architecture = str(model.get("architecture", "gam_sacb")).strip().lower().replace("-", "_")
    aliases = {
        "gam": "gam_sacb",
        "gam_sacb": "gam_sacb",
        "gam_sacb_net": "gam_sacb",
        "sacb": "sacb",
        "sacb_net": "sacb",
        "baseline_sacb": "sacb",
    }
    if architecture not in aliases:
        raise ValueError("unsupported model.architecture: %s" % architecture)
    return aliases[architecture]


def _configure_sacb_kmeans(
    model: nn.Module,
    fix_rng: bool,
    max_iter: int,
    tolerance: float,
) -> None:
    """Apply the same deterministic KMeans runtime policy to either architecture."""
    if max_iter <= 0 or tolerance <= 0.0:
        raise ValueError("KMeans iterations and tolerance must be positive")
    for module in model.modules():
        km_wrapper = getattr(module, "km", None)
        kmeans = getattr(km_wrapper, "km", None)
        if kmeans is None:
            continue
        km_wrapper.fix_rng = bool(fix_rng)
        kmeans.max_iter = int(max_iter)
        kmeans.tolerance = float(tolerance)


def build_model(config: Mapping[str, object]) -> nn.Module:
    data = dict(config.get("data", {}))
    model = dict(config.get("model", {}))
    shape = tuple(int(v) for v in data.get("shape_dhw", (128, 160, 160)))
    if len(shape) != 3 or any(value % 16 for value in shape):
        raise ValueError("data.shape_dhw must contain three values divisible by 16")
    num_k = model.get("num_k", 7)
    if isinstance(num_k, list):
        num_k = tuple(int(value) for value in num_k)
    common = {
        "inshape": shape,
        "in_c": int(model.get("in_channels", 1)),
        "ch_scale": int(model.get("channel_scale", 4)),
        "num_k": num_k,
        "scale": float(model.get("scale", 1.0)),
        "mean_type": str(model.get("mean_type", "s")),
    }
    architecture = config_architecture(config)
    if architecture == "sacb":
        baseline = BaselineSACBNet(**common)
        _configure_sacb_kmeans(
            baseline,
            fix_rng=bool(model.get("fix_kmeans_rng", True)),
            max_iter=int(model.get("kmeans_max_iter", 20)),
            tolerance=float(model.get("kmeans_tolerance", 1.0e-4)),
        )
        return baseline
    return GAM_SACB_Net(
        **common,
        token_dim=int(model.get("token_dim", 64)),
        token_num_l5=int(model.get("token_num_l5", 128)),
        token_num_l4=int(model.get("token_num_l4", 192)),
        num_types=int(model.get("num_types", 8)),
        context_ch=int(model.get("context_channels", 11)),
        fusion_hidden_ch=int(model.get("fusion_hidden_channels", 64)),
        fix_kmeans_rng=bool(model.get("fix_kmeans_rng", True)),
        kmeans_max_iter=int(model.get("kmeans_max_iter", 20)),
        kmeans_tolerance=float(model.get("kmeans_tolerance", 1.0e-4)),
    )


def warp_volume(source: torch.Tensor, flow_dhw: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    """Warp ``source`` with fixed-to-moving sampling flow in DHW voxel units."""
    if source.ndim != 5 or flow_dhw.ndim != 5 or flow_dhw.shape[1] != 3:
        raise AssertionError("source and flow must be [B,C,D,H,W] and [B,3,D,H,W]")
    if source.shape[0] != flow_dhw.shape[0] or source.shape[2:] != flow_dhw.shape[2:]:
        raise AssertionError("source and flow batch/spatial shapes must match")
    spatial = flow_dhw.shape[2:]
    axes = [
        torch.linspace(-1.0, 1.0, int(size), device=source.device, dtype=source.dtype)
        for size in spatial
    ]
    dd, hh, ww = torch.meshgrid(*axes, indexing="ij")
    identity = torch.stack((dd, hh, ww), dim=0).unsqueeze(0)
    scale = flow_dhw.new_tensor(
        [2.0 / max(spatial[0] - 1, 1), 2.0 / max(spatial[1] - 1, 1), 2.0 / max(spatial[2] - 1, 1)]
    ).view(1, 3, 1, 1, 1)
    sampling_dhw = identity + flow_dhw.to(source.dtype) * scale.to(source.dtype)
    sampling_whd = sampling_dhw[:, [2, 1, 0]].permute(0, 2, 3, 4, 1)
    return F.grid_sample(
        source,
        sampling_whd,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )


class RegistrationObjective(nn.Module):
    """Unsupervised image/geometry objective for the dual-module model."""

    def __init__(self, loss_config: Mapping[str, object]) -> None:
        super().__init__()
        self.weights = {
            "similarity": float(loss_config.get("similarity", 1.0)),
            "smoothness": float(loss_config.get("smoothness", 0.3)),
            "deep_similarity": float(loss_config.get("deep_similarity", 1.0)),
            "token": float(loss_config.get("token", 0.01)),
            "transport": float(loss_config.get("transport", 0.02)),
            "anchor": float(loss_config.get("anchor", 0.05)),
        }
        if any(value < 0.0 for value in self.weights.values()):
            raise ValueError("loss weights must be nonnegative")
        self.deep_scale_weights = {
            "phi4_native": 0.05,
            "phi3_native": 0.10,
            "phi2_native": 0.15,
        }
        configured_deep = loss_config.get("deep_scale_weights")
        if configured_deep is not None:
            self.deep_scale_weights.update({key: float(value) for key, value in dict(configured_deep).items()})
        ncc_window = int(loss_config.get("ncc_window", 9))
        deep_ncc_window = int(loss_config.get("deep_ncc_window", 5))
        if ncc_window <= 0 or ncc_window % 2 == 0 or deep_ncc_window <= 0 or deep_ncc_window % 2 == 0:
            raise ValueError("NCC windows must be positive odd integers")
        self.ncc = NCC_vxm(win=[ncc_window] * 3)
        self.deep_ncc = NCC_vxm(win=[deep_ncc_window] * 3)
        self.smoothness = Grad3d(penalty="l2")
        self.token_regularization = GaussianTokenRegularization(
            repulsion_scale=float(loss_config.get("token_repulsion_scale", 0.08))
        )
        self.transport_cost = TransportCostLoss()
        self.anchor_consistency = AnchorFlowConsistencyLoss()

    def forward(
        self,
        output: Mapping[str, object],
        moving: torch.Tensor,
        fixed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        similarity = self.ncc(fixed, output["warped"])
        smoothness = self.smoothness(output["flow"], None)

        deep_similarity = similarity.new_zeros(())
        for name, scale_weight in self.deep_scale_weights.items():
            flow = output[name]
            moving_scaled = F.interpolate(moving, size=flow.shape[2:], mode="trilinear", align_corners=True)
            fixed_scaled = F.interpolate(fixed, size=flow.shape[2:], mode="trilinear", align_corners=True)
            deep_similarity = deep_similarity + scale_weight * self.deep_ncc(
                fixed_scaled,
                warp_volume(moving_scaled, flow),
            )

        token_terms = []
        transport_terms = []
        for level in ("gacm5", "gacm4"):
            gacm = output[level]
            token_terms.extend(
                (
                    self.token_regularization(gacm["moving_tokens"]),
                    self.token_regularization(gacm["fixed_tokens"]),
                )
            )
            transport_terms.append(self.transport_cost(gacm["transport"], gacm["cost"]))
        token = torch.stack(token_terms).mean()
        transport = torch.stack(transport_terms).mean()
        anchor = 0.5 * (
            self.anchor_consistency(
                output["phi5_native"],
                output["gacm5"]["fixed_tokens"].mu,
                output["gacm5"]["anchor_disp"],
                output["gacm5"]["anchor_conf"],
            )
            + self.anchor_consistency(
                output["delta4"],
                output["gacm4"]["fixed_tokens"].mu,
                output["gacm4"]["anchor_disp"],
                output["gacm4"]["anchor_conf"],
            )
        )
        terms = {
            "similarity": similarity,
            "smoothness": smoothness,
            "deep_similarity": deep_similarity,
            "token": token,
            "transport": transport,
            "anchor": anchor,
        }
        total = sum(self.weights[name] * value for name, value in terms.items())
        terms["total"] = total
        return terms


class BaselineRegistrationObjective(nn.Module):
    """Original SACB-Net objective using only losses shared with the full model."""

    def __init__(self, loss_config: Mapping[str, object]) -> None:
        super().__init__()
        self.weights = {
            "similarity": float(loss_config.get("similarity", 1.0)),
            "smoothness": float(loss_config.get("smoothness", 0.3)),
        }
        if any(value < 0.0 for value in self.weights.values()):
            raise ValueError("loss weights must be nonnegative")
        ncc_window = int(loss_config.get("ncc_window", 9))
        if ncc_window <= 0 or ncc_window % 2 == 0:
            raise ValueError("NCC window must be a positive odd integer")
        self.ncc = NCC_vxm(win=[ncc_window] * 3)
        self.smoothness = Grad3d(penalty="l2")

    def forward(
        self,
        output: Mapping[str, object],
        moving: torch.Tensor,
        fixed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        del moving
        similarity = self.ncc(fixed, output["warped"])
        smoothness = self.smoothness(output["flow"], None)
        terms = {
            "similarity": similarity,
            "smoothness": smoothness,
        }
        terms["total"] = sum(self.weights[name] * value for name, value in terms.items())
        return terms


def build_objective(config: Mapping[str, object]) -> nn.Module:
    loss_config = dict(config.get("loss", {}))
    if config_architecture(config) == "sacb":
        return BaselineRegistrationObjective(loss_config)
    return RegistrationObjective(loss_config)


def output_diagnostics(output: Mapping[str, object]) -> Dict[str, float]:
    if "gate5" not in output:
        return {}
    result = {
        "gate5": float(output["gate5"].detach().float().mean().cpu()),
        "gate4": float(output["gate4"].detach().float().mean().cpu()),
    }
    for level in ("gacm5", "gacm4"):
        gacm = output[level]
        result[level + "_visibility"] = float(
            0.5
            * (
                gacm["moving_tokens"].visibility.detach().float().mean()
                + gacm["fixed_tokens"].visibility.detach().float().mean()
            ).cpu()
        )
        result[level + "_transport_mass"] = float(gacm["transport"].detach().float().sum(dim=(1, 2)).mean().cpu())
    return result


def learning_rate_factor(epoch: int, epochs: int, warmup_epochs: int, minimum_factor: float) -> float:
    if epochs <= 0 or warmup_epochs < 0 or warmup_epochs >= epochs or not 0.0 <= minimum_factor <= 1.0:
        raise ValueError("invalid scheduler configuration")
    if warmup_epochs and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    denominator = max(epochs - warmup_epochs - 1, 1)
    progress = min(max(float(epoch - warmup_epochs) / float(denominator), 0.0), 1.0)
    return minimum_factor + 0.5 * (1.0 - minimum_factor) * (1.0 + math.cos(math.pi * progress))


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, str(temporary))
    os.replace(str(temporary), str(path))


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def bootstrap_mean_ci(
    values: Sequence[float],
    samples: int = 2000,
    seed: int = 2026,
) -> Dict[str, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0}
    if array.size == 1:
        return {"mean": float(array[0]), "ci95_low": float(array[0]), "ci95_high": float(array[0]), "n": 1}
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        estimates[index] = rng.choice(array, size=array.size, replace=True).mean()
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.percentile(estimates, 2.5)),
        "ci95_high": float(np.percentile(estimates, 97.5)),
        "n": int(array.size),
    }


__all__ = [
    "BaselineRegistrationObjective",
    "BaselineSACBNet",
    "RegistrationObjective",
    "atomic_torch_save",
    "bootstrap_mean_ci",
    "build_model",
    "build_objective",
    "config_architecture",
    "cuda_autocast",
    "finite_mean",
    "learning_rate_factor",
    "load_json",
    "make_grad_scaler",
    "output_diagnostics",
    "resolve_device",
    "save_json",
    "seed_worker",
    "set_reproducibility",
    "to_json_safe",
    "warp_volume",
]
