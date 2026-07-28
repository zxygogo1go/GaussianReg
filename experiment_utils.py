"""Shared experiment utilities for controlled registration experiments."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from losses import Grad3d, NCC_vxm
from model import SACB_Net
from gaussian_native import GaussianNativeObjective, GaussianNativeRegistration


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


def cuda_autocast(
    enabled: bool,
    dtype: str = "float16",
    cache_enabled: bool = False,
):
    """Version-compatible CUDA autocast with explicit weight-cache policy.

    The Gaussian matcher is called first for fixed-to-fixed calibration under
    ``no_grad`` and then for fixed-to-moving registration with gradients. AMP's
    weight cache can otherwise reuse the detached calibration cast and silently
    disconnect the trainable matcher. The safe default is therefore disabled.
    """
    normalized = str(dtype).strip().lower()
    if normalized not in {"float16", "bfloat16"}:
        raise ValueError("autocast dtype must be float16 or bfloat16")
    torch_dtype = torch.float16 if normalized == "float16" else torch.bfloat16
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(
            device_type="cuda",
            enabled=bool(enabled),
            dtype=torch_dtype,
            cache_enabled=bool(cache_enabled),
        )
    return torch.cuda.amp.autocast(
        enabled=bool(enabled),
        dtype=torch_dtype,
        cache_enabled=bool(cache_enabled),
    )


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
    architecture = str(model.get("architecture", "gaussian_native")).strip().lower().replace("-", "_")
    aliases = {
        "gaussian": "gaussian_native",
        "gaussian_native": "gaussian_native",
        "gaussian_native_registration": "gaussian_native",
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
    architecture = config_architecture(config)
    if architecture == "sacb":
        num_k = model.get("num_k", 7)
        if isinstance(num_k, list):
            num_k = tuple(int(value) for value in num_k)
        baseline = BaselineSACBNet(
            inshape=shape,
            in_c=int(model.get("in_channels", 1)),
            ch_scale=int(model.get("channel_scale", 4)),
            num_k=num_k,
            scale=float(model.get("scale", 1.0)),
            mean_type=str(model.get("mean_type", "s")),
        )
        _configure_sacb_kmeans(
            baseline,
            fix_rng=bool(model.get("fix_kmeans_rng", True)),
            max_iter=int(model.get("kmeans_max_iter", 20)),
            tolerance=float(model.get("kmeans_tolerance", 1.0e-4)),
        )
        return baseline
    root_grid = tuple(int(value) for value in model.get("root_grid_shape", (4, 4, 4)))
    children_per_parent = int(model.get("children_per_parent", 4))
    root_count = int(np.prod(root_grid))
    expected_counts = [
        root_count,
        root_count * children_per_parent,
        root_count * children_per_parent * children_per_parent,
    ]
    configured_counts = model.get("gaussian_counts")
    if configured_counts is not None and [
        int(value) for value in configured_counts
    ] != expected_counts:
        raise ValueError(
            "model.gaussian_counts must match root_grid_shape and children_per_parent"
        )
    spacing = tuple(
        float(value)
        for value in data.get("spacing_dhw", (1.5, 1.5, 1.5))
    )
    architecture_revision = str(
        model.get("architecture_revision", "gaussian_native_v3")
    ).strip().lower()
    stable_motion_basis = architecture_revision in {
        "gaussian_native_v3",
        "gaussian_native_v4",
        "gaussian_native_v5",
        "gaussian_native_v6",
        "gaussian_native_v7",
    }
    direct_limits = model.get(
        "direct_displacement_limits_mm",
        (12.0, 6.0, 3.0) if stable_motion_basis else None,
    )
    learned_fractions = model.get(
        "learned_translation_fractions",
        (0.20, 0.12, 0.08) if stable_motion_basis else None,
    )
    return GaussianNativeRegistration(
        inshape=shape,
        spacing_dhw=spacing,
        root_grid_shape=root_grid,
        children_per_parent=children_per_parent,
        feature_dim=int(model.get("feature_dim", 96)),
        hidden_dim=int(model.get("hidden_dim", 128)),
        graph_heads=int(model.get("graph_heads", 4)),
        graph_neighbors=int(model.get("graph_neighbors", 16)),
        graph_blocks_per_level=int(model.get("graph_blocks_per_level", 2)),
        samples_per_axis=int(model.get("samples_per_axis", 3)),
        pyramid_factors=tuple(int(value) for value in model.get("pyramid_factors", (16, 8, 4))),
        sinkhorn_iterations=int(model.get("sinkhorn_iterations", 12)),
        match_temperature=float(model.get("match_temperature", 0.08)),
        position_weight=float(model.get("position_weight", 0.12)),
        scale_weight=float(model.get("scale_weight", 0.04)),
        dustbin_mass=float(model.get("dustbin_mass", 0.15)),
        parent_candidates=int(model.get("parent_candidates", 4)),
        velocity_hidden_dim=int(model.get("velocity_hidden_dim", 192)),
        raster_chunk=int(model.get("raster_chunk", 64)),
        cutoff_sigma=float(model.get("cutoff_sigma", 3.5)),
        integration_steps=int(model.get("integration_steps", 7)),
        motion_mode=str(model.get("motion_mode", "affine")),
        integration_mode=str(model.get("integration_mode", "svf")),
        covariance_mode=str(model.get("covariance_mode", "full")),
        geometry_mode=str(model.get("geometry_mode", "adaptive")),
        architecture_revision=architecture_revision,
        appearance_weight=float(model.get("appearance_weight", 0.0)),
        transport_mode=str(model.get("transport_mode", "sinkhorn")),
        correspondence_score_mode=str(
            model.get("correspondence_score_mode", "convex")
        ),
        feature_residual_weight=float(
            model.get("feature_residual_weight", 0.0)
        ),
        max_feature_residual_logit=float(
            model.get("max_feature_residual_logit", 2.0)
        ),
        pair_score_hidden_dim=int(
            model.get("pair_score_hidden_dim", 32)
        ),
        include_identity_candidate=(
            None
            if "include_identity_candidate" not in model
            else bool(model["include_identity_candidate"])
        ),
        match_evidence_power=float(
            model.get("match_evidence_power", 1.0)
        ),
        direct_displacement_fractions=tuple(
            float(value)
            for value in model.get(
                "direct_displacement_fractions",
                (1.0, 1.0, 1.0),
            )
        ),
        direct_displacement_limit=float(
            model.get("direct_displacement_limit", 1.5)
        ),
        direct_displacement_limits_mm=(
            None
            if direct_limits is None
            else tuple(float(value) for value in direct_limits)
        ),
        learned_translation_fractions=(
            None
            if learned_fractions is None
            else tuple(float(value) for value in learned_fractions)
        ),
        max_rotation_radians=(
            None
            if model.get("max_rotation_radians") is None
            else float(model["max_rotation_radians"])
        ),
        max_strain=(
            None
            if model.get("max_strain") is None
            else float(model["max_strain"])
        ),
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


class JacobianFoldingLoss(nn.Module):
    """Penalize local Jacobian determinants below a configurable margin."""

    def __init__(self, margin: float = 0.0) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        if flow.ndim != 5 or flow.shape[1] != 3:
            raise AssertionError("flow must have shape [B,3,D,H,W]")
        center = flow[:, :, :-1, :-1, :-1].float()
        derivative_d = flow[:, :, 1:, :-1, :-1].float() - center
        derivative_h = flow[:, :, :-1, 1:, :-1].float() - center
        derivative_w = flow[:, :, :-1, :-1, 1:].float() - center
        j00 = 1.0 + derivative_d[:, 0]
        j01 = derivative_h[:, 0]
        j02 = derivative_w[:, 0]
        j10 = derivative_d[:, 1]
        j11 = 1.0 + derivative_h[:, 1]
        j12 = derivative_w[:, 1]
        j20 = derivative_d[:, 2]
        j21 = derivative_h[:, 2]
        j22 = 1.0 + derivative_w[:, 2]
        determinant = (
            j00 * (j11 * j22 - j12 * j21)
            - j01 * (j10 * j22 - j12 * j20)
            + j02 * (j10 * j21 - j11 * j20)
        )
        return F.relu(self.margin - determinant).square().mean()


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
    return GaussianNativeObjective(loss_config)


def output_diagnostics(output: Mapping[str, object]) -> Dict[str, float]:
    if "correspondence" not in output:
        return {}
    diagnostics = {
        "velocity_vox_abs": float(
            output["velocity_vox"].detach().float().abs().mean().cpu()
        ),
        "inverse_composition_abs": float(
            (
                output["flow"].detach().float()
                + warp_volume(
                    output["inverse_flow"].detach().float(),
                    output["flow"].detach().float(),
                )
            ).abs().mean().cpu()
        ),
    }
    for index, result in enumerate(output["correspondence"]):
        plan = result["plan"].detach().float()
        row = plan / plan.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        entropy = -(row * row.clamp_min(1.0e-8).log()).sum(dim=-1)
        normalizer = max(math.log(float(plan.shape[-1])), 1.0)
        diagnostics["match_entropy_l%d" % index] = float(
            (entropy / normalizer).mean().cpu()
        )
        diagnostics["matched_mass_l%d" % index] = float(
            result["matched_mass_fraction"].detach().float().mean().cpu()
        )
        diagnostics["mutual_concentration_l%d" % index] = float(
            result["mutual_concentration"].detach().float().cpu()
        )
        diagnostics["row_max_probability_l%d" % index] = diagnostics[
            "mutual_concentration_l%d" % index
        ]
        if "support_entropy" in result:
            diagnostics["support_entropy_l%d" % index] = float(
                result["support_entropy"].detach().float().mean().cpu()
            )
        if "match_evidence" in result:
            diagnostics["match_evidence_l%d" % index] = float(
                result["match_evidence"].detach().float().mean().cpu()
            )
        if "support_size" in result:
            diagnostics["support_size_l%d" % index] = float(
                result["support_size"].detach().float().mean().cpu()
            )
        if "feature_residual_logit" in result:
            diagnostics["feature_residual_logit_l%d" % index] = float(
                result["feature_residual_logit"]
                .detach()
                .float()
                .abs()
                .mean()
                .cpu()
            )
        if plan.shape[1] == plan.shape[2]:
            diagnostics["diagonal_probability_l%d" % index] = float(
                torch.diagonal(row, dim1=1, dim2=2).mean().cpu()
            )
        if "transport_delta_mm" in result:
            diagnostics["transport_delta_l%d_mm" % index] = float(
                torch.linalg.vector_norm(
                    result["transport_delta_mm"].detach().float(),
                    dim=-1,
                ).mean().cpu()
            )
    for index, parameters in enumerate(output.get("local_velocities", ())):
        diagnostics["motion_evidence_l%d" % index] = float(
            parameters.motion_evidence.detach().float().mean().cpu()
        )
        diagnostics["direct_translation_l%d_mm" % index] = float(
            torch.linalg.vector_norm(
                parameters.direct_translation_mm.detach().float(),
                dim=-1,
            ).mean().cpu()
        )
        diagnostics["learned_translation_l%d_mm" % index] = float(
            torch.linalg.vector_norm(
                parameters.learned_translation_mm.detach().float(),
                dim=-1,
            ).mean().cpu()
        )
    for index, field in enumerate(output.get("level_velocity_mm", ())):
        diagnostics["level_velocity_l%d_abs_mm" % index] = float(
            field.detach().float().abs().mean().cpu()
        )
    fixed_levels = output.get("fixed_decomposition", {}).get("levels", ())
    for index, level in enumerate(fixed_levels):
        if level.anchor_centers_mm is None or level.anchor_scales_mm is None:
            continue
        normalized_offset = (
            (level.centers_mm - level.anchor_centers_mm)
            / level.anchor_scales_mm.clamp_min(1.0e-3)
        )
        diagnostics["anchor_offset_l%d" % index] = float(
            torch.linalg.vector_norm(
                normalized_offset.detach().float(),
                dim=-1,
            ).mean().cpu()
        )
    return diagnostics


def learning_rate_factor(epoch: int, epochs: int, warmup_epochs: int, minimum_factor: float) -> float:
    if epochs <= 0 or warmup_epochs < 0 or warmup_epochs >= epochs or not 0.0 <= minimum_factor <= 1.0:
        raise ValueError("invalid scheduler configuration")
    if warmup_epochs and epoch < warmup_epochs:
        return float(epoch + 1) / float(warmup_epochs)
    denominator = max(epochs - warmup_epochs - 1, 1)
    progress = min(max(float(epoch - warmup_epochs) / float(denominator), 0.0), 1.0)
    return minimum_factor + 0.5 * (1.0 - minimum_factor) * (1.0 + math.cos(math.pi * progress))


def correspondence_temperature_for_epoch(
    config: Mapping[str, object],
    epoch: int,
) -> float:
    """Cosine-anneal Gaussian matching temperature using one-indexed epochs."""
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    model = dict(config.get("model", {}))
    start = float(model.get("match_temperature", 0.08))
    end = float(model.get("match_temperature_end", start))
    anneal_epochs = int(model.get("match_temperature_anneal_epochs", 1))
    if start <= 0.0 or end <= 0.0 or anneal_epochs <= 0:
        raise ValueError("matching temperatures and anneal epochs must be positive")
    if anneal_epochs == 1:
        return end
    progress = min(max(float(epoch - 1) / float(anneal_epochs - 1), 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def appearance_weight_for_epoch(
    config: Mapping[str, object],
    epoch: int,
) -> float:
    """Cosine-anneal the fixed Gaussian appearance anchor contribution."""
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    model = dict(config.get("model", {}))
    start = float(model.get("appearance_weight", 0.0))
    end = float(model.get("appearance_weight_end", start))
    anneal_epochs = int(model.get("appearance_weight_anneal_epochs", 1))
    if (
        not 0.0 <= start <= 1.0
        or not 0.0 <= end <= 1.0
        or anneal_epochs <= 0
    ):
        raise ValueError(
            "appearance weights must lie in [0, 1] and anneal epochs be positive"
        )
    if anneal_epochs == 1:
        return end
    progress = min(
        max(float(epoch - 1) / float(anneal_epochs - 1), 0.0),
        1.0,
    )
    return end + 0.5 * (start - end) * (
        1.0 + math.cos(math.pi * progress)
    )


def feature_residual_weight_for_epoch(
    config: Mapping[str, object],
    epoch: int,
) -> float:
    """Cosine-ramp the learned correction to fixed Gaussian matching."""
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    model = dict(config.get("model", {}))
    start = float(model.get("feature_residual_weight", 0.0))
    end = float(model.get("feature_residual_weight_end", start))
    ramp_epochs = int(model.get("feature_residual_weight_anneal_epochs", 1))
    if start < 0.0 or end < 0.0 or ramp_epochs <= 0:
        raise ValueError(
            "feature residual weights must be nonnegative and ramp epochs positive"
        )
    if ramp_epochs == 1:
        return end
    progress = min(
        max(float(epoch - 1) / float(ramp_epochs - 1), 0.0),
        1.0,
    )
    return end + 0.5 * (start - end) * (
        1.0 + math.cos(math.pi * progress)
    )


def configure_model_for_epoch(
    model: nn.Module,
    config: Mapping[str, object],
    epoch: int,
) -> Optional[float]:
    """Apply deterministic epoch-dependent model settings for train/eval/resume."""
    setter = getattr(model, "set_correspondence_temperature", None)
    if setter is None:
        return None
    temperature = correspondence_temperature_for_epoch(config, epoch)
    setter(temperature)
    appearance_setter = getattr(
        model,
        "set_correspondence_appearance_weight",
        None,
    )
    if appearance_setter is not None:
        appearance_setter(appearance_weight_for_epoch(config, epoch))
    residual_setter = getattr(
        model,
        "set_correspondence_feature_residual_weight",
        None,
    )
    if residual_setter is not None:
        residual_setter(feature_residual_weight_for_epoch(config, epoch))
    return temperature


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
    "JacobianFoldingLoss",
    "atomic_torch_save",
    "appearance_weight_for_epoch",
    "bootstrap_mean_ci",
    "build_model",
    "build_objective",
    "configure_model_for_epoch",
    "config_architecture",
    "correspondence_temperature_for_epoch",
    "cuda_autocast",
    "finite_mean",
    "feature_residual_weight_for_epoch",
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
