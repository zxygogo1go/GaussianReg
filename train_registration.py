"""Train a configured registration model on preprocessed HNTS-MRG24 pairs."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.head_neck import HeadNeckRegistrationDataset, manifest_sha256
from experiment_utils import (
    atomic_torch_save,
    build_model,
    build_objective,
    configure_model_for_epoch,
    config_architecture,
    cuda_autocast,
    finite_mean,
    learning_rate_factor,
    load_json,
    make_grad_scaler,
    output_diagnostics,
    resolve_device,
    save_json,
    seed_worker,
    set_reproducibility,
    to_json_safe,
    warp_volume,
)
from metrics import evaluate_segmentation_pair, jacobian_metrics


class _NullWriter:
    def add_scalar(self, *args, **kwargs):
        del args, kwargs

    def close(self):
        pass


def _make_writer(output_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("warning: tensorboard is unavailable; JSONL logging remains enabled")
        return _NullWriter()
    return SummaryWriter(log_dir=str(output_dir / "tensorboard"))


def _append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(to_json_safe(dict(record)), sort_keys=True, allow_nan=False) + "\n")


def _make_loader(
    dataset: HeadNeckRegistrationDataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(workers > 0),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def _augment_intensity(
    volume: torch.Tensor,
    config: Mapping[str, object],
) -> torch.Tensor:
    transform = _sample_intensity_transform(volume, config)
    return _apply_intensity_transform(volume, transform)


def _sample_intensity_transform(
    reference: torch.Tensor,
    config: Mapping[str, object],
) -> Dict[str, torch.Tensor]:
    probability = float(config.get("intensity_probability", 0.0))
    if probability <= 0.0 or float(torch.rand((), device=reference.device)) >= probability:
        return {}
    gamma_low, gamma_high = (float(value) for value in config.get("gamma_range", (1.0, 1.0)))
    scale_low, scale_high = (float(value) for value in config.get("scale_range", (1.0, 1.0)))
    shift_low, shift_high = (float(value) for value in config.get("shift_range", (0.0, 0.0)))
    noise_low, noise_high = (float(value) for value in config.get("noise_std_range", (0.0, 0.0)))
    if gamma_low <= 0.0 or gamma_high < gamma_low or scale_high < scale_low:
        raise ValueError("invalid intensity augmentation ranges")
    return {
        "gamma": gamma_low
        + (gamma_high - gamma_low) * torch.rand((), device=reference.device),
        "scale": scale_low
        + (scale_high - scale_low) * torch.rand((), device=reference.device),
        "shift": shift_low
        + (shift_high - shift_low) * torch.rand((), device=reference.device),
        "noise_std": noise_low
        + (noise_high - noise_low) * torch.rand((), device=reference.device),
    }


def _apply_intensity_transform(
    volume: torch.Tensor,
    transform: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if not transform:
        return volume
    gamma = transform["gamma"]
    scale = transform["scale"]
    shift = transform["shift"]
    noise_std = transform["noise_std"]
    augmented = volume.clamp(0.0, 1.0).pow(gamma) * scale + shift
    if float(noise_std) > 0.0:
        augmented = augmented + noise_std * torch.randn_like(augmented)
    return augmented.clamp(0.0, 1.0)


def _augment_pair(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    config: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not bool(config.get("enabled", False)):
        return moving, fixed
    if float(torch.rand((), device=moving.device)) < float(
        config.get("reverse_pair_probability", 0.0)
    ):
        moving, fixed = fixed, moving
    if float(torch.rand((), device=moving.device)) < float(
        config.get("shared_flip_probability", 0.0)
    ):
        moving = torch.flip(moving, dims=(-1,))
        fixed = torch.flip(fixed, dims=(-1,))
    intensity_pair_mode = str(
        config.get("intensity_pair_mode", "independent")
    ).strip().lower()
    if intensity_pair_mode not in {"independent", "shared"}:
        raise ValueError("augmentation.intensity_pair_mode must be independent or shared")
    if intensity_pair_mode == "shared":
        transform = _sample_intensity_transform(moving, config)
        return (
            _apply_intensity_transform(moving, transform),
            _apply_intensity_transform(fixed, transform),
        )
    return _augment_intensity(moving, config), _augment_intensity(fixed, config)


def _train_epoch(
    model: torch.nn.Module,
    objective: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    amp_cache_enabled: bool,
    augmentation: Mapping[str, object],
    gradient_clip: float,
    epoch: int,
    log_every: int,
) -> Dict[str, float]:
    model.train()
    totals = defaultdict(float)
    sample_count = 0
    successful_steps = 0
    first_step_pair_gradient_l1 = None
    start = time.perf_counter()
    for step, sample in enumerate(loader, start=1):
        moving = sample["moving"].to(device, non_blocking=True)
        fixed = sample["fixed"].to(device, non_blocking=True)
        moving, fixed = _augment_pair(moving, fixed, augmentation)
        optimizer.zero_grad(set_to_none=True)
        with cuda_autocast(
            amp,
            amp_dtype,
            cache_enabled=amp_cache_enabled,
        ):
            output = model(moving, fixed, return_aux=True)
            terms = objective(output, moving, fixed)
        if not bool(torch.isfinite(terms["total"]).detach()):
            raise FloatingPointError("non-finite loss for patients %s" % list(sample["patient_id"]))
        scaler.scale(terms["total"]).backward()
        scaler.unscale_(optimizer)
        if step == 1:
            pair_gradients = []
            correspondence = getattr(model, "correspondence", None)
            for level_index, matcher in enumerate(
                getattr(correspondence, "matchers", ())
            ):
                scorer = getattr(matcher, "pair_residual_score", None)
                if scorer is None:
                    continue
                gradient = scorer[-1].weight.grad
                if gradient is None:
                    raise RuntimeError(
                        "level %d pair residual scorer has no gradient; "
                        "disable the AMP weight cache"
                        % level_index
                    )
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError(
                        "level %d pair residual scorer gradient is non-finite"
                        % level_index
                    )
                gradient_l1 = float(
                    gradient.detach().float().abs().sum().cpu()
                )
                if gradient_l1 <= 0.0:
                    raise RuntimeError(
                        "level %d pair residual scorer gradient is zero"
                        % level_index
                    )
                pair_gradients.append(gradient_l1)
            if pair_gradients:
                first_step_pair_gradient_l1 = float(sum(pair_gradients))
                print(
                    "epoch %d pair_residual_grad_l1=%s"
                    % (
                        epoch,
                        "/".join("%.6g" % value for value in pair_gradients),
                    )
                )
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm)).detach()):
            if not amp or amp_dtype != "float16":
                raise FloatingPointError("non-finite gradient norm for patients %s" % list(sample["patient_id"]))
            previous_scale = float(scaler.get_scale())
            # ``unscale_`` has already recorded the non-finite gradients.
            # GradScaler.step therefore skips the optimizer update, and
            # update lowers the scale for the next sample.
            scaler.step(optimizer)
            scaler.update()
            new_scale = float(scaler.get_scale())
            count = int(moving.shape[0])
            sample_count += count
            for name, value in terms.items():
                totals[name] += float(value.detach().float().cpu()) * count
            for name, value in output_diagnostics(output).items():
                totals[name] += float(value) * count
            totals["skipped_amp_steps"] += count
            print(
                "epoch %d step %d/%d AMP overflow scale %.1f -> %.1f; optimizer step skipped"
                % (epoch, step, len(loader), previous_scale, new_scale)
            )
            if new_scale >= previous_scale or previous_scale <= 1.0:
                raise FloatingPointError(
                    "AMP could not recover non-finite gradients for patients %s"
                    % list(sample["patient_id"])
                )
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.step(optimizer)
        scaler.update()
        successful_steps += 1

        count = int(moving.shape[0])
        sample_count += count
        for name, value in terms.items():
            totals[name] += float(value.detach().float().cpu()) * count
        diagnostics = output_diagnostics(output)
        for name, value in diagnostics.items():
            totals[name] += float(value) * count
        totals["gradient_norm"] += float(torch.as_tensor(gradient_norm).detach().float().cpu()) * count
        if step % max(int(log_every), 1) == 0 or step == len(loader):
            print(
                "epoch %d step %d/%d total=%.5f sim=%.5f"
                % (
                    epoch,
                    step,
                    len(loader),
                    totals["total"] / sample_count,
                    totals["similarity"] / sample_count,
                )
            )
    if successful_steps == 0:
        raise FloatingPointError("all optimizer steps in the epoch were skipped")
    result = {name: value / max(sample_count, 1) for name, value in totals.items()}
    if first_step_pair_gradient_l1 is not None:
        result["pair_residual_gradient_l1_first_step"] = (
            first_step_pair_gradient_l1
        )
    result["seconds"] = time.perf_counter() - start
    return result


@torch.inference_mode()
def _validate(
    model: torch.nn.Module,
    objective: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    amp_cache_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    case_records = []
    for sample in loader:
        if int(sample["moving"].shape[0]) != 1:
            raise AssertionError("validation requires batch size one for per-patient metrics")
        moving = sample["moving"].to(device, non_blocking=True)
        fixed = sample["fixed"].to(device, non_blocking=True)
        with cuda_autocast(
            amp,
            amp_dtype,
            cache_enabled=amp_cache_enabled,
        ):
            warped, flow = model(moving, fixed, return_aux=False)
            ncc_before = -objective.ncc(fixed, moving)
            ncc_after = -objective.ncc(fixed, warped)
        moving_seg = sample["moving_seg"].to(device, non_blocking=True).float()
        warped_seg = warp_volume(moving_seg, flow.float(), mode="nearest").round().long()
        moving_seg_np = sample["moving_seg"][0, 0].numpy()
        fixed_seg_np = sample["fixed_seg"][0, 0].numpy()
        warped_seg_np = warped_seg[0, 0].cpu().numpy()
        valid_labels = [index + 1 for index, flag in enumerate(sample["response_valid"][0].tolist()) if flag]
        spacing = tuple(float(value) for value in sample["spacing_dhw"][0].tolist())
        spacing_tensor = flow.new_tensor(spacing).view(1, 3, 1, 1, 1)
        displacement_mm = torch.linalg.vector_norm(
            flow.float() * spacing_tensor,
            dim=1,
        )
        segmentation_before = evaluate_segmentation_pair(
            moving_seg_np,
            fixed_seg_np,
            labels=(1, 2),
            spacing_dhw=spacing,
            response_aware=True,
            valid_labels=valid_labels,
        )
        segmentation_after = evaluate_segmentation_pair(
            warped_seg_np,
            fixed_seg_np,
            labels=(1, 2),
            spacing_dhw=spacing,
            response_aware=True,
            valid_labels=valid_labels,
        )
        jacobian = jacobian_metrics(flow.float(), spacing_dhw=spacing)
        case_records.append(
            {
                "ncc_before": float(ncc_before.float().cpu()),
                "ncc_after": float(ncc_after.float().cpu()),
                "ncc_improvement": float(
                    (ncc_after - ncc_before).float().cpu()
                ),
                "dice_before": float(segmentation_before["mean_dice"]),
                "mean_dice": float(segmentation_after["mean_dice"]),
                "dice_improvement": float(
                    segmentation_after["mean_dice"]
                    - segmentation_before["mean_dice"]
                ),
                "mean_hd95": float(segmentation_after["mean_hd95"]),
                "mean_assd": float(segmentation_after["mean_assd"]),
                "mean_displacement_mm": float(displacement_mm.mean().cpu()),
                "p95_displacement_mm": float(
                    torch.quantile(displacement_mm, 0.95).cpu()
                ),
                **jacobian,
                **{
                    "dice_label_%d" % label: float(value)
                    for label, value in segmentation_after[
                        "dice_per_class"
                    ].items()
                },
            }
        )
    names = sorted({name for record in case_records for name in record})
    return {name: finite_mean(record.get(name, float("nan")) for record in case_records) for name in names}


def _collapse_warning(
    epoch: int,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
) -> str:
    """Describe identity-collapse, near-identity, and topology warning states."""
    entropy = float(train_metrics.get("match_entropy_l0", float("nan")))
    diagonal = float(
        train_metrics.get("diagonal_probability_l0", float("nan"))
    )
    displacement = float(
        train_metrics.get("transport_delta_l0_mm", float("nan"))
    )
    ncc_gain = float(
        validation_metrics.get("ncc_improvement", float("nan"))
    )
    p95 = float(
        validation_metrics.get("p95_displacement_mm", float("nan"))
    )
    folding = float(
        validation_metrics.get("negative_jacobian_ratio", float("nan"))
    )
    warnings = []
    if (
        epoch >= 10
        and np.isfinite(entropy)
        and np.isfinite(diagonal)
        and np.isfinite(displacement)
        and np.isfinite(ncc_gain)
        and entropy < 0.03
        and diagonal > 0.97
        and displacement < 0.10
        and ncc_gain < 0.02
    ):
        warnings.append(
            (
                "warning: coarse Gaussian correspondence is locked to the "
                "same-index identity path (entropy=%.4f diagonal=%.4f "
                "delta=%.4fmm val_ncc_gain=%.4f)"
                % (entropy, diagonal, displacement, ncc_gain)
            )
        )
    if (
        epoch >= 20
        and np.isfinite(ncc_gain)
        and np.isfinite(p95)
        and ncc_gain < 0.03
        and p95 < 2.0
    ):
        warnings.append(
            "warning: validation remains near identity "
            "(val_ncc_gain=%.4f p95=%.3fmm)" % (ncc_gain, p95)
        )
    if np.isfinite(folding) and folding > 0.02:
        warnings.append(
            "warning: negative Jacobian ratio %.5f exceeds 0.02" % folding
        )
    return "\n".join(warnings)


def _fail_fast_reason(
    epoch: int,
    history: Sequence[Mapping[str, Mapping[str, float]]],
    config: Mapping[str, object],
) -> str:
    """Return a reproducible failure reason for a clearly non-viable run."""
    monitoring = dict(config.get("monitoring", {}))
    if not bool(monitoring.get("fail_fast_enabled", False)) or not history:
        return ""
    current_validation = history[-1]["validation"]
    finite_names = (
        "ncc_after",
        "ncc_improvement",
        "negative_jacobian_ratio",
        "p95_displacement_mm",
    )
    for name in finite_names:
        value = float(current_validation.get(name, float("nan")))
        if not np.isfinite(value):
            return "non-finite validation metric: %s" % name
    folding = float(current_validation["negative_jacobian_ratio"])
    maximum_folding = float(
        monitoring.get("maximum_negative_jacobian_ratio", 0.02)
    )
    if folding > maximum_folding:
        return (
            "negative Jacobian ratio %.5f exceeds %.5f"
            % (folding, maximum_folding)
        )
    current_train = history[-1]["train"]
    anchor_offset_l2 = float(
        current_train.get("anchor_offset_l2", float("nan"))
    )
    maximum_anchor_offset_l2 = float(
        monitoring.get("maximum_anchor_offset_l2", float("inf"))
    )
    if (
        np.isfinite(anchor_offset_l2)
        and anchor_offset_l2 > maximum_anchor_offset_l2
    ):
        return (
            "fine Gaussian anchor offset %.5f exceeds %.5f"
            % (anchor_offset_l2, maximum_anchor_offset_l2)
        )

    patience = int(monitoring.get("fail_fast_patience", 3))
    start_epoch = int(monitoring.get("fail_fast_start_epoch", 5))
    if patience <= 0 or start_epoch <= 0:
        raise ValueError("fail-fast patience and start epoch must be positive")
    recent = list(history[-patience:])
    if epoch >= start_epoch and len(recent) == patience:
        ncc_values = np.asarray(
            [
                float(item["validation"].get("ncc_improvement", float("nan")))
                for item in recent
            ],
            dtype=np.float64,
        )
        dice_values = np.asarray(
            [
                float(item["validation"].get("dice_improvement", float("nan")))
                for item in recent
            ],
            dtype=np.float64,
        )
        ncc_floor = float(
            monitoring.get("recent_mean_ncc_floor", -0.02)
        )
        dice_floor = float(
            monitoring.get("recent_mean_dice_floor", -0.03)
        )
        if np.isfinite(ncc_values).all() and float(ncc_values.mean()) < ncc_floor:
            return (
                "recent %d-validation mean NCC improvement %.5f is below %.5f"
                % (patience, float(ncc_values.mean()), ncc_floor)
            )
        if np.isfinite(dice_values).all() and float(dice_values.mean()) < dice_floor:
            return (
                "recent %d-validation mean Dice improvement %.5f is below %.5f"
                % (patience, float(dice_values.mean()), dice_floor)
            )

        entropy_threshold = float(
            monitoring.get("uniform_entropy_threshold", 0.985)
        )
        row_maximum_threshold = float(
            monitoring.get("uniform_row_maximum_threshold", 0.04)
        )
        uniform = all(
            float(
                item["train"].get(
                    "support_entropy_l0",
                    float("nan"),
                )
            )
            > entropy_threshold
            and float(
                item["train"].get(
                    "row_max_probability_l0",
                    float("nan"),
                )
            )
            < row_maximum_threshold
            and float(
                item["validation"].get(
                    "ncc_improvement",
                    float("nan"),
                )
            )
            <= 0.0
            for item in recent
        )
        if uniform:
            return (
                "coarse correspondence remained uniform for %d validations"
                % patience
            )

    finite_gains = [
        float(item["validation"].get("ncc_improvement", float("nan")))
        for item in history
    ]
    finite_gains = [value for value in finite_gains if np.isfinite(value)]
    best_gain = max(finite_gains, default=-float("inf"))
    for threshold in monitoring.get("progress_thresholds", ()):
        threshold = dict(threshold)
        threshold_epoch = int(threshold["epoch"])
        minimum_gain = float(threshold["minimum_best_ncc_improvement"])
        if epoch == threshold_epoch and best_gain <= minimum_gain:
            return (
                "best NCC improvement %.5f did not exceed %.5f at epoch %d"
                % (best_gain, minimum_gain, epoch)
            )
    return ""


def _checkpoint(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: Mapping[str, object],
    manifests: Mapping[str, str],
    best_validation_ncc: float,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
) -> Dict[str, object]:
    return {
        "format_version": 3,
        "architecture_revision": getattr(
            model,
            "architecture_revision",
            "original",
        ),
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": dict(config),
        "manifest_sha256": dict(manifests),
        "best_validation_ncc": float(best_validation_ncc),
        "correspondence_temperature": train_metrics.get(
            "correspondence_temperature"
        ),
        "correspondence_appearance_weight": train_metrics.get(
            "correspondence_appearance_weight"
        ),
        "correspondence_feature_residual_weight": train_metrics.get(
            "correspondence_feature_residual_weight"
        ),
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
    }


def main(expected_architecture: str = "gaussian_native") -> None:
    description = (
        "Train the original SACB-Net baseline on preprocessed HNTS-MRG24 longitudinal pairs."
        if expected_architecture == "sacb"
        else "Train Gaussian-native diffeomorphic registration on HNTS-MRG24."
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="JSON experiment config")
    parser.add_argument("--data-root", required=True, help="preprocessed dataset root")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_json(args.config)
    architecture = config_architecture(config)
    if architecture != expected_architecture:
        raise ValueError(
            "this entry point requires model.architecture=%s, got %s"
            % (expected_architecture, architecture)
        )
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError("output path exists and is not a directory")
        if any(output_dir.iterdir()) and not args.resume:
            raise FileExistsError("output directory is not empty; choose a new directory or use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 2026))
    set_reproducibility(seed)
    device = resolve_device(args.device)
    optimization = dict(config.get("optimization", {}))
    augmentation = dict(config.get("augmentation", {}))
    data_config = dict(config.get("data", {}))
    shape = tuple(int(value) for value in data_config.get("shape_dhw", (128, 160, 160)))
    train_dataset = HeadNeckRegistrationDataset(
        args.train_manifest,
        args.data_root,
        expected_shape=shape,
        load_segmentations=False,
    )
    validation_dataset = HeadNeckRegistrationDataset(
        args.validation_manifest,
        args.data_root,
        expected_shape=shape,
        load_segmentations=True,
    )
    workers = int(optimization.get("workers", 4))
    batch_size = int(optimization.get("batch_size", 1))
    if workers < 0 or batch_size <= 0:
        raise ValueError("workers must be nonnegative and batch_size must be positive")
    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        workers=workers,
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    validation_loader = _make_loader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        workers=max(0, min(workers, 2)),
        seed=seed,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    architecture_revision = getattr(
        model,
        "architecture_revision",
        "original",
    )
    print(
        "architecture=%s revision=%s trainable_parameters=%d"
        % (architecture, architecture_revision, trainable_parameters)
    )
    objective = build_objective(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization.get("learning_rate", 1.0e-4)),
        weight_decay=float(optimization.get("weight_decay", 1.0e-5)),
    )
    epochs = int(optimization.get("epochs", 500))
    warmup_epochs = int(optimization.get("warmup_epochs", 5))
    minimum_lr_factor = float(optimization.get("minimum_lr_factor", 0.05))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: learning_rate_factor(epoch, epochs, warmup_epochs, minimum_lr_factor),
    )
    amp = bool(optimization.get("amp", True)) and device.type == "cuda"
    amp_dtype = str(optimization.get("amp_dtype", "bfloat16")).strip().lower()
    amp_cache_enabled = bool(
        optimization.get("amp_cache_enabled", False)
    )
    if amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("optimization.amp_dtype must be float16 or bfloat16")
    if (
        amp
        and amp_cache_enabled
        and getattr(
            getattr(model, "correspondence", None),
            "identity_calibration",
            False,
        )
        and not getattr(
            getattr(model, "correspondence", None),
            "calibration_gradient",
            True,
        )
    ):
        raise ValueError(
            "AMP weight caching is unsafe when no-grad identity "
            "calibration reuses the trainable matcher"
        )
    scaler = make_grad_scaler(
        amp and amp_dtype == "float16",
        initial_scale=float(optimization.get("amp_initial_scale", 1024.0)),
        growth_interval=int(optimization.get("amp_growth_interval", 2000)),
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(optimization.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(optimization.get("allow_tf32", True))
    print(
        "amp=%s amp_dtype=%s amp_cache_enabled=%s"
        % (amp, amp_dtype, amp_cache_enabled)
    )

    manifests = {
        "train": manifest_sha256(args.train_manifest),
        "validation": manifest_sha256(args.validation_manifest),
    }
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu")
        checkpoint_revision = resume_checkpoint.get(
            "architecture_revision",
            "legacy_v1",
        )
        if checkpoint_revision != architecture_revision:
            raise ValueError(
                "resume checkpoint revision %s does not match model revision %s"
                % (checkpoint_revision, architecture_revision)
            )
        if resume_checkpoint.get("config") != config:
            raise ValueError("resume checkpoint config does not match --config")
        if resume_checkpoint.get("manifest_sha256") != manifests:
            raise ValueError("resume checkpoint manifest hashes do not match current manifests")
    resolved = {
        **config,
        "runtime": {
            "data_root": str(Path(args.data_root).resolve()),
            "train_manifest": str(Path(args.train_manifest).resolve()),
            "validation_manifest": str(Path(args.validation_manifest).resolve()),
            "device": str(device),
            "architecture": architecture,
            "architecture_revision": architecture_revision,
            "trainable_parameters": int(trainable_parameters),
            "manifest_sha256": manifests,
        },
    }
    save_json(output_dir / "resolved_config.json", resolved)

    start_epoch = 1
    best_validation_ncc = -float("inf")
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        scaler.load_state_dict(resume_checkpoint.get("scaler", {}))
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_validation_ncc = float(resume_checkpoint.get("best_validation_ncc", -float("inf")))
        print("resumed from epoch %d" % (start_epoch - 1))

    writer = _make_writer(output_dir)
    log_path = output_dir / "metrics.jsonl"
    validate_every = int(optimization.get("validate_every", 1))
    checkpoint_every = int(optimization.get("checkpoint_every", 25))
    gradient_clip = float(optimization.get("gradient_clip", 5.0))
    log_every = int(optimization.get("log_every", 10))
    monitoring = dict(config.get("monitoring", {}))
    validation_history = []
    if validate_every <= 0 or checkpoint_every <= 0 or gradient_clip <= 0.0 or log_every <= 0:
        raise ValueError("validation/checkpoint/log intervals and gradient_clip must be positive")
    try:
        for epoch in range(start_epoch, epochs + 1):
            match_temperature = configure_model_for_epoch(
                model,
                config,
                epoch,
            )
            match_appearance_weight = getattr(
                getattr(model, "correspondence", None),
                "appearance_weight",
                None,
            )
            match_feature_residual_weight = getattr(
                getattr(model, "correspondence", None),
                "feature_residual_weight",
                None,
            )
            lr = float(optimizer.param_groups[0]["lr"])
            train_metrics = _train_epoch(
                model,
                objective,
                train_loader,
                optimizer,
                scaler,
                device,
                amp,
                amp_dtype,
                amp_cache_enabled,
                augmentation,
                gradient_clip,
                epoch,
                log_every,
            )
            if match_temperature is not None:
                train_metrics["correspondence_temperature"] = float(
                    match_temperature
                )
            if match_appearance_weight is not None:
                train_metrics["correspondence_appearance_weight"] = float(
                    match_appearance_weight
                )
            if match_feature_residual_weight is not None:
                train_metrics[
                    "correspondence_feature_residual_weight"
                ] = float(match_feature_residual_weight)
            validation_metrics: Dict[str, float] = {}
            if epoch % validate_every == 0 or epoch == epochs:
                validation_metrics = _validate(
                    model,
                    objective,
                    validation_loader,
                    device,
                    amp,
                    amp_dtype,
                    amp_cache_enabled,
                )
                validation_history.append(
                    {
                        "train": dict(train_metrics),
                        "validation": dict(validation_metrics),
                    }
                )
            scheduler.step()
            record = {
                "epoch": epoch,
                "learning_rate": lr,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            _append_jsonl(log_path, record)
            for name, value in train_metrics.items():
                writer.add_scalar("train/" + name, value, epoch)
            for name, value in validation_metrics.items():
                writer.add_scalar("validation/" + name, value, epoch)
            writer.add_scalar("optimization/learning_rate", lr, epoch)

            score = float(validation_metrics.get("ncc_after", -float("inf")))
            ncc_gain = float(
                validation_metrics.get(
                    "ncc_improvement",
                    float("nan"),
                )
            )
            folding = float(
                validation_metrics.get(
                    "negative_jacobian_ratio",
                    float("nan"),
                )
            )
            eligible = True
            if monitoring:
                eligible = (
                    np.isfinite(ncc_gain)
                    and ncc_gain
                    > float(
                        monitoring.get(
                            "best_checkpoint_minimum_ncc_improvement",
                            -float("inf"),
                        )
                    )
                    and np.isfinite(folding)
                    and folding
                    <= float(
                        monitoring.get(
                            "best_checkpoint_maximum_negative_jacobian_ratio",
                            float("inf"),
                        )
                    )
                )
            improved = (
                eligible
                and np.isfinite(score)
                and score > best_validation_ncc
            )
            if improved:
                best_validation_ncc = score
            state = _checkpoint(
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                manifests,
                best_validation_ncc,
                train_metrics,
                validation_metrics,
            )
            atomic_torch_save(state, output_dir / "latest.pt")
            if improved:
                atomic_torch_save(state, output_dir / "best_validation_ncc.pt")
            if epoch % checkpoint_every == 0 or epoch == epochs:
                atomic_torch_save(state, output_dir / ("epoch_%04d.pt" % epoch))
            validation_dice = float(
                validation_metrics.get("mean_dice", float("nan"))
            )
            validation_p95 = float(
                validation_metrics.get("p95_displacement_mm", float("nan"))
            )
            print(
                "epoch %d complete lr=%.3e match_temp=%s appearance=%s "
                "feature_residual=%s val_ncc=%s "
                "val_dice=%s val_p95_mm=%s support_h=%.4f/%.4f/%.4f "
                "raw_e=%.4f/%.4f/%.4f motion_e=%.4f/%.4f/%.4f "
                "diag0=%.4f residual0=%.3f delta0_mm=%.3f anchor2=%.3f "
                "ncc_gain=%.5f dice_gain=%.5f "
                "fold=%.6f best=%.5f"
                % (
                    epoch,
                    lr,
                    "%.4f" % match_temperature
                    if match_temperature is not None
                    else "n/a",
                    "%.4f" % match_appearance_weight
                    if match_appearance_weight is not None
                    else "n/a",
                    "%.4f" % match_feature_residual_weight
                    if match_feature_residual_weight is not None
                    else "n/a",
                    "%.5f" % score if np.isfinite(score) else "not-run",
                    "%.5f" % validation_dice
                    if np.isfinite(validation_dice)
                    else "not-run",
                    "%.3f" % validation_p95
                    if np.isfinite(validation_p95)
                    else "not-run",
                    float(
                        train_metrics.get(
                            "support_entropy_l0",
                            float("nan"),
                        )
                    ),
                    float(
                        train_metrics.get(
                            "support_entropy_l1",
                            float("nan"),
                        )
                    ),
                    float(
                        train_metrics.get(
                            "support_entropy_l2",
                            float("nan"),
                        )
                    ),
                    float(
                        train_metrics.get(
                            "match_evidence_l0",
                            float("nan"),
                        )
                    ),
                    float(
                        train_metrics.get(
                            "match_evidence_l1",
                            float("nan"),
                        )
                    ),
                    float(
                        train_metrics.get(
                            "match_evidence_l2",
                            float("nan"),
                        )
                    ),
                    float(
                        train_metrics.get(
                            "motion_evidence_l0",
                            float("nan"),
                        )
                    ),
                    float(
                        train_metrics.get(
                            "motion_evidence_l1",
                            float("nan"),
                        )
                    ),
                    float(
                        train_metrics.get(
                            "motion_evidence_l2",
                            float("nan"),
                        )
                    ),
                    float(
                        train_metrics.get(
                            "diagonal_probability_l0",
                            float("nan"),
                        )
                    ),
                    float(
                        train_metrics.get(
                            "feature_residual_logit_l0",
                            float("nan"),
                        )
                    ),
                    float(
                        train_metrics.get(
                            "transport_delta_l0_mm",
                            float("nan"),
                        )
                    ),
                    float(
                        train_metrics.get(
                            "anchor_offset_l2",
                            float("nan"),
                        )
                    ),
                    float(
                        validation_metrics.get(
                            "ncc_improvement",
                            float("nan"),
                        )
                    ),
                    float(
                        validation_metrics.get(
                            "dice_improvement",
                            float("nan"),
                        )
                    ),
                    float(
                        validation_metrics.get(
                            "negative_jacobian_ratio",
                            float("nan"),
                        )
                    ),
                    best_validation_ncc,
                )
            )
            warning = _collapse_warning(
                epoch,
                train_metrics,
                validation_metrics,
            )
            if warning:
                print(warning)
            fail_reason = _fail_fast_reason(
                epoch,
                validation_history,
                config,
            )
            if fail_reason:
                failed_state = dict(state)
                failed_state["failure_reason"] = fail_reason
                atomic_torch_save(
                    failed_state,
                    output_dir / ("failed_epoch_%04d.pt" % epoch),
                )
                print("training stopped by fail-fast monitor: %s" % fail_reason)
                break
    finally:
        writer.close()


if __name__ == "__main__":
    main()
