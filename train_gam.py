"""Train GAM-SACB-Net on preprocessed HNTS-MRG24 longitudinal pairs."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.head_neck import HeadNeckRegistrationDataset, manifest_sha256
from experiment_utils import (
    RegistrationObjective,
    atomic_torch_save,
    build_model,
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


def _train_epoch(
    model: torch.nn.Module,
    objective: RegistrationObjective,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp: bool,
    gradient_clip: float,
    epoch: int,
    log_every: int,
) -> Dict[str, float]:
    model.train()
    totals = defaultdict(float)
    sample_count = 0
    successful_steps = 0
    start = time.perf_counter()
    for step, sample in enumerate(loader, start=1):
        moving = sample["moving"].to(device, non_blocking=True)
        fixed = sample["fixed"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with cuda_autocast(amp):
            output = model(moving, fixed, return_aux=True)
            terms = objective(output, moving, fixed)
        if not bool(torch.isfinite(terms["total"]).detach()):
            raise FloatingPointError("non-finite loss for patients %s" % list(sample["patient_id"]))
        scaler.scale(terms["total"]).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip))
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm)).detach()):
            if not amp:
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
    result["seconds"] = time.perf_counter() - start
    return result


@torch.inference_mode()
def _validate(
    model: torch.nn.Module,
    objective: RegistrationObjective,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> Dict[str, float]:
    model.eval()
    case_records = []
    for sample in loader:
        if int(sample["moving"].shape[0]) != 1:
            raise AssertionError("validation requires batch size one for per-patient metrics")
        moving = sample["moving"].to(device, non_blocking=True)
        fixed = sample["fixed"].to(device, non_blocking=True)
        with cuda_autocast(amp):
            warped, flow = model(moving, fixed, return_aux=False)
            ncc_before = -objective.ncc(fixed, moving)
            ncc_after = -objective.ncc(fixed, warped)
        moving_seg = sample["moving_seg"].to(device, non_blocking=True).float()
        warped_seg = warp_volume(moving_seg, flow.float(), mode="nearest").round().long()
        fixed_seg_np = sample["fixed_seg"][0, 0].numpy()
        warped_seg_np = warped_seg[0, 0].cpu().numpy()
        valid_labels = [index + 1 for index, flag in enumerate(sample["response_valid"][0].tolist()) if flag]
        spacing = tuple(float(value) for value in sample["spacing_dhw"][0].tolist())
        segmentation = evaluate_segmentation_pair(
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
                "mean_dice": float(segmentation["mean_dice"]),
                "mean_hd95": float(segmentation["mean_hd95"]),
                "mean_assd": float(segmentation["mean_assd"]),
                **jacobian,
                **{
                    "dice_label_%d" % label: float(value)
                    for label, value in segmentation["dice_per_class"].items()
                },
            }
        )
    names = sorted({name for record in case_records for name in record})
    return {name: finite_mean(record.get(name, float("nan")) for record in case_records) for name in names}


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
        "format_version": 1,
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": dict(config),
        "manifest_sha256": dict(manifests),
        "best_validation_ncc": float(best_validation_ncc),
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON experiment config")
    parser.add_argument("--data-root", required=True, help="preprocessed dataset root")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    if args.resume and args.baseline_checkpoint:
        raise ValueError("--resume and --baseline-checkpoint are mutually exclusive")
    config = load_json(args.config)
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
    data_config = dict(config.get("data", {}))
    loss_config = dict(config.get("loss", {}))
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
    if args.baseline_checkpoint:
        incompatible = model.load_sacb_checkpoint(args.baseline_checkpoint)
        print(
            "loaded baseline checkpoint: %d new keys missing, %d unexpected keys"
            % (len(incompatible.missing_keys), len(incompatible.unexpected_keys))
        )
    objective = RegistrationObjective(loss_config).to(device)
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
    scaler = make_grad_scaler(
        amp,
        initial_scale=float(optimization.get("amp_initial_scale", 1024.0)),
        growth_interval=int(optimization.get("amp_growth_interval", 2000)),
    )
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(optimization.get("allow_tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(optimization.get("allow_tf32", True))

    manifests = {
        "train": manifest_sha256(args.train_manifest),
        "validation": manifest_sha256(args.validation_manifest),
    }
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu")
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
    if validate_every <= 0 or checkpoint_every <= 0 or gradient_clip <= 0.0 or log_every <= 0:
        raise ValueError("validation/checkpoint/log intervals and gradient_clip must be positive")
    try:
        for epoch in range(start_epoch, epochs + 1):
            lr = float(optimizer.param_groups[0]["lr"])
            train_metrics = _train_epoch(
                model,
                objective,
                train_loader,
                optimizer,
                scaler,
                device,
                amp,
                gradient_clip,
                epoch,
                log_every,
            )
            validation_metrics: Dict[str, float] = {}
            if epoch % validate_every == 0 or epoch == epochs:
                validation_metrics = _validate(model, objective, validation_loader, device, amp)
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
            improved = np.isfinite(score) and score > best_validation_ncc
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
            print(
                "epoch %d complete lr=%.3e val_ncc=%s best=%.5f"
                % (
                    epoch,
                    lr,
                    "%.5f" % score if np.isfinite(score) else "not-run",
                    best_validation_ncc,
                )
            )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
