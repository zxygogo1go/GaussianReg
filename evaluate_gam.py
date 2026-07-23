"""Evaluate a GAM-SACB-Net checkpoint on patient-disjoint HNTS-MRG24 pairs."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.head_neck import HeadNeckRegistrationDataset, manifest_sha256
from experiment_utils import (
    bootstrap_mean_ci,
    build_model,
    cuda_autocast,
    load_json,
    resolve_device,
    save_json,
    set_reproducibility,
    to_json_safe,
    warp_volume,
)
from losses import NCC_vxm
from metrics import evaluate_segmentation_pair, jacobian_metrics


def _finite(values: Iterable[float]) -> Sequence[float]:
    return [float(value) for value in values if np.isfinite(value)]


def _case_metric(record: Mapping[str, object], path: Sequence[str]) -> float:
    value: object = record
    for key in path:
        value = value[key]
    return float(value)


def _summarize(records: Sequence[Mapping[str, object]], seed: int, bootstrap_samples: int):
    paths = {
        "ncc_before": ("image", "ncc_before"),
        "ncc_after": ("image", "ncc_after"),
        "dice_before": ("segmentation_before", "mean_dice"),
        "dice_after": ("segmentation_after", "mean_dice"),
        "hd95_after_mm": ("segmentation_after", "mean_hd95"),
        "assd_after_mm": ("segmentation_after", "mean_assd"),
        "negative_jacobian_ratio": ("jacobian", "negative_jacobian_ratio"),
        "below_safe_jacobian_ratio": ("jacobian", "below_safe_jacobian_ratio"),
        "minimum_jacobian": ("jacobian", "minimum_jacobian"),
        "mean_displacement_mm": ("deformation", "mean_displacement_mm"),
        "p95_displacement_mm": ("deformation", "p95_displacement_mm"),
        "inference_seconds": ("runtime", "inference_seconds"),
    }
    summary: Dict[str, object] = {"num_patients": len(records)}
    for offset, (name, path) in enumerate(paths.items()):
        values = _finite(_case_metric(record, path) for record in records)
        summary[name] = bootstrap_mean_ci(values, samples=bootstrap_samples, seed=seed + offset)
    ncc_improvements = _finite(
        _case_metric(record, ("image", "ncc_after")) - _case_metric(record, ("image", "ncc_before"))
        for record in records
    )
    dice_improvements = _finite(
        _case_metric(record, ("segmentation_after", "mean_dice"))
        - _case_metric(record, ("segmentation_before", "mean_dice"))
        for record in records
    )
    summary["ncc_improvement"] = bootstrap_mean_ci(
        ncc_improvements, samples=bootstrap_samples, seed=seed + 100
    )
    summary["dice_improvement"] = bootstrap_mean_ci(
        dice_improvements, samples=bootstrap_samples, seed=seed + 101
    )
    for label in (1, 2):
        values = []
        for record in records:
            per_class = record["segmentation_after"]["dice_per_class"]
            value = per_class.get(label, per_class.get(str(label)))
            if value is not None and np.isfinite(value):
                values.append(float(value))
        summary["dice_label_%d" % label] = bootstrap_mean_ci(
            values, samples=bootstrap_samples, seed=seed + 200 + label
        )
    return summary


def _write_csv(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "patient_id",
        "valid_labels",
        "ncc_before",
        "ncc_after",
        "mean_dice_before",
        "mean_dice_after",
        "dice_label_1_before",
        "dice_label_1_after",
        "dice_label_2_before",
        "dice_label_2_after",
        "mean_hd95_after_mm",
        "mean_assd_after_mm",
        "negative_jacobian_ratio",
        "below_safe_jacobian_ratio",
        "minimum_jacobian",
        "mean_jacobian",
        "mean_displacement_mm",
        "p95_displacement_mm",
        "inference_seconds",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            before = record["segmentation_before"]
            after = record["segmentation_after"]
            before_dice = before["dice_per_class"]
            after_dice = after["dice_per_class"]
            writer.writerow(
                {
                    "patient_id": record["patient_id"],
                    "valid_labels": ";".join(str(value) for value in record["valid_labels"]),
                    "ncc_before": record["image"]["ncc_before"],
                    "ncc_after": record["image"]["ncc_after"],
                    "mean_dice_before": before["mean_dice"],
                    "mean_dice_after": after["mean_dice"],
                    "dice_label_1_before": before_dice.get(1, before_dice.get("1", "")),
                    "dice_label_1_after": after_dice.get(1, after_dice.get("1", "")),
                    "dice_label_2_before": before_dice.get(2, before_dice.get("2", "")),
                    "dice_label_2_after": after_dice.get(2, after_dice.get("2", "")),
                    "mean_hd95_after_mm": after["mean_hd95"],
                    "mean_assd_after_mm": after["mean_assd"],
                    **record["jacobian"],
                    **record["deformation"],
                    "inference_seconds": record["runtime"]["inference_seconds"],
                }
            )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None, help="required only for a raw state_dict checkpoint")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    if args.num_workers < 0 or args.bootstrap_samples <= 0:
        raise ValueError("num-workers must be nonnegative and bootstrap-samples must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError("output path exists and is not a directory")
        if any(output_dir.iterdir()):
            raise FileExistsError("output directory is not empty; choose a new result directory")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    config = load_json(args.config) if args.config else checkpoint_config
    if not isinstance(config, dict):
        raise ValueError("checkpoint has no config; provide --config")
    seed = int(config.get("seed", 2026))
    set_reproducibility(seed)
    device = resolve_device(args.device)
    shape = tuple(int(value) for value in dict(config.get("data", {})).get("shape_dhw", (128, 160, 160)))
    dataset = HeadNeckRegistrationDataset(
        args.manifest,
        args.data_root,
        expected_shape=shape,
        load_segmentations=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=bool(args.num_workers > 0),
    )
    model = build_model(config)
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    ncc = NCC_vxm(win=[int(dict(config.get("loss", {})).get("ncc_window", 9))] * 3).to(device)
    amp = bool(dict(config.get("optimization", {})).get("amp", True)) and device.type == "cuda"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    if args.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for sample in loader:
        patient_id = str(sample["patient_id"][0])
        moving = sample["moving"].to(device, non_blocking=True)
        fixed = sample["fixed"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with cuda_autocast(amp):
            warped, flow = model(moving, fixed, return_aux=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - start
        if not bool(torch.isfinite(flow).all() and torch.isfinite(warped).all()):
            raise FloatingPointError("non-finite prediction for patient %s" % patient_id)

        moving_seg_device = sample["moving_seg"].to(device, non_blocking=True).float()
        warped_seg = warp_volume(moving_seg_device, flow.float(), mode="nearest").round().long()
        moving_seg = sample["moving_seg"][0, 0].numpy()
        fixed_seg = sample["fixed_seg"][0, 0].numpy()
        warped_seg_np = warped_seg[0, 0].cpu().numpy().astype(np.int16, copy=False)
        valid_labels = [index + 1 for index, flag in enumerate(sample["response_valid"][0].tolist()) if flag]
        spacing = tuple(float(value) for value in sample["spacing_dhw"][0].tolist())
        before = evaluate_segmentation_pair(
            moving_seg,
            fixed_seg,
            labels=(1, 2),
            spacing_dhw=spacing,
            response_aware=True,
            valid_labels=valid_labels,
        )
        after = evaluate_segmentation_pair(
            warped_seg_np,
            fixed_seg,
            labels=(1, 2),
            spacing_dhw=spacing,
            response_aware=True,
            valid_labels=valid_labels,
        )
        flow_float = flow.float()
        jacobian = jacobian_metrics(flow_float, spacing_dhw=spacing)
        spacing_tensor = flow_float.new_tensor(spacing).view(1, 3, 1, 1, 1)
        magnitude_mm = torch.linalg.vector_norm(flow_float * spacing_tensor, dim=1)
        deformation = {
            "mean_displacement_mm": float(magnitude_mm.mean().cpu()),
            "p95_displacement_mm": float(torch.quantile(magnitude_mm.reshape(-1), 0.95).cpu()),
        }
        runtime = {
            "inference_seconds": float(inference_seconds),
            "peak_gpu_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)) if device.type == "cuda" else 0.0
            ),
        }
        record = {
            "patient_id": patient_id,
            "valid_labels": valid_labels,
            "spacing_dhw": list(spacing),
            "image": {
                "ncc_before": float((-ncc(fixed, moving)).float().cpu()),
                "ncc_after": float((-ncc(fixed, warped)).float().cpu()),
            },
            "segmentation_before": before,
            "segmentation_after": after,
            "jacobian": jacobian,
            "deformation": deformation,
            "runtime": runtime,
        }
        records.append(record)
        if args.save_predictions:
            np.save(str(prediction_dir / (patient_id + "_flow_dhw.npy")), flow_float[0].cpu().numpy(), allow_pickle=False)
            np.save(str(prediction_dir / (patient_id + "_warped_seg.npy")), warped_seg_np, allow_pickle=False)
            np.save(str(prediction_dir / (patient_id + "_warped_image.npy")), warped[0, 0].float().cpu().numpy(), allow_pickle=False)
        print(
            "%s ncc %.4f -> %.4f dice %.4f -> %.4f fold %.6f"
            % (
                patient_id,
                record["image"]["ncc_before"],
                record["image"]["ncc_after"],
                before["mean_dice"],
                after["mean_dice"],
                jacobian["negative_jacobian_ratio"],
            )
        )

    summary = _summarize(records, seed=seed, bootstrap_samples=int(args.bootstrap_samples))
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest_sha256(args.manifest),
        "summary": summary,
        "patients": records,
        "notes": {
            "flow_convention": "fixed-grid to moving-image sampling displacement in DHW voxel units",
            "response_aware": "only labels present in both original moving and fixed masks are eligible",
            "hd95_assd_units": "millimetres",
            "tre": "not reported because HNTS-MRG24 provides no landmark annotations",
        },
    }
    save_json(output_dir / "evaluation.json", result)
    _write_csv(output_dir / "per_patient_metrics.csv", records)
    print(json.dumps(to_json_safe(summary), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
