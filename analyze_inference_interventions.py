"""Patient-paired analysis for inference-only GAM-SACB-Net interventions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


METRICS = {
    "ncc_after": {
        "path": ("image", "ncc_after"),
        "direction": 1,
        "practical_threshold": 0.005,
    },
    "dice_after": {
        "path": ("segmentation_after", "mean_dice"),
        "direction": 1,
        "practical_threshold": 0.01,
    },
    "dice_label_1": {
        "path": ("segmentation_after", "dice_per_class", "1"),
        "direction": 1,
        "practical_threshold": 0.01,
    },
    "dice_label_2": {
        "path": ("segmentation_after", "dice_per_class", "2"),
        "direction": 1,
        "practical_threshold": 0.01,
    },
    "hd95_after_mm": {
        "path": ("segmentation_after", "mean_hd95"),
        "direction": -1,
        "practical_threshold": 1.5,
    },
    "assd_after_mm": {
        "path": ("segmentation_after", "mean_assd"),
        "direction": -1,
        "practical_threshold": 0.5,
    },
    "negative_jacobian_ratio": {
        "path": ("jacobian", "negative_jacobian_ratio"),
        "direction": -1,
        "practical_threshold": 0.001,
    },
    "below_safe_jacobian_ratio": {
        "path": ("jacobian", "below_safe_jacobian_ratio"),
        "direction": -1,
        "practical_threshold": 0.01,
    },
    "minimum_jacobian": {
        "path": ("jacobian", "minimum_jacobian"),
        "direction": 1,
        "practical_threshold": 0.1,
    },
    "mean_displacement_mm": {
        "path": ("deformation", "mean_displacement_mm"),
        "direction": None,
        "practical_threshold": 0.75,
    },
    "p95_displacement_mm": {
        "path": ("deformation", "p95_displacement_mm"),
        "direction": None,
        "practical_threshold": 1.5,
    },
}


def _read_evaluation(path_value: str) -> Tuple[Path, Mapping[str, object]]:
    path = Path(path_value)
    if path.is_dir():
        path = path / "evaluation.json"
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("patients"), list):
        raise ValueError("not an evaluation JSON: %s" % path)
    return path.resolve(), value


def _patient_map(evaluation: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    records = {}
    for record in evaluation["patients"]:
        patient_id = str(record["patient_id"])
        if patient_id in records:
            raise ValueError("duplicate patient_id %s" % patient_id)
        records[patient_id] = record
    return records


def _nested_value(record: Mapping[str, object], path: Sequence[str]) -> float:
    value: object = record
    for key in path:
        if not isinstance(value, Mapping):
            return float("nan")
        if key in value:
            value = value[key]
        elif key.isdigit() and int(key) in value:
            value = value[int(key)]
        else:
            return float("nan")
    if value is None:
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _bootstrap_mean(values: np.ndarray, samples: int, seed: int) -> Dict[str, object]:
    if values.ndim != 1 or not values.size:
        raise ValueError("bootstrap input must be a nonempty vector")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(samples), values.size))
    estimates = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95_low": float(np.percentile(estimates, 2.5)),
        "ci95_high": float(np.percentile(estimates, 97.5)),
        "n": int(values.size),
    }


def _paired_classification(benefit: Mapping[str, object], threshold: float) -> str:
    mean = float(benefit["mean"])
    low = float(benefit["ci95_low"])
    high = float(benefit["ci95_high"])
    if low > 0.0 and mean >= threshold:
        return "meaningful_benefit"
    if high < 0.0 and mean <= -threshold:
        return "meaningful_harm"
    if low >= -threshold and high <= threshold:
        return "practical_equivalence"
    return "inconclusive_or_mixed"


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_condition(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("condition must be NAME=EVALUATION_JSON")
    name, path = value.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise argparse.ArgumentTypeError("condition must be NAME=EVALUATION_JSON")
    return name, path


def _validate_compatible(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    candidate_path: Path,
) -> None:
    for field in ("architecture", "checkpoint_sha256", "manifest_sha256"):
        if reference.get(field) != candidate.get(field):
            raise ValueError("%s mismatch for %s" % (field, candidate_path))
    reference_patients = _patient_map(reference)
    candidate_patients = _patient_map(candidate)
    if set(reference_patients) != set(candidate_patients):
        raise ValueError("patient set mismatch for %s" % candidate_path)
    for patient_id, reference_record in reference_patients.items():
        candidate_record = candidate_patients[patient_id]
        if list(reference_record["valid_labels"]) != list(candidate_record["valid_labels"]):
            raise ValueError("valid_labels mismatch for patient %s" % patient_id)
        if not np.allclose(reference_record["spacing_dhw"], candidate_record["spacing_dhw"]):
            raise ValueError("spacing mismatch for patient %s" % patient_id)


def analyze(
    reference_path: str,
    conditions: Sequence[Tuple[str, str]],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, object]:
    resolved_reference_path, reference = _read_evaluation(reference_path)
    grouped = defaultdict(list)
    for name, path_value in conditions:
        path, evaluation = _read_evaluation(path_value)
        _validate_compatible(reference, evaluation, path)
        grouped[name].append((path, evaluation))
    if not grouped:
        raise ValueError("at least one condition is required")

    reference_patients = _patient_map(reference)
    comparison_rows = []
    per_patient_rows = []
    condition_metadata = {}
    metric_counter = 0
    for condition_name, replicates in sorted(grouped.items()):
        replicate_maps = [_patient_map(evaluation) for _, evaluation in replicates]
        condition_metadata[condition_name] = {
            "replicates": len(replicates),
            "paths": [str(path) for path, _ in replicates],
            "interventions": [evaluation.get("intervention") for _, evaluation in replicates],
        }
        for metric_name, definition in METRICS.items():
            path = definition["path"]
            direction = definition["direction"]
            threshold = float(definition["practical_threshold"])
            reference_values = []
            condition_values = []
            paired_patient_ids = []
            for patient_id in sorted(reference_patients, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value)):
                reference_value = _nested_value(reference_patients[patient_id], path)
                replicate_values = np.asarray(
                    [_nested_value(patient_map[patient_id], path) for patient_map in replicate_maps],
                    dtype=np.float64,
                )
                replicate_values = replicate_values[np.isfinite(replicate_values)]
                if not math.isfinite(reference_value) or not replicate_values.size:
                    continue
                condition_value = float(replicate_values.mean())
                raw_delta = condition_value - reference_value
                reference_values.append(reference_value)
                condition_values.append(condition_value)
                paired_patient_ids.append(patient_id)
                per_patient_rows.append(
                    {
                        "condition": condition_name,
                        "metric": metric_name,
                        "patient_id": patient_id,
                        "reference": reference_value,
                        "condition_value": condition_value,
                        "raw_delta": raw_delta,
                        "benefit": None if direction is None else int(direction) * raw_delta,
                        "replicate_min": float(replicate_values.min()),
                        "replicate_max": float(replicate_values.max()),
                        "replicates": int(replicate_values.size),
                    }
                )
            if not paired_patient_ids:
                continue
            reference_array = np.asarray(reference_values, dtype=np.float64)
            condition_array = np.asarray(condition_values, dtype=np.float64)
            raw_delta = condition_array - reference_array
            raw_stats = _bootstrap_mean(raw_delta, bootstrap_samples, seed + metric_counter)
            row = {
                "condition": condition_name,
                "metric": metric_name,
                "direction": direction,
                "practical_threshold": threshold,
                "replicates": len(replicates),
                "n": len(paired_patient_ids),
                "reference_mean": float(reference_array.mean()),
                "condition_mean": float(condition_array.mean()),
                "raw_delta": raw_stats,
                "condition_to_reference_ratio": (
                    float(condition_array.mean() / reference_array.mean())
                    if abs(float(reference_array.mean())) > 1.0e-12
                    else float("nan")
                ),
            }
            if direction is None:
                row["size_interpretation"] = (
                    "larger"
                    if raw_stats["ci95_low"] > 0.0
                    else "smaller"
                    if raw_stats["ci95_high"] < 0.0
                    else "uncertain"
                )
            else:
                benefit_values = int(direction) * raw_delta
                benefit_stats = _bootstrap_mean(benefit_values, bootstrap_samples, seed + 10000 + metric_counter)
                practical_wins = int((benefit_values > threshold).sum())
                practical_losses = int((benefit_values < -threshold).sum())
                practical_ties = int(benefit_values.size - practical_wins - practical_losses)
                row.update(
                    {
                        "benefit": benefit_stats,
                        "classification": _paired_classification(benefit_stats, threshold),
                        "raw_sign_wins": int((benefit_values > 0.0).sum()),
                        "raw_sign_ties": int((benefit_values == 0.0).sum()),
                        "raw_sign_losses": int((benefit_values < 0.0).sum()),
                        "practical_wins": practical_wins,
                        "practical_ties": practical_ties,
                        "practical_losses": practical_losses,
                    }
                )
            comparison_rows.append(row)
            metric_counter += 1

    conclusions = [
        {
            "condition": row["condition"],
            "metric": row["metric"],
            "classification": row.get("classification", row.get("size_interpretation")),
        }
        for row in comparison_rows
    ]
    return {
        "metadata": {
            "reference_path": str(resolved_reference_path),
            "reference_intervention": reference.get("intervention"),
            "architecture": reference.get("architecture"),
            "checkpoint": reference.get("checkpoint"),
            "checkpoint_sha256": reference.get("checkpoint_sha256"),
            "checkpoint_epoch": reference.get("checkpoint_epoch"),
            "manifest": reference.get("manifest"),
            "manifest_sha256": reference.get("manifest_sha256"),
            "bootstrap_samples": int(bootstrap_samples),
            "seed": int(seed),
            "note": "Exploratory inference interventions; confidence intervals are patient-paired.",
        },
        "metric_definitions": METRICS,
        "conditions": condition_metadata,
        "paired_comparisons": comparison_rows,
        "per_patient": per_patient_rows,
        "conclusions": conclusions,
    }


def _write_outputs(output_dir: Path, result: Mapping[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "paired_intervention_summary.json").open("w") as handle:
        json.dump(_json_safe(result), handle, indent=2, sort_keys=True, allow_nan=False)

    comparison_fields = [
        "condition",
        "metric",
        "direction",
        "practical_threshold",
        "replicates",
        "n",
        "reference_mean",
        "condition_mean",
        "raw_delta_mean",
        "raw_delta_median",
        "raw_delta_ci95_low",
        "raw_delta_ci95_high",
        "benefit_mean",
        "benefit_median",
        "benefit_ci95_low",
        "benefit_ci95_high",
        "classification",
        "practical_wins",
        "practical_ties",
        "practical_losses",
        "condition_to_reference_ratio",
        "size_interpretation",
    ]
    with (output_dir / "paired_intervention_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        for row in result["paired_comparisons"]:
            raw = row["raw_delta"]
            benefit = row.get("benefit", {})
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in comparison_fields},
                    "raw_delta_mean": raw["mean"],
                    "raw_delta_median": raw["median"],
                    "raw_delta_ci95_low": raw["ci95_low"],
                    "raw_delta_ci95_high": raw["ci95_high"],
                    "benefit_mean": benefit.get("mean", ""),
                    "benefit_median": benefit.get("median", ""),
                    "benefit_ci95_low": benefit.get("ci95_low", ""),
                    "benefit_ci95_high": benefit.get("ci95_high", ""),
                }
            )

    patient_fields = [
        "condition",
        "metric",
        "patient_id",
        "reference",
        "condition_value",
        "raw_delta",
        "benefit",
        "replicate_min",
        "replicate_max",
        "replicates",
    ]
    with (output_dir / "per_patient_deltas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=patient_fields)
        writer.writeheader()
        writer.writerows(result["per_patient"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument(
        "--condition",
        action="append",
        type=_parse_condition,
        required=True,
        help="NAME=EVALUATION_JSON; repeat the same NAME for replicated roll seeds",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    result = analyze(
        args.reference,
        args.condition,
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    _write_outputs(Path(args.output_dir), result)
    for row in result["paired_comparisons"]:
        if row["metric"] not in ("ncc_after", "dice_after", "negative_jacobian_ratio"):
            continue
        effect = row.get("benefit", row["raw_delta"])
        print(
            "%-24s %-24s effect=%+.6f CI=[%+.6f,%+.6f] %s"
            % (
                row["condition"],
                row["metric"],
                effect["mean"],
                effect["ci95_low"],
                effect["ci95_high"],
                row.get("classification", row.get("size_interpretation", "")),
            )
        )


if __name__ == "__main__":
    main()
