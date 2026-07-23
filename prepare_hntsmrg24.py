"""Prepare HNTS-MRG24 longitudinal T2 pairs for GAM-SACB-Net.

Raw preRT is rigid/affine prealigned to midRT. The challenge-provided
deformably registered preRT volume is deliberately excluded from model inputs
and may only be used in a separate QA analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class HNTSCase:
    patient_id: str
    pre_image: str
    pre_mask: str
    mid_image: str
    mid_mask: str
    official_registered_pre_image: str
    official_registered_pre_mask: str


def discover_cases(source_root: str) -> List[HNTSCase]:
    root = Path(source_root)
    if not root.is_dir():
        raise FileNotFoundError("source root does not exist: %s" % root)
    directories = [path for path in root.iterdir() if path.is_dir()]
    try:
        directories.sort(key=lambda path: int(path.name))
    except ValueError as exc:
        raise ValueError("HNTS patient directories must use numeric IDs") from exc
    cases = []
    for directory in directories:
        patient = directory.name
        paths = {
            "pre_image": directory / "preRT" / (patient + "_preRT_T2.nii.gz"),
            "pre_mask": directory / "preRT" / (patient + "_preRT_mask.nii.gz"),
            "mid_image": directory / "midRT" / (patient + "_midRT_T2.nii.gz"),
            "mid_mask": directory / "midRT" / (patient + "_midRT_mask.nii.gz"),
            "official_registered_pre_image": directory / "midRT" / (patient + "_preRT_T2_registered.nii.gz"),
            "official_registered_pre_mask": directory / "midRT" / (patient + "_preRT_mask_registered.nii.gz"),
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing HNTS files for patient %s: %s" % (patient, missing))
        cases.append(HNTSCase(patient_id=patient, **{key: str(value) for key, value in paths.items()}))
    if not cases:
        raise ValueError("no HNTS cases found")
    return cases


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError("SimpleITK is required; install requirements.txt in the server environment") from exc
    return sitk


def _validate_image_mask_geometry(image, mask, patient_id: str, stage: str) -> None:
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError("%s %s image/mask must be 3D" % (patient_id, stage))
    if image.GetSize() != mask.GetSize():
        raise ValueError("%s %s image/mask size mismatch" % (patient_id, stage))
    for image_value, mask_value in zip(image.GetSpacing(), mask.GetSpacing()):
        if abs(float(image_value) - float(mask_value)) > 1.0e-5:
            raise ValueError("%s %s image/mask spacing mismatch" % (patient_id, stage))
    if not np.allclose(image.GetDirection(), mask.GetDirection(), atol=1.0e-5, rtol=0.0):
        raise ValueError("%s %s image/mask direction mismatch" % (patient_id, stage))
    if not np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=1.0e-4, rtol=0.0):
        raise ValueError("%s %s image/mask origin mismatch" % (patient_id, stage))


def _validate_mask_labels(mask_array: np.ndarray, patient_id: str) -> np.ndarray:
    rounded = np.rint(mask_array)
    if not np.allclose(mask_array, rounded, atol=1.0e-5, rtol=0.0):
        raise ValueError("non-integer mask values for patient %s" % patient_id)
    labels = set(int(value) for value in np.unique(rounded))
    if not labels.issubset({0, 1, 2}):
        raise ValueError("mask labels outside {0,1,2} for patient %s: %s" % (patient_id, labels))
    return rounded.astype(np.int16)


def _registration_method(iterations: int, seed: int, sampling: float, learning_rate: float):
    sitk = _sitk()
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(float(sampling), int(seed))
    method.SetInterpolator(sitk.sitkLinear)
    method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=float(learning_rate),
        minStep=1.0e-4,
        numberOfIterations=int(iterations),
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1.0e-8,
    )
    method.SetOptimizerScalesFromPhysicalShift()
    method.SetShrinkFactorsPerLevel([4, 2, 1])
    method.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return method


def _as_euler3d(transform):
    """Recover the concrete Euler3D API from a generic SimpleITK transform.

    SimpleITK 2.3 may expose ``CenteredTransformInitializer`` results as the
    base ``Transform`` wrapper even though the underlying transform is
    Euler3D. Newer releases preserve the concrete Python type. Explicit
    down-casting keeps both versions compatible and makes ``GetCenter`` /
    ``GetMatrix`` / ``GetTranslation`` available.
    """
    sitk = _sitk()
    if transform.GetDimension() != 3:
        raise ValueError("rigid initializer must return a 3D transform")
    try:
        return sitk.Euler3DTransform(transform)
    except RuntimeError as exc:
        raise TypeError(
            "rigid initializer returned an incompatible transform: %s"
            % transform.GetName()
        ) from exc


def _rigid_affine_prealign(case: HNTSCase, seed: int, iterations: int):
    sitk = _sitk()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    fixed = sitk.Cast(sitk.ReadImage(case.mid_image), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.ReadImage(case.pre_image), sitk.sitkFloat32)
    moving_mask = sitk.ReadImage(case.pre_mask)
    fixed_mask = sitk.ReadImage(case.mid_mask)
    _validate_image_mask_geometry(moving, moving_mask, case.patient_id, "preRT")
    _validate_image_mask_geometry(fixed, fixed_mask, case.patient_id, "midRT")

    fixed_metric = sitk.Normalize(fixed)
    moving_metric = sitk.Normalize(moving)
    sampling = min(1.0, max(0.02, 10000.0 / float(fixed.GetNumberOfPixels())))
    rigid = _as_euler3d(
        sitk.CenteredTransformInitializer(
            fixed_metric,
            moving_metric,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    )
    rigid_method = _registration_method(iterations, seed, sampling, learning_rate=2.0)
    rigid_method.SetInitialTransform(rigid, inPlace=True)
    rigid_before = float(rigid_method.MetricEvaluate(fixed_metric, moving_metric))
    rigid_method.Execute(fixed_metric, moving_metric)
    rigid_after = float(rigid_method.MetricEvaluate(fixed_metric, moving_metric))
    if not np.isfinite([rigid_before, rigid_after]).all() or rigid_after > rigid_before + 1.0e-4:
        raise ValueError("rigid MI failed QA for patient %s" % case.patient_id)

    affine = sitk.AffineTransform(3)
    affine.SetCenter(rigid.GetCenter())
    affine.SetMatrix(rigid.GetMatrix())
    affine.SetTranslation(rigid.GetTranslation())
    affine_method = _registration_method(iterations, seed + 1, sampling, learning_rate=0.1)
    affine_method.SetInitialTransform(affine, inPlace=True)
    affine_before = float(affine_method.MetricEvaluate(fixed_metric, moving_metric))
    affine_method.Execute(fixed_metric, moving_metric)
    affine_after = float(affine_method.MetricEvaluate(fixed_metric, moving_metric))
    matrix = np.asarray(affine.GetMatrix(), dtype=np.float64).reshape(3, 3)
    translation = np.asarray(affine.GetTranslation(), dtype=np.float64)
    determinant = float(np.linalg.det(matrix))
    reasons = []
    if not np.isfinite([affine_before, affine_after, determinant]).all():
        reasons.append("non-finite affine result")
    if affine_after > affine_before + 1.0e-4:
        reasons.append("affine worsened full-volume MI")
    if not 0.5 <= determinant <= 2.0:
        reasons.append("affine determinant outside [0.5,2.0]")
    if float(np.linalg.norm(translation)) > 250.0:
        reasons.append("affine translation exceeded 250 mm")
    selected = rigid if reasons else affine
    stage = "rigid" if reasons else "affine"
    registered = sitk.Resample(moving, fixed, selected, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    registered_mask = sitk.Resample(moving_mask, fixed, selected, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    record = {
        "selected_stage": stage,
        "fallback_reasons": reasons,
        "sampling_percentage": sampling,
        "rigid_metric_before": rigid_before,
        "rigid_metric_after": rigid_after,
        "affine_metric_before": affine_before,
        "affine_metric_after": affine_after,
        "affine_matrix_lps": matrix.tolist(),
        "affine_translation_lps": translation.tolist(),
        "affine_determinant": determinant,
    }
    return registered, registered_mask, fixed, fixed_mask, record


def _target_reference(fixed, spacing_dhw: Sequence[float], shape_dhw: Sequence[int]):
    sitk = _sitk()
    spacing_xyz = tuple(float(v) for v in reversed(spacing_dhw))
    size_xyz = tuple(int(v) for v in reversed(shape_dhw))
    direction = np.asarray(fixed.GetDirection(), dtype=np.float64).reshape(3, 3)
    fixed_center_index = tuple((float(size) - 1.0) / 2.0 for size in fixed.GetSize())
    fixed_center = np.asarray(fixed.TransformContinuousIndexToPhysicalPoint(fixed_center_index))
    target_half_extent = (np.asarray(size_xyz, dtype=np.float64) - 1.0) * np.asarray(spacing_xyz) / 2.0
    origin = fixed_center - direction @ target_half_extent
    reference = sitk.Image(size_xyz, sitk.sitkFloat32)
    reference.SetSpacing(spacing_xyz)
    reference.SetDirection(tuple(float(v) for v in direction.reshape(-1)))
    reference.SetOrigin(tuple(float(v) for v in origin))
    return reference


def _bbox_fits(mask, reference, tolerance: float = 0.51) -> bool:
    sitk = _sitk()
    array = sitk.GetArrayViewFromImage(mask)
    foreground = np.argwhere(array > 0)
    if foreground.size == 0:
        return True
    lower_zyx = foreground.min(axis=0)
    upper_zyx = foreground.max(axis=0)
    for z in (int(lower_zyx[0]), int(upper_zyx[0])):
        for y in (int(lower_zyx[1]), int(upper_zyx[1])):
            for x in (int(lower_zyx[2]), int(upper_zyx[2])):
                physical = mask.TransformIndexToPhysicalPoint((x, y, z))
                index = reference.TransformPhysicalPointToContinuousIndex(physical)
                if any(value < -tolerance or value > size - 1.0 + tolerance for value, size in zip(index, reference.GetSize())):
                    return False
    return True


def robust_normalize(volume: np.ndarray, lower: float = 0.5, upper: float = 99.5):
    array = np.asarray(volume, dtype=np.float32)
    samples = array[np.abs(array) > 1.0e-6]
    if samples.size < 100:
        samples = array.reshape(-1)
    low, high = np.percentile(samples, [float(lower), float(upper)])
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("degenerate MRI intensity range")
    normalized = np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)
    return normalized, {"lower_value": float(low), "upper_value": float(high), "percentiles": [float(lower), float(upper)]}


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(str(temporary), str(path))


def preprocess_case(
    case: HNTSCase,
    output_root: str,
    spacing_dhw: Sequence[float],
    shape_dhw: Sequence[int],
    seed: int,
    registration_iterations: int,
    overwrite: bool,
) -> Dict[str, object]:
    sitk = _sitk()
    root = Path(output_root)
    metadata_path = root / "metadata" / (case.patient_id + ".json")
    if metadata_path.is_file() and not overwrite:
        with metadata_path.open("r") as handle:
            cached = json.load(handle)
        if tuple(cached.get("spacing_dhw", ())) != tuple(float(v) for v in spacing_dhw):
            raise ValueError("cached spacing differs for patient %s; rerun with --overwrite" % case.patient_id)
        if tuple(cached.get("shape_dhw", ())) != tuple(int(v) for v in shape_dhw):
            raise ValueError("cached shape differs for patient %s; rerun with --overwrite" % case.patient_id)
        missing = [str(root / cached[field]) for field in ("moving", "fixed", "moving_seg", "fixed_seg") if not (root / cached[field]).is_file()]
        if missing:
            raise FileNotFoundError("cached metadata references missing arrays: %s" % missing)
        return cached

    moving, moving_mask, fixed, fixed_mask, alignment = _rigid_affine_prealign(
        case,
        seed=seed + int(case.patient_id),
        iterations=registration_iterations,
    )
    reference = _target_reference(fixed, spacing_dhw, shape_dhw)
    if not _bbox_fits(moving_mask, reference) or not _bbox_fits(fixed_mask, reference):
        raise ValueError("tumor bounding box falls outside target ROI for patient %s" % case.patient_id)
    moving_resampled = sitk.Resample(moving, reference, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    fixed_resampled = sitk.Resample(fixed, reference, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    moving_mask_resampled = sitk.Resample(moving_mask, reference, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    fixed_mask_resampled = sitk.Resample(fixed_mask, reference, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)

    moving_array, moving_norm = robust_normalize(sitk.GetArrayFromImage(moving_resampled))
    fixed_array, fixed_norm = robust_normalize(sitk.GetArrayFromImage(fixed_resampled))
    moving_seg = _validate_mask_labels(sitk.GetArrayFromImage(moving_mask_resampled), case.patient_id)
    fixed_seg = _validate_mask_labels(sitk.GetArrayFromImage(fixed_mask_resampled), case.patient_id)
    relative = {
        "moving": "moving/%s.npy" % case.patient_id,
        "fixed": "fixed/%s.npy" % case.patient_id,
        "moving_seg": "moving_seg/%s.npy" % case.patient_id,
        "fixed_seg": "fixed_seg/%s.npy" % case.patient_id,
    }
    _save_array(root / relative["moving"], moving_array)
    _save_array(root / relative["fixed"], fixed_array)
    _save_array(root / relative["moving_seg"], moving_seg)
    _save_array(root / relative["fixed_seg"], fixed_seg)
    record: Dict[str, object] = {
        "patient_id": case.patient_id,
        **relative,
        "spacing_dhw": [float(v) for v in spacing_dhw],
        "shape_dhw": [int(v) for v in shape_dhw],
        "direction_lps": list(reference.GetDirection()),
        "origin_lps": list(reference.GetOrigin()),
        "prealignment": alignment,
        "moving_normalization": moving_norm,
        "fixed_normalization": fixed_norm,
        "moving_labels": [int(v) for v in np.unique(moving_seg)],
        "fixed_labels": [int(v) for v in np.unique(fixed_seg)],
        "source": asdict(case),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    with metadata_temporary.open("w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    os.replace(str(metadata_temporary), str(metadata_path))
    return record


def _stratified_split(records: Sequence[Dict[str, object]], seed: int):
    groups: Dict[Tuple[bool, bool], List[Dict[str, object]]] = {}
    for record in records:
        labels = set(int(v) for v in record["fixed_labels"])
        groups.setdefault((1 in labels, 2 in labels), []).append(record)
    rng = np.random.default_rng(int(seed))
    splits = {"train": [], "validation": [], "test": []}
    for group in groups.values():
        order = rng.permutation(len(group)).tolist()
        shuffled = [group[index] for index in order]
        n_test = max(1, int(round(0.10 * len(shuffled)))) if len(shuffled) >= 3 else 0
        n_validation = max(1, int(round(0.10 * len(shuffled)))) if len(shuffled) - n_test >= 2 else 0
        splits["test"].extend(shuffled[:n_test])
        splits["validation"].extend(shuffled[n_test:n_test + n_validation])
        splits["train"].extend(shuffled[n_test + n_validation:])
    for split in splits.values():
        split.sort(key=lambda record: int(record["patient_id"]))
    return splits


def _write_manifest(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = [
        "patient_id",
        "moving",
        "fixed",
        "moving_seg",
        "fixed_seg",
        "spacing_d",
        "spacing_h",
        "spacing_w",
    ]
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            spacing = record["spacing_dhw"]
            writer.writerow(
                {
                    "patient_id": record["patient_id"],
                    "moving": record["moving"],
                    "fixed": record["fixed"],
                    "moving_seg": record["moving_seg"],
                    "fixed_seg": record["fixed_seg"],
                    "spacing_d": spacing[0],
                    "spacing_h": spacing[1],
                    "spacing_w": spacing[2],
                }
            )
    os.replace(str(temporary), str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-spacing", nargs=3, type=float, default=(1.5, 1.5, 1.5))
    parser.add_argument("--target-shape", nargs=3, type=int, default=(128, 160, 160))
    parser.add_argument("--registration-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    cases = discover_cases(args.source_root)
    records = []
    failures = []
    arguments = (
        args.output_root,
        tuple(args.target_spacing),
        tuple(args.target_shape),
        args.seed,
        args.registration_iterations,
        args.overwrite,
    )
    if args.num_workers <= 1:
        for case in cases:
            try:
                records.append(preprocess_case(case, *arguments))
            except Exception as exc:
                failures.append({"patient_id": case.patient_id, "error": repr(exc)})
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(preprocess_case, case, *arguments): case for case in cases}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    records.append(future.result())
                except Exception as exc:
                    failures.append({"patient_id": case.patient_id, "error": repr(exc)})
    records.sort(key=lambda record: int(record["patient_id"]))
    output_root = Path(args.output_root)
    splits = _stratified_split(records, args.seed) if records else {"train": [], "validation": [], "test": []}
    for name, split_records in splits.items():
        _write_manifest(output_root / "manifests" / (name + ".csv"), split_records)
    summary = {
        "source_root": str(Path(args.source_root).resolve()),
        "output_root": str(output_root.resolve()),
        "num_discovered": len(cases),
        "num_successful": len(records),
        "num_failed": len(failures),
        "split_sizes": {name: len(value) for name, value in splits.items()},
        "failures": failures,
        "target_spacing_dhw": list(args.target_spacing),
        "target_shape_dhw": list(args.target_shape),
        "seed": args.seed,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "dataset_summary.json"
    summary_temporary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with summary_temporary.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    os.replace(str(summary_temporary), str(summary_path))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures and not args.allow_failures:
        raise SystemExit("preprocessing had QA failures; inspect dataset_summary.json")


if __name__ == "__main__":
    main()
