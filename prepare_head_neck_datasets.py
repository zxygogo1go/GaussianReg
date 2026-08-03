"""Prepare HaN-Seg, Head-Neck-CBCT-CT, or SegRap2023 for registration.

HaN-Seg and SegRap2023 are treated as patient-disjoint inter-patient CT
registration cohorts.  Each scan is geometry-centred, resampled, and optionally
rigid/affine aligned to one training-only atlas before deterministic pairs are
created.  Head-Neck-CBCT-CT is treated as paired CBCT-to-CT registration; its
provided voxel correspondence is retained and inconsistent NIfTI origins are
deliberately ignored.

Multi-organ segmentations are saved as independent binary channels.  This is
essential because several head-and-neck labels are nested or overlapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from prepare_hntsmrg24 import robust_normalize


HAN_SEG_LABEL_NAMES = (
    "A_Carotid_L",
    "A_Carotid_R",
    "Arytenoid",
    "Bone_Mandible",
    "Brainstem",
    "BuccalMucosa",
    "Cavity_Oral",
    "Cochlea_L",
    "Cochlea_R",
    "Cricopharyngeus",
    "Esophagus_S",
    "Eye_AL",
    "Eye_AR",
    "Eye_PL",
    "Eye_PR",
    "Glnd_Lacrimal_L",
    "Glnd_Lacrimal_R",
    "Glnd_Submand_L",
    "Glnd_Submand_R",
    "Glnd_Thyroid",
    "Glottis",
    "Larynx_SG",
    "Lips",
    "OpticChiasm",
    "OpticNrv_L",
    "OpticNrv_R",
    "Parotid_L",
    "Parotid_R",
    "Pituitary",
    "SpinalCord",
)


SEGRAP_OAR_LABEL_NAMES = (
    "Brain",
    "BrainStem",
    "Chiasm",
    "Cochlea_L",
    "Cochlea_R",
    "ETbone_L",
    "ETbone_R",
    "Esophagus",
    "Eye_L",
    "Eye_R",
    "Hippocampus_L",
    "Hippocampus_R",
    "IAC_L",
    "IAC_R",
    "Larynx",
    "Larynx_Glottic",
    "Larynx_Supraglot",
    "Lens_L",
    "Lens_R",
    "Mandible_L",
    "Mandible_R",
    "Mastoid_L",
    "Mastoid_R",
    "MiddleEar_L",
    "MiddleEar_R",
    "OpticNerve_L",
    "OpticNerve_R",
    "OralCavity",
    "Parotid_L",
    "Parotid_R",
    "PharynxConst",
    "Pituitary",
    "SpinalCord",
    "Submandibular_L",
    "Submandibular_R",
    "TMjoint_L",
    "TMjoint_R",
    "TemporalLobe_L",
    "TemporalLobe_R",
    "Thyroid",
    "Trachea",
    "TympanicCavity_L",
    "TympanicCavity_R",
    "VestibulSemi_L",
    "VestibulSemi_R",
)


SEGRAP_LABEL_NAMES = SEGRAP_OAR_LABEL_NAMES + ("GTVp", "GTVnd")


HAN_QUALITY_NAME = {
    "Eye_AL": "Eye_L",
    "Eye_AR": "Eye_R",
    "Eye_PL": "Lens_L",
    "Eye_PR": "Lens_R",
}


@dataclass(frozen=True)
class Subject:
    subject_id: str
    image: str
    masks: Tuple[Tuple[int, str], ...]
    valid_labels: Tuple[int, ...]


@dataclass(frozen=True)
class PairedSubject:
    subject_id: str
    moving: str
    fixed: str


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "SimpleITK is required; install requirements.txt first"
        ) from exc
    return sitk


def _natural_key(value: str):
    return tuple(
        int(item) if item.isdigit() else item.lower()
        for item in re.split(r"(\d+)", value)
    )


def _strip_medical_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".seg.nrrd", ".nrrd", ".mha", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _read_han_quality(path: Path) -> Dict[str, Dict[str, float]]:
    quality: Dict[str, Dict[str, float]] = {}
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        rows = list(reader)
    if not rows or len(rows[0]) < 2:
        raise ValueError("invalid HaN-Seg OAR_data.csv: %s" % path)
    names = rows[0][1:]
    for row in rows[1:]:
        if not row:
            continue
        quality[row[0]] = {
            name: float(value)
            for name, value in zip(names, row[1:])
            if value.strip()
        }
    return quality


def discover_han_seg(source_root: str) -> Tuple[List[Subject], Tuple[str, ...]]:
    root = Path(source_root)
    dataset_root = root / "set_1" if (root / "set_1").is_dir() else root
    quality_path = dataset_root / "OAR_data.csv"
    if not quality_path.is_file():
        raise FileNotFoundError("HaN-Seg OAR_data.csv is missing: %s" % quality_path)
    quality = _read_han_quality(quality_path)
    subjects = []
    for directory in sorted(dataset_root.glob("case_*"), key=lambda path: _natural_key(path.name)):
        if not directory.is_dir():
            continue
        subject_id = directory.name
        image = directory / (subject_id + "_IMG_CT.nrrd")
        if not image.is_file():
            raise FileNotFoundError("missing HaN-Seg CT: %s" % image)
        masks = []
        valid = []
        case_quality = quality.get(subject_id, {})
        for label, name in enumerate(HAN_SEG_LABEL_NAMES, start=1):
            mask = directory / (subject_id + "_OAR_" + name + ".seg.nrrd")
            if mask.is_file():
                masks.append((label, str(mask)))
            quality_name = HAN_QUALITY_NAME.get(name, name)
            if mask.is_file() and float(case_quality.get(quality_name, 0.0)) == 1.0:
                valid.append(label)
        subjects.append(
            Subject(
                subject_id=subject_id,
                image=str(image),
                masks=tuple(masks),
                valid_labels=tuple(valid),
            )
        )
    if not subjects:
        raise ValueError("no HaN-Seg cases found under %s" % dataset_root)
    return subjects, HAN_SEG_LABEL_NAMES


def discover_segrap(
    source_root: str,
    modality: str,
) -> Tuple[List[Subject], Tuple[str, ...]]:
    root = Path(source_root)
    image_name = (
        "image_contrast.nii.gz" if modality == "contrast" else "image.nii.gz"
    )
    subjects = []
    for directory in sorted(root.glob("segrap_*"), key=lambda path: _natural_key(path.name)):
        if not directory.is_dir():
            continue
        image = directory / image_name
        if not image.is_file():
            raise FileNotFoundError("missing SegRap image: %s" % image)
        masks = []
        valid = []
        for label, name in enumerate(SEGRAP_LABEL_NAMES, start=1):
            mask = directory / (name + ".nii.gz")
            if mask.is_file():
                masks.append((label, str(mask)))
                valid.append(label)
        subjects.append(
            Subject(
                subject_id=directory.name,
                image=str(image),
                masks=tuple(masks),
                valid_labels=tuple(valid),
            )
        )
    if not subjects:
        raise ValueError("no SegRap training cases found under %s" % root)
    return subjects, SEGRAP_LABEL_NAMES


def discover_cbct_ct(source_root: str) -> List[PairedSubject]:
    root = Path(source_root)
    moving_paths = {
        _strip_medical_suffix(path): path
        for path in (root / "cbct").glob("*.nii.gz")
    }
    fixed_paths = {
        _strip_medical_suffix(path): path
        for path in (root / "ct").glob("*.nii.gz")
    }
    if set(moving_paths) != set(fixed_paths):
        raise ValueError(
            "CBCT and CT identifiers differ: moving-only=%s fixed-only=%s"
            % (
                sorted(set(moving_paths) - set(fixed_paths), key=_natural_key),
                sorted(set(fixed_paths) - set(moving_paths), key=_natural_key),
            )
        )
    subjects = [
        PairedSubject(
            subject_id=subject_id,
            moving=str(moving_paths[subject_id]),
            fixed=str(fixed_paths[subject_id]),
        )
        for subject_id in sorted(moving_paths, key=_natural_key)
    ]
    if not subjects:
        raise ValueError("no paired CBCT/CT cases found under %s" % root)
    return subjects


def _orient_lps(image):
    sitk = _sitk()
    if image.GetDimension() != 3:
        raise ValueError("all source images must be 3D")
    return sitk.DICOMOrient(image, "LPS")


def _validate_geometry(image, mask, path: str) -> None:
    if image.GetSize() != mask.GetSize():
        raise ValueError("image/mask size mismatch: %s" % path)
    if not np.allclose(image.GetSpacing(), mask.GetSpacing(), atol=1.0e-5, rtol=0.0):
        raise ValueError("image/mask spacing mismatch: %s" % path)
    if not np.allclose(image.GetDirection(), mask.GetDirection(), atol=1.0e-5, rtol=0.0):
        raise ValueError("image/mask direction mismatch: %s" % path)
    if not np.allclose(image.GetOrigin(), mask.GetOrigin(), atol=1.0e-4, rtol=0.0):
        raise ValueError("image/mask origin mismatch: %s" % path)


def _axis_quantile(marginal: np.ndarray, quantile: float) -> float:
    cumulative = np.cumsum(np.asarray(marginal, dtype=np.float64))
    if cumulative.size == 0 or cumulative[-1] <= 0.0:
        return 0.5 * float(max(cumulative.size - 1, 0))
    target = float(quantile) * float(cumulative[-1])
    return float(np.searchsorted(cumulative, target, side="left"))


def _body_center_physical(image) -> Tuple[float, float, float]:
    sitk = _sitk()
    array = sitk.GetArrayViewFromImage(image)
    if not np.isfinite(array).all():
        raise ValueError("source image contains non-finite intensities")
    foreground = array > -500.0
    if float(foreground.mean()) < 0.005:
        nonzero = array != 0
        foreground = nonzero if float(nonzero.mean()) >= 0.005 else array > np.percentile(array, 60.0)
    marginals = (
        foreground.sum(axis=(1, 2)),
        foreground.sum(axis=(0, 2)),
        foreground.sum(axis=(0, 1)),
    )
    center_zyx = [
        0.5 * (_axis_quantile(values, 0.01) + _axis_quantile(values, 0.99))
        for values in marginals
    ]
    return tuple(
        float(value)
        for value in image.TransformContinuousIndexToPhysicalPoint(
            tuple(reversed(center_zyx))
        )
    )


def _geometric_center_physical(image) -> Tuple[float, float, float]:
    continuous_index = tuple(
        0.5 * (float(size) - 1.0) for size in image.GetSize()
    )
    return tuple(
        float(value)
        for value in image.TransformContinuousIndexToPhysicalPoint(
            continuous_index
        )
    )


def _crop_center_physical(
    image,
    center_policy: str,
) -> Tuple[float, float, float]:
    if center_policy == "geometric":
        return _geometric_center_physical(image)
    if center_policy == "body":
        return _body_center_physical(image)
    raise ValueError("unknown center policy: %s" % center_policy)


def _target_reference(
    center_xyz: Sequence[float],
    spacing_dhw: Sequence[float],
    shape_dhw: Sequence[int],
):
    sitk = _sitk()
    spacing_xyz = np.asarray(tuple(reversed(spacing_dhw)), dtype=np.float64)
    size_xyz = np.asarray(tuple(reversed(shape_dhw)), dtype=np.int64)
    origin = np.asarray(center_xyz, dtype=np.float64) - (
        size_xyz.astype(np.float64) - 1.0
    ) * spacing_xyz / 2.0
    reference = sitk.Image(
        tuple(int(value) for value in size_xyz),
        sitk.sitkFloat32,
    )
    reference.SetSpacing(tuple(float(value) for value in spacing_xyz))
    reference.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    reference.SetOrigin(tuple(float(value) for value in origin))
    return reference


def _set_common_geometry(image, spacing_dhw: Sequence[float]):
    image.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    image.SetSpacing(tuple(float(value) for value in reversed(spacing_dhw)))
    return image


def _ct_normalize_image(image):
    sitk = _sitk()
    array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    normalized = np.clip((array + 1000.0) / 2500.0, 0.0, 1.0).astype(np.float32)
    result = sitk.GetImageFromArray(normalized)
    result.CopyInformation(image)
    return result


def _center_subject(
    subject: Subject,
    label_count: int,
    spacing_dhw: Sequence[float],
    shape_dhw: Sequence[int],
    center_policy: str,
):
    sitk = _sitk()
    original_source = sitk.ReadImage(subject.image)
    source = _orient_lps(sitk.Cast(original_source, sitk.sitkFloat32))
    center = _crop_center_physical(source, center_policy)
    reference = _target_reference(center, spacing_dhw, shape_dhw)
    centered = sitk.Resample(
        source,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        -1000.0,
        sitk.sitkFloat32,
    )
    centered = _set_common_geometry(_ct_normalize_image(centered), spacing_dhw)
    channels = np.zeros((label_count,) + tuple(shape_dhw), dtype=np.uint8)
    valid = set(int(value) for value in subject.valid_labels)
    expected_labels = []
    for label, path in subject.masks:
        original_mask = sitk.ReadImage(path)
        _validate_geometry(original_source, original_mask, path)
        source_nonempty = bool(
            np.asarray(sitk.GetArrayViewFromImage(original_mask) > 0).any()
        )
        if int(label) in valid and source_nonempty:
            expected_labels.append(int(label))
        mask = _orient_lps(original_mask)
        resampled = sitk.Resample(
            mask,
            reference,
            sitk.Transform(3, sitk.sitkIdentity),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        array = sitk.GetArrayFromImage(resampled) > 0
        channels[int(label) - 1] = array
    missing = [
        label
        for label in expected_labels
        if not bool(channels[int(label) - 1].any())
    ]
    if missing:
        raise ValueError(
            "crop removed non-empty valid labels for %s: %s"
            % (subject.subject_id, missing)
        )
    return centered, channels, tuple(expected_labels), center


def _registration_method(iterations: int, seed: int, learning_rate: float):
    sitk = _sitk()
    method = sitk.ImageRegistrationMethod()
    method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    method.SetMetricSamplingStrategy(method.RANDOM)
    method.SetMetricSamplingPercentage(0.02, int(seed))
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


def _rigid_affine_to_atlas(moving, fixed, seed: int, iterations: int):
    sitk = _sitk()
    rigid = sitk.Euler3DTransform(
        sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    )
    rigid_method = _registration_method(iterations, seed, 2.0)
    rigid_method.SetInitialTransform(rigid, inPlace=True)
    identity_metric = float(rigid_method.MetricEvaluate(fixed, moving))
    rigid_method.Execute(fixed, moving)
    rigid_metric = float(rigid_method.MetricEvaluate(fixed, moving))
    identity = sitk.Transform(3, sitk.sitkIdentity)
    selected = rigid if np.isfinite(rigid_metric) and rigid_metric <= identity_metric else identity
    stage = "rigid" if selected is rigid else "identity"

    if selected is rigid:
        affine = sitk.AffineTransform(3)
        affine.SetCenter(rigid.GetCenter())
        affine.SetMatrix(rigid.GetMatrix())
        affine.SetTranslation(rigid.GetTranslation())
        affine_method = _registration_method(iterations, seed + 1, 0.1)
        affine_method.SetInitialTransform(affine, inPlace=True)
        affine_before = float(affine_method.MetricEvaluate(fixed, moving))
        affine_method.Execute(fixed, moving)
        affine_metric = float(affine_method.MetricEvaluate(fixed, moving))
        matrix = np.asarray(affine.GetMatrix(), dtype=np.float64).reshape(3, 3)
        determinant = float(np.linalg.det(matrix))
        translation = float(np.linalg.norm(np.asarray(affine.GetTranslation())))
        if (
            np.isfinite([affine_before, affine_metric, determinant, translation]).all()
            and affine_metric <= affine_before
            and 0.65 <= determinant <= 1.55
            and translation <= 80.0
        ):
            selected = affine
            stage = "affine"
    registered = sitk.Resample(
        moving,
        fixed,
        selected,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    return registered, selected, {
        "selected_stage": stage,
        "identity_metric": identity_metric,
        "rigid_metric": rigid_metric,
    }


def _warp_binary_channels(channels: np.ndarray, reference, transform) -> np.ndarray:
    sitk = _sitk()
    result = np.zeros_like(channels, dtype=np.uint8)
    for index in range(int(channels.shape[0])):
        image = sitk.GetImageFromArray(channels[index].astype(np.uint8))
        image.CopyInformation(reference)
        warped = sitk.Resample(
            image,
            reference,
            transform,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8,
        )
        result[index] = sitk.GetArrayFromImage(warped) > 0
    return result


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(str(temporary), str(path))


def _save_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(str(temporary), str(path))


def _preprocess_cross_subject(
    subject: Subject,
    output_root: str,
    label_count: int,
    spacing_dhw: Sequence[float],
    shape_dhw: Sequence[int],
    center_policy: str,
    atlas_path: str,
    atlas_subject_id: str,
    prealignment: str,
    registration_iterations: int,
    seed: int,
    overwrite: bool,
) -> Dict[str, object]:
    sitk = _sitk()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    root = Path(output_root)
    metadata_path = root / "metadata" / (subject.subject_id + ".json")
    if metadata_path.is_file() and not overwrite:
        with metadata_path.open("r") as handle:
            record = json.load(handle)
        if (
            tuple(record.get("spacing_dhw", ()))
            != tuple(float(value) for value in spacing_dhw)
            or tuple(record.get("shape_dhw", ()))
            != tuple(int(value) for value in shape_dhw)
            or record.get("center_policy") != center_policy
            or record.get("prealignment_mode") != prealignment
        ):
            raise ValueError(
                "cached preprocessing settings differ for %s; use --overwrite"
                % subject.subject_id
            )
        for field in ("image", "segmentation"):
            if not (root / record[field]).is_file():
                raise FileNotFoundError("cached file is missing: %s" % (root / record[field]))
        return record

    image, channels, valid_labels, center = _center_subject(
        subject,
        label_count,
        spacing_dhw,
        shape_dhw,
        center_policy,
    )
    alignment: Dict[str, object] = {
        "selected_stage": "%s_center" % center_policy
    }
    if prealignment == "rigid_affine" and subject.subject_id != atlas_subject_id:
        atlas = sitk.ReadImage(atlas_path, sitk.sitkFloat32)
        image, transform, alignment = _rigid_affine_to_atlas(
            image,
            atlas,
            seed=seed,
            iterations=registration_iterations,
        )
        channels = _warp_binary_channels(channels, atlas, transform)
    retained = [
        label
        for label in valid_labels
        if bool(channels[int(label) - 1].any())
    ]
    missing = sorted(set(valid_labels) - set(retained))
    if missing:
        raise ValueError(
            "prealignment removed non-empty valid labels for %s: %s"
            % (subject.subject_id, missing)
        )
    image_array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    relative_image = "images/%s.npy" % subject.subject_id
    relative_segmentation = "segmentations/%s.npy" % subject.subject_id
    _save_array(root / relative_image, image_array)
    _save_array(root / relative_segmentation, channels.astype(np.uint8, copy=False))
    record = {
        "subject_id": subject.subject_id,
        "image": relative_image,
        "segmentation": relative_segmentation,
        "segmentation_labels": list(range(1, label_count + 1)),
        "valid_labels": retained,
        "spacing_dhw": [float(value) for value in spacing_dhw],
        "shape_dhw": [int(value) for value in shape_dhw],
        "crop_center_lps": list(center),
        "center_policy": center_policy,
        "prealignment_mode": prealignment,
        "prealignment": alignment,
        "source_image": subject.image,
    }
    _save_json(metadata_path, record)
    return record


def _resample_array_grid(image, reference, default: float):
    sitk = _sitk()
    array_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(image).astype(np.float32))
    array_image.SetSpacing(image.GetSpacing())
    array_image.SetOrigin((0.0, 0.0, 0.0))
    array_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return sitk.Resample(
        array_image,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        float(default),
        sitk.sitkFloat32,
    )


def _preprocess_paired_subject(
    subject: PairedSubject,
    output_root: str,
    spacing_dhw: Sequence[float],
    shape_dhw: Sequence[int],
    overwrite: bool,
) -> Dict[str, object]:
    sitk = _sitk()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    root = Path(output_root)
    metadata_path = root / "metadata" / (subject.subject_id + ".json")
    if metadata_path.is_file() and not overwrite:
        with metadata_path.open("r") as handle:
            record = json.load(handle)
        if (
            tuple(record.get("spacing_dhw", ()))
            != tuple(float(value) for value in spacing_dhw)
            or tuple(record.get("shape_dhw", ()))
            != tuple(int(value) for value in shape_dhw)
        ):
            raise ValueError(
                "cached preprocessing settings differ for %s; use --overwrite"
                % subject.subject_id
            )
        for field in ("moving", "fixed"):
            if not (root / record[field]).is_file():
                raise FileNotFoundError("cached file is missing: %s" % (root / record[field]))
        return record
    moving = sitk.ReadImage(subject.moving, sitk.sitkFloat32)
    fixed = sitk.ReadImage(subject.fixed, sitk.sitkFloat32)
    if moving.GetSize() != fixed.GetSize() or not np.allclose(
        moving.GetSpacing(), fixed.GetSpacing(), atol=1.0e-5, rtol=0.0
    ):
        raise ValueError("paired CBCT/CT array grids differ for %s" % subject.subject_id)
    source_size = np.asarray(moving.GetSize(), dtype=np.float64)
    source_spacing = np.asarray(moving.GetSpacing(), dtype=np.float64)
    source_center = (source_size - 1.0) * source_spacing / 2.0
    reference = _target_reference(source_center, spacing_dhw, shape_dhw)
    moving_resampled = _resample_array_grid(moving, reference, 0.0)
    fixed_resampled = _resample_array_grid(fixed, reference, 0.0)
    moving_array, moving_norm = robust_normalize(
        sitk.GetArrayFromImage(moving_resampled)
    )
    fixed_array, fixed_norm = robust_normalize(
        sitk.GetArrayFromImage(fixed_resampled)
    )
    relative_moving = "moving/%s.npy" % subject.subject_id
    relative_fixed = "fixed/%s.npy" % subject.subject_id
    _save_array(root / relative_moving, moving_array)
    _save_array(root / relative_fixed, fixed_array)
    record = {
        "subject_id": subject.subject_id,
        "moving": relative_moving,
        "fixed": relative_fixed,
        "spacing_dhw": [float(value) for value in spacing_dhw],
        "shape_dhw": [int(value) for value in shape_dhw],
        "moving_normalization": moving_norm,
        "fixed_normalization": fixed_norm,
        "source_moving": subject.moving,
        "source_fixed": subject.fixed,
        "geometry_policy": "provided voxel grid; source origins ignored",
    }
    _save_json(metadata_path, record)
    return record


def _split_subject_ids(
    subject_ids: Sequence[str],
    seed: int,
) -> Dict[str, List[str]]:
    if len(subject_ids) < 10:
        raise ValueError("at least ten subjects are required for train/validation/test")
    ordered = sorted(subject_ids, key=_natural_key)
    rng = np.random.default_rng(int(seed))
    shuffled = [ordered[index] for index in rng.permutation(len(ordered))]
    n_test = max(1, int(round(0.10 * len(shuffled))))
    n_validation = max(1, int(round(0.10 * len(shuffled))))
    return {
        "train": sorted(shuffled[n_test + n_validation :], key=_natural_key),
        "validation": sorted(shuffled[n_test : n_test + n_validation], key=_natural_key),
        "test": sorted(shuffled[:n_test], key=_natural_key),
    }


def _cross_patient_pairs(
    split: str,
    subject_ids: Sequence[str],
    records: Mapping[str, Mapping[str, object]],
    pairs_per_subject: int,
    seed: int,
) -> List[Dict[str, object]]:
    if len(subject_ids) < 2:
        raise ValueError("cross-patient pairing requires at least two subjects")
    count = min(int(pairs_per_subject), len(subject_ids) - 1)
    if count <= 0:
        raise ValueError("pairs_per_subject must be positive")
    rng = np.random.default_rng(int(seed))
    order = [subject_ids[index] for index in rng.permutation(len(subject_ids))]
    rows = []
    for moving_index, moving_id in enumerate(order):
        for offset in range(1, count + 1):
            fixed_id = order[(moving_index + offset) % len(order)]
            moving = records[moving_id]
            fixed = records[fixed_id]
            valid = sorted(
                set(int(value) for value in moving["valid_labels"])
                & set(int(value) for value in fixed["valid_labels"])
            )
            rows.append(
                {
                    "patient_id": "%s__to__%s" % (moving_id, fixed_id),
                    "moving_subject_id": moving_id,
                    "fixed_subject_id": fixed_id,
                    "moving": moving["image"],
                    "fixed": fixed["image"],
                    "moving_seg": moving["segmentation"],
                    "fixed_seg": fixed["segmentation"],
                    "segmentation_labels": ";".join(
                        str(value) for value in moving["segmentation_labels"]
                    ),
                    "valid_labels": ";".join(str(value) for value in valid),
                    "spacing_d": moving["spacing_dhw"][0],
                    "spacing_h": moving["spacing_dhw"][1],
                    "spacing_w": moving["spacing_dhw"][2],
                    "split": split,
                }
            )
    return rows


def _paired_rows(
    split: str,
    subject_ids: Sequence[str],
    records: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    return [
        {
            "patient_id": subject_id,
            "moving_subject_id": subject_id,
            "fixed_subject_id": subject_id,
            "moving": records[subject_id]["moving"],
            "fixed": records[subject_id]["fixed"],
            "spacing_d": records[subject_id]["spacing_dhw"][0],
            "spacing_h": records[subject_id]["spacing_dhw"][1],
            "spacing_w": records[subject_id]["spacing_dhw"][2],
            "split": split,
        }
        for subject_id in subject_ids
    ]


def _write_manifest(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty manifest: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _run_jobs(function, jobs: Iterable[Tuple[object, ...]], workers: int):
    records = []
    failures = []
    jobs = list(jobs)
    if workers <= 1:
        for job in jobs:
            try:
                records.append(function(*job))
            except Exception as exc:
                failures.append({"subject_id": job[0].subject_id, "error": repr(exc)})
        return records, failures
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        futures = {executor.submit(function, *job): job[0] for job in jobs}
        for future in as_completed(futures):
            subject = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append({"subject_id": subject.subject_id, "error": repr(exc)})
    return records, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("han-seg", "head-neck-cbct-ct", "segrap2023"),
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-spacing", nargs=3, type=float, default=None)
    parser.add_argument("--target-shape", nargs=3, type=int, default=None)
    parser.add_argument(
        "--center-policy",
        choices=("geometric", "body"),
        default="geometric",
        help="crop centre for inter-patient datasets; ignored for paired data",
    )
    parser.add_argument(
        "--prealignment",
        choices=("auto", "body_center", "rigid_affine"),
        default="auto",
    )
    parser.add_argument("--registration-iterations", type=int, default=100)
    parser.add_argument("--train-pairs-per-subject", type=int, default=3)
    parser.add_argument("--eval-pairs-per-subject", type=int, default=3)
    parser.add_argument(
        "--segrap-modality",
        choices=("contrast", "plain"),
        default="contrast",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    if args.num_workers < 0 or args.registration_iterations <= 0:
        raise ValueError("workers must be nonnegative and iterations positive")
    paired = args.dataset == "head-neck-cbct-ct"
    default_spacing = (1.5, 1.5, 1.5) if paired else (2.5, 2.0, 2.0)
    default_shape = (128, 160, 160) if paired else (192, 160, 160)
    spacing_dhw = tuple(args.target_spacing or default_spacing)
    shape_dhw = tuple(args.target_shape or default_shape)
    if len(spacing_dhw) != 3 or min(spacing_dhw) <= 0.0:
        raise ValueError("target spacing must contain three positive values")
    if len(shape_dhw) != 3 or any(value <= 0 or value % 16 for value in shape_dhw):
        raise ValueError("target shape values must be positive and divisible by 16")
    output_root = Path(args.output_root)

    if paired:
        subjects = discover_cbct_ct(args.source_root)
        splits = _split_subject_ids(
            [subject.subject_id for subject in subjects],
            args.seed,
        )
        jobs = [
            (
                subject,
                args.output_root,
                spacing_dhw,
                shape_dhw,
                args.overwrite,
            )
            for subject in subjects
        ]
        processed, failures = _run_jobs(
            _preprocess_paired_subject,
            jobs,
            args.num_workers,
        )
        records = {record["subject_id"]: record for record in processed}
        for name, subject_ids in splits.items():
            available = [value for value in subject_ids if value in records]
            _write_manifest(
                output_root / "manifests" / (name + ".csv"),
                _paired_rows(name, available, records),
            )
        label_names: Sequence[str] = ()
        atlas_subject_id: Optional[str] = None
        prealignment = "provided_voxel_grid"
        center_policy = "provided_voxel_grid"
    else:
        if args.dataset == "han-seg":
            subjects, label_names = discover_han_seg(args.source_root)
        else:
            subjects, label_names = discover_segrap(
                args.source_root,
                args.segrap_modality,
            )
        splits = _split_subject_ids(
            [subject.subject_id for subject in subjects],
            args.seed,
        )
        atlas_subject_id = splits["train"][0]
        by_id = {subject.subject_id: subject for subject in subjects}
        prealignment = (
            "rigid_affine"
            if args.prealignment == "auto"
            else args.prealignment
        )
        center_policy = args.center_policy
        atlas_image, _, _, _ = _center_subject(
            by_id[atlas_subject_id],
            len(label_names),
            spacing_dhw,
            shape_dhw,
            center_policy,
        )
        atlas_path = output_root / "metadata" / "training_atlas.nii.gz"
        atlas_path.parent.mkdir(parents=True, exist_ok=True)
        _sitk().WriteImage(atlas_image, str(atlas_path), True)
        jobs = [
            (
                subject,
                args.output_root,
                len(label_names),
                spacing_dhw,
                shape_dhw,
                center_policy,
                str(atlas_path),
                atlas_subject_id,
                prealignment,
                args.registration_iterations,
                args.seed + index * 17,
                args.overwrite,
            )
            for index, subject in enumerate(subjects)
        ]
        processed, failures = _run_jobs(
            _preprocess_cross_subject,
            jobs,
            args.num_workers,
        )
        records = {record["subject_id"]: record for record in processed}
        for offset, (name, subject_ids) in enumerate(splits.items()):
            available = [value for value in subject_ids if value in records]
            pairs_per_subject = (
                args.train_pairs_per_subject
                if name == "train"
                else args.eval_pairs_per_subject
            )
            _write_manifest(
                output_root / "manifests" / (name + ".csv"),
                _cross_patient_pairs(
                    name,
                    available,
                    records,
                    pairs_per_subject,
                    args.seed + 1000 + offset,
                ),
            )

    summary = {
        "dataset": args.dataset,
        "source_root": str(Path(args.source_root).resolve()),
        "output_root": str(output_root.resolve()),
        "num_discovered": len(subjects),
        "num_successful": len(records),
        "num_failed": len(failures),
        "failures": failures,
        "split_subject_ids": splits,
        "split_subject_counts": {
            name: len(values) for name, values in splits.items()
        },
        "target_spacing_dhw": list(spacing_dhw),
        "target_shape_dhw": list(shape_dhw),
        "center_policy": center_policy,
        "prealignment": prealignment,
        "training_atlas_subject_id": atlas_subject_id,
        "label_names": {
            str(index): name
            for index, name in enumerate(label_names, start=1)
        },
        "seed": args.seed,
    }
    _save_json(output_root / "dataset_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures and not args.allow_failures:
        raise SystemExit(
            "preprocessing had failures; inspect dataset_summary.json"
        )


if __name__ == "__main__":
    main()
