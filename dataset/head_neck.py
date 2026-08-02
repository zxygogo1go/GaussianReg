"""Manifest-based preprocessed head-and-neck registration datasets.

Segmentations may be stored either as one 3-D integer label map (the legacy
HNTS-MRG24 representation) or as a 4-D ``[C,D,H,W]`` binary array.  The latter
is required for head-and-neck OAR datasets because nested structures, such as
the lens and eye, cannot be represented faithfully by one exclusive label map.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = ("patient_id", "moving", "fixed")


def manifest_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Union[str, Path]) -> List[Dict[str, str]]:
    path = Path(path)
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("manifest must contain a header")
        missing = set(REQUIRED_COLUMNS).difference(reader.fieldnames)
        if missing:
            raise ValueError("manifest is missing columns: %s" % sorted(missing))
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("manifest is empty: %s" % path)
    patient_ids = [row["patient_id"] for row in rows]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("patient_id/pair_id values must be unique per manifest")
    return rows


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_npy(path: Path, dtype: np.dtype, dimensions: Sequence[int] = (3,)) -> np.ndarray:
    if path.suffix != ".npy":
        raise ValueError("preprocessed training volumes must be .npy files: %s" % path)
    array = np.load(str(path), allow_pickle=False)
    if array.ndim not in tuple(int(value) for value in dimensions):
        raise ValueError(
            "array must have dimensions %s: %s" % (tuple(dimensions), path)
        )
    if not np.isfinite(array).all():
        raise ValueError("volume contains non-finite values: %s" % path)
    return np.ascontiguousarray(array.astype(dtype, copy=False))


def _load_segmentation(path: Path) -> np.ndarray:
    if path.suffix != ".npy":
        raise ValueError("preprocessed segmentations must be .npy files: %s" % path)
    array = np.load(str(path), allow_pickle=False)
    if array.ndim not in (3, 4):
        raise ValueError("segmentation must be 3D or 4D: %s" % path)
    if not np.isfinite(array).all():
        raise ValueError("segmentation contains non-finite values: %s" % path)
    if array.ndim == 4:
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise ValueError("4D segmentation channels must be binary: %s" % path)
        return np.ascontiguousarray(array.astype(np.uint8, copy=False))
    return np.ascontiguousarray(array.astype(np.int16, copy=False))


def _parse_labels(value: str) -> List[int]:
    if not value or not value.strip():
        return []
    normalized = value.replace(",", ";")
    labels = [int(item.strip()) for item in normalized.split(";") if item.strip()]
    if len(labels) != len(set(labels)) or any(label <= 0 for label in labels):
        raise ValueError("labels must be unique positive integers: %s" % value)
    return labels


def _select_segmentation_channels(
    segmentation: np.ndarray,
    row: Dict[str, str],
    labels: Sequence[int],
    field: str,
) -> np.ndarray:
    """Return a stored label map or requested binary channels.

    A channel array is subset/reordered here so training may supervise a
    memory-conscious label subset while validation evaluates every class.
    """
    if segmentation.ndim == 3:
        return segmentation
    source_labels = _parse_labels(row.get("segmentation_labels", ""))
    if len(source_labels) != int(segmentation.shape[0]):
        raise ValueError(
            "%s channel count does not match segmentation_labels for %s"
            % (field, row["patient_id"])
        )
    source_index = {label: index for index, label in enumerate(source_labels)}
    selected = np.zeros(
        (len(labels),) + tuple(segmentation.shape[1:]),
        dtype=np.uint8,
    )
    for output_index, label in enumerate(labels):
        if int(label) in source_index:
            selected[output_index] = segmentation[source_index[int(label)]] > 0
    return np.ascontiguousarray(selected)


class HeadNeckRegistrationDataset(Dataset):
    """Manifest-defined paired head-and-neck volumes."""

    def __init__(
        self,
        manifest: Union[str, Path],
        data_root: Union[str, Path],
        expected_shape: Optional[Sequence[int]] = (128, 160, 160),
        load_segmentations: bool = True,
        labels: Sequence[int] = (1, 2),
    ) -> None:
        self.manifest_path = Path(manifest)
        self.data_root = Path(data_root)
        self.rows = read_manifest(self.manifest_path)
        self.expected_shape = None if expected_shape is None else tuple(int(v) for v in expected_shape)
        self.load_segmentations = bool(load_segmentations)
        self.labels = tuple(int(value) for value in labels)
        if len(self.labels) != len(set(self.labels)) or any(
            value <= 0 for value in self.labels
        ):
            raise ValueError("labels must be unique positive integers")
        if self.load_segmentations and not self.labels:
            raise ValueError("at least one label is required when loading segmentations")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        moving = _load_npy(_resolve(self.data_root, row["moving"]), np.float32)
        fixed = _load_npy(_resolve(self.data_root, row["fixed"]), np.float32)
        if moving.shape != fixed.shape:
            raise ValueError("moving/fixed shape mismatch for patient %s" % row["patient_id"])
        if self.expected_shape is not None and moving.shape != self.expected_shape:
            raise ValueError(
                "unexpected shape for patient %s: %s != %s"
                % (row["patient_id"], moving.shape, self.expected_shape)
            )
        spacing = tuple(float(row.get(name, 1.5) or 1.5) for name in ("spacing_d", "spacing_h", "spacing_w"))
        sample: Dict[str, object] = {
            "patient_id": row["patient_id"],
            "moving": torch.from_numpy(moving[None]),
            "fixed": torch.from_numpy(fixed[None]),
            "spacing_dhw": torch.tensor(spacing, dtype=torch.float32),
        }
        for field in ("moving_subject_id", "fixed_subject_id"):
            if row.get(field):
                sample[field] = row[field]
        if self.load_segmentations:
            for field in ("moving_seg", "fixed_seg"):
                if not row.get(field):
                    raise ValueError("%s is required for patient %s" % (field, row["patient_id"]))
            moving_seg = _load_segmentation(
                _resolve(self.data_root, row["moving_seg"])
            )
            fixed_seg = _load_segmentation(
                _resolve(self.data_root, row["fixed_seg"])
            )
            moving_seg = _select_segmentation_channels(
                moving_seg,
                row,
                self.labels,
                "moving_seg",
            )
            fixed_seg = _select_segmentation_channels(
                fixed_seg,
                row,
                self.labels,
                "fixed_seg",
            )
            if moving_seg.shape[-3:] != moving.shape or fixed_seg.shape[-3:] != fixed.shape:
                raise ValueError("image/segmentation shape mismatch for patient %s" % row["patient_id"])
            declared_valid = (
                set(_parse_labels(row.get("valid_labels", "")))
                if "valid_labels" in row
                else None
            )
            if moving_seg.ndim == 3:
                present_both = [
                    bool(
                        np.any(moving_seg == label)
                        and np.any(fixed_seg == label)
                        and (declared_valid is None or label in declared_valid)
                    )
                    for label in self.labels
                ]
                moving_tensor = torch.from_numpy(moving_seg[None].astype(np.int64))
                fixed_tensor = torch.from_numpy(fixed_seg[None].astype(np.int64))
            else:
                present_both = [
                    bool(
                        np.any(moving_seg[index] > 0)
                        and np.any(fixed_seg[index] > 0)
                        and (declared_valid is None or label in declared_valid)
                    )
                    for index, label in enumerate(self.labels)
                ]
                moving_tensor = torch.from_numpy(moving_seg.astype(np.uint8))
                fixed_tensor = torch.from_numpy(fixed_seg.astype(np.uint8))
            sample.update(
                {
                    "moving_seg": moving_tensor,
                    "fixed_seg": fixed_tensor,
                    "response_valid": torch.tensor(present_both, dtype=torch.bool),
                }
            )
        return sample


__all__ = [
    "HeadNeckRegistrationDataset",
    "read_manifest",
    "manifest_sha256",
]
