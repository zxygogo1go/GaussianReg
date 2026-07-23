"""Manifest-based preprocessed head-and-neck registration datasets."""

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
        raise ValueError("each longitudinal patient may appear only once per manifest")
    return rows


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_npy(path: Path, dtype: np.dtype) -> np.ndarray:
    if path.suffix != ".npy":
        raise ValueError("preprocessed training volumes must be .npy files: %s" % path)
    array = np.load(str(path), allow_pickle=False)
    if array.ndim != 3:
        raise ValueError("volume must be 3D: %s" % path)
    if not np.isfinite(array).all():
        raise ValueError("volume contains non-finite values: %s" % path)
    return np.ascontiguousarray(array.astype(dtype, copy=False))


class HeadNeckRegistrationDataset(Dataset):
    """One longitudinal moving/fixed pair per patient."""

    def __init__(
        self,
        manifest: Union[str, Path],
        data_root: Union[str, Path],
        expected_shape: Optional[Sequence[int]] = (128, 160, 160),
        load_segmentations: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest)
        self.data_root = Path(data_root)
        self.rows = read_manifest(self.manifest_path)
        self.expected_shape = None if expected_shape is None else tuple(int(v) for v in expected_shape)
        self.load_segmentations = bool(load_segmentations)

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
        if self.load_segmentations:
            for field in ("moving_seg", "fixed_seg"):
                if not row.get(field):
                    raise ValueError("%s is required for patient %s" % (field, row["patient_id"]))
            moving_seg = _load_npy(_resolve(self.data_root, row["moving_seg"]), np.int16)
            fixed_seg = _load_npy(_resolve(self.data_root, row["fixed_seg"]), np.int16)
            if moving_seg.shape != moving.shape or fixed_seg.shape != fixed.shape:
                raise ValueError("image/segmentation shape mismatch for patient %s" % row["patient_id"])
            present_both = [
                bool(np.any(moving_seg == label) and np.any(fixed_seg == label))
                for label in (1, 2)
            ]
            sample.update(
                {
                    "moving_seg": torch.from_numpy(moving_seg[None].astype(np.int64)),
                    "fixed_seg": torch.from_numpy(fixed_seg[None].astype(np.int64)),
                    "response_valid": torch.tensor(present_both, dtype=torch.bool),
                }
            )
        return sample


__all__ = [
    "HeadNeckRegistrationDataset",
    "read_manifest",
    "manifest_sha256",
]
