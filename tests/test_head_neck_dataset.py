import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset.head_neck import HeadNeckRegistrationDataset, manifest_sha256, read_manifest
from prepare_hntsmrg24 import _stratified_split, robust_normalize


class HeadNeckDatasetTests(unittest.TestCase):
    def test_manifest_loading_shapes_and_response_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (8, 8, 8)
            moving = np.linspace(0.0, 1.0, np.prod(shape), dtype=np.float32).reshape(shape)
            fixed = moving[::-1].copy()
            moving_seg = np.zeros(shape, dtype=np.int16)
            fixed_seg = np.zeros(shape, dtype=np.int16)
            moving_seg[1:3, 1:3, 1:3] = 1
            fixed_seg[2:4, 2:4, 2:4] = 1
            fixed_seg[5:7, 5:7, 5:7] = 2
            arrays = {
                "moving.npy": moving,
                "fixed.npy": fixed,
                "moving_seg.npy": moving_seg,
                "fixed_seg.npy": fixed_seg,
            }
            for name, array in arrays.items():
                np.save(str(root / name), array, allow_pickle=False)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "patient_id",
                        "moving",
                        "fixed",
                        "moving_seg",
                        "fixed_seg",
                        "spacing_d",
                        "spacing_h",
                        "spacing_w",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "patient_id": "001",
                        "moving": "moving.npy",
                        "fixed": "fixed.npy",
                        "moving_seg": "moving_seg.npy",
                        "fixed_seg": "fixed_seg.npy",
                        "spacing_d": 1.5,
                        "spacing_h": 1.5,
                        "spacing_w": 1.5,
                    }
                )
            dataset = HeadNeckRegistrationDataset(manifest, root, expected_shape=shape)
            sample = dataset[0]
            self.assertEqual(tuple(sample["moving"].shape), (1,) + shape)
            self.assertEqual(sample["response_valid"].tolist(), [True, False])
            self.assertEqual(len(manifest_sha256(manifest)), 64)

    def test_duplicate_patients_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["patient_id", "moving", "fixed"])
                writer.writeheader()
                writer.writerow({"patient_id": "1", "moving": "a.npy", "fixed": "b.npy"})
                writer.writerow({"patient_id": "1", "moving": "c.npy", "fixed": "d.npy"})
            with self.assertRaises(ValueError):
                read_manifest(manifest)

    def test_normalization_and_stratified_split_are_deterministic(self):
        volume = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
        normalized, metadata = robust_normalize(volume)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertGreaterEqual(float(normalized.min()), 0.0)
        self.assertLessEqual(float(normalized.max()), 1.0)
        self.assertLess(metadata["lower_value"], metadata["upper_value"])
        records = [
            {"patient_id": str(index), "fixed_labels": [0, 1, 2]}
            for index in range(20)
        ]
        first = _stratified_split(records, seed=2026)
        second = _stratified_split(records, seed=2026)
        self.assertEqual(first, second)
        self.assertEqual({name: len(values) for name, values in first.items()}, {"train": 16, "validation": 2, "test": 2})


if __name__ == "__main__":
    unittest.main()
