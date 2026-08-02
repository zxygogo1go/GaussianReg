import unittest

from prepare_head_neck_datasets import (
    HAN_SEG_LABEL_NAMES,
    SEGRAP_LABEL_NAMES,
    SEGRAP_OAR_LABEL_NAMES,
    _cross_patient_pairs,
    _split_subject_ids,
)


class ExternalPreprocessingTests(unittest.TestCase):
    def test_label_tables_and_patient_split(self):
        self.assertEqual(len(HAN_SEG_LABEL_NAMES), 30)
        self.assertEqual(len(SEGRAP_OAR_LABEL_NAMES), 45)
        self.assertEqual(len(SEGRAP_LABEL_NAMES), 47)
        subject_ids = ["case_%02d" % index for index in range(1, 43)]
        first = _split_subject_ids(subject_ids, seed=2026)
        second = _split_subject_ids(subject_ids, seed=2026)
        self.assertEqual(first, second)
        self.assertEqual(
            {name: len(values) for name, values in first.items()},
            {"train": 34, "validation": 4, "test": 4},
        )
        self.assertFalse(set(first["train"]) & set(first["validation"]))
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["validation"]) & set(first["test"]))

    def test_cross_patient_pairing_is_deterministic_and_response_aware(self):
        subject_ids = ["s%d" % index for index in range(5)]
        records = {
            subject_id: {
                "image": "images/%s.npy" % subject_id,
                "segmentation": "segmentations/%s.npy" % subject_id,
                "segmentation_labels": [1, 2, 3],
                "valid_labels": [1, 2] if index else [1],
                "spacing_dhw": [2.0, 2.0, 2.0],
            }
            for index, subject_id in enumerate(subject_ids)
        }
        first = _cross_patient_pairs(
            "train",
            subject_ids,
            records,
            pairs_per_subject=2,
            seed=7,
        )
        second = _cross_patient_pairs(
            "train",
            subject_ids,
            records,
            pairs_per_subject=2,
            seed=7,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(len({row["patient_id"] for row in first}), 10)
        self.assertTrue(
            all(
                row["moving_subject_id"] != row["fixed_subject_id"]
                for row in first
            )
        )
        for row in first:
            if "s0" in (
                row["moving_subject_id"],
                row["fixed_subject_id"],
            ):
                self.assertEqual(row["valid_labels"], "1")


if __name__ == "__main__":
    unittest.main()
