import json
import tempfile
import unittest
from pathlib import Path

from analyze_inference_interventions import analyze


def _evaluation(dice_values, ncc_values, intervention):
    patients = []
    for index, (dice, ncc) in enumerate(zip(dice_values, ncc_values), start=1):
        patients.append(
            {
                "patient_id": str(index),
                "valid_labels": [1, 2],
                "spacing_dhw": [1.5, 1.5, 1.5],
                "image": {"ncc_after": ncc},
                "segmentation_after": {
                    "mean_dice": dice,
                    "dice_per_class": {"1": dice - 0.1, "2": dice + 0.1},
                    "mean_hd95": 5.0,
                    "mean_assd": 1.0,
                },
                "jacobian": {
                    "negative_jacobian_ratio": 0.002,
                    "below_safe_jacobian_ratio": 0.1,
                    "minimum_jacobian": -1.0,
                },
                "deformation": {
                    "mean_displacement_mm": 3.0,
                    "p95_displacement_mm": 6.0,
                },
            }
        )
    return {
        "architecture": "gam_sacb",
        "checkpoint": "/checkpoint.pt",
        "checkpoint_sha256": "checkpoint-hash",
        "checkpoint_epoch": 10,
        "manifest": "/test.csv",
        "manifest_sha256": "manifest-hash",
        "intervention": intervention,
        "patients": patients,
    }


class InferenceInterventionAnalysisTests(unittest.TestCase):
    def test_patient_paired_effect_and_replicate_averaging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = {
                "reference.json": _evaluation([0.5, 0.6], [0.7, 0.8], {"name": "learned"}),
                "roll_a.json": _evaluation([0.4, 0.5], [0.69, 0.79], {"name": "gaussian_roll", "seed": 1}),
                "roll_b.json": _evaluation([0.6, 0.5], [0.67, 0.77], {"name": "gaussian_roll", "seed": 2}),
            }
            for name, value in values.items():
                (root / name).write_text(json.dumps(value))
            result = analyze(
                str(root / "reference.json"),
                [
                    ("gaussian_roll", str(root / "roll_a.json")),
                    ("gaussian_roll", str(root / "roll_b.json")),
                ],
                bootstrap_samples=100,
                seed=7,
            )
        dice = next(
            row
            for row in result["paired_comparisons"]
            if row["condition"] == "gaussian_roll" and row["metric"] == "dice_after"
        )
        ncc = next(
            row
            for row in result["paired_comparisons"]
            if row["condition"] == "gaussian_roll" and row["metric"] == "ncc_after"
        )
        self.assertEqual(dice["replicates"], 2)
        self.assertAlmostEqual(dice["condition_mean"], 0.5)
        self.assertAlmostEqual(dice["raw_delta"]["mean"], -0.05)
        self.assertAlmostEqual(ncc["raw_delta"]["mean"], -0.02)
        self.assertEqual(ncc["classification"], "meaningful_harm")


if __name__ == "__main__":
    unittest.main()
