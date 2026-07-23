import unittest

import numpy as np
import torch

from metrics import (
    dice_per_class,
    jacobian_determinant,
    jacobian_metrics,
    surface_distance_metrics,
)


class RegistrationMetricTests(unittest.TestCase):
    def test_identical_masks(self):
        mask = np.zeros((8, 8, 8), dtype=np.int16)
        mask[2:6, 2:6, 2:6] = 1
        dice = dice_per_class(mask, mask, labels=[1])
        surface = surface_distance_metrics(mask == 1, mask == 1, spacing_dhw=(1.5, 1.5, 1.5))
        self.assertEqual(dice[1], 1.0)
        self.assertEqual(surface["hd95"], 0.0)
        self.assertEqual(surface["assd"], 0.0)

    def test_response_aware_missing_class(self):
        moving = np.zeros((4, 4, 4), dtype=np.int16)
        fixed = np.zeros_like(moving)
        fixed[1:3, 1:3, 1:3] = 2
        self.assertNotIn(2, dice_per_class(moving, fixed, labels=[2], response_aware=True))
        self.assertEqual(dice_per_class(moving, fixed, labels=[2], response_aware=False)[2], 0.0)

    def test_explicit_valid_class_does_not_hide_failed_warp(self):
        prediction = np.zeros((4, 4, 4), dtype=np.int16)
        fixed = np.zeros_like(prediction)
        fixed[1:3, 1:3, 1:3] = 1
        score = dice_per_class(
            prediction,
            fixed,
            labels=[1, 2],
            response_aware=True,
            valid_labels=[1],
        )
        self.assertEqual(score, {1: 0.0})
        surface = surface_distance_metrics(
            prediction == 1,
            fixed == 1,
            spacing_dhw=(1.0, 1.0, 1.0),
        )
        expected_diagonal = np.sqrt(3.0 * 3.0 ** 2)
        self.assertAlmostEqual(surface["hd95"], expected_diagonal)
        self.assertAlmostEqual(surface["assd"], expected_diagonal)

    def test_identity_flow_has_unit_jacobian(self):
        flow = torch.zeros(2, 3, 5, 6, 7)
        determinant = jacobian_determinant(flow, spacing_dhw=(1.5, 1.5, 1.5))
        self.assertTrue(torch.allclose(determinant, torch.ones_like(determinant)))
        metrics = jacobian_metrics(flow)
        self.assertEqual(metrics["negative_jacobian_ratio"], 0.0)
        self.assertEqual(metrics["minimum_jacobian"], 1.0)


if __name__ == "__main__":
    unittest.main()
