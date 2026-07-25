import unittest

import torch

from geometry_conditioned_registration import (
    GeometryConditionedResidualCorrector,
)


class GeometryConditionedRegistrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def test_shapes_near_identity_bounded_residual_and_gradients(self):
        block = GeometryConditionedResidualCorrector(
            feat_ch=4,
            context_ch=5,
            hidden_ch=8,
            max_residual=0.75,
        )
        moving = torch.randn(
            2,
            4,
            4,
            4,
            4,
            requires_grad=True,
        )
        fixed = torch.randn_like(moving, requires_grad=True)
        dense = torch.randn(
            2,
            3,
            4,
            4,
            4,
            requires_grad=True,
        )
        context = torch.randn(
            2,
            5,
            4,
            4,
            4,
            requires_grad=True,
        )
        corrected, residual = block(moving, fixed, dense, context)
        self.assertEqual(corrected.shape, dense.shape)
        self.assertEqual(residual.shape, dense.shape)
        self.assertTrue(torch.isfinite(corrected).all())
        self.assertLessEqual(float(residual.abs().max()), 0.75)
        self.assertLess(float(residual.abs().mean()), 1.0e-3)
        self.assertTrue(torch.allclose(corrected, dense + residual))

        corrected.square().mean().backward()
        for gradient in (
            moving.grad,
            fixed.grad,
            dense.grad,
            context.grad,
            block.residual_head[-1].weight.grad,
        ):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_rejects_shape_mismatch(self):
        block = GeometryConditionedResidualCorrector(
            feat_ch=4,
            context_ch=5,
            hidden_ch=8,
        )
        feature = torch.randn(1, 4, 4, 4, 4)
        flow = torch.randn(1, 3, 4, 4, 4)
        bad_context = torch.randn(1, 4, 4, 4, 4)
        with self.assertRaises(AssertionError):
            block(feature, feature, flow, bad_context)


if __name__ == "__main__":
    unittest.main()
