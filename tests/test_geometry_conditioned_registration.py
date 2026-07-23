import unittest

import torch

from geometry_conditioned_registration import GeometryConditionedDenseRegistrationBlock


class GeometryConditionedRegistrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)

    def test_shapes_safe_initialization_and_gradients(self):
        block = GeometryConditionedDenseRegistrationBlock(
            feat_ch=4,
            context_ch=11,
            hidden_ch=8,
        )
        moving = torch.randn(2, 4, 4, 4, 4, requires_grad=True)
        fixed = torch.randn(2, 4, 4, 4, 4, requires_grad=True)
        dense = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
        gaussian = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
        context = torch.randn(2, 11, 4, 4, 4, requires_grad=True)
        fused, gate = block(moving, fixed, dense, gaussian, context)
        self.assertEqual(fused.shape, dense.shape)
        self.assertEqual(gate.shape, (2, 1, 4, 4, 4))
        self.assertTrue(torch.isfinite(fused).all())
        self.assertTrue(torch.isfinite(gate).all())
        expected_gate = torch.sigmoid(torch.tensor(-4.0))
        self.assertTrue(torch.allclose(gate, torch.full_like(gate, expected_gate), atol=1.0e-6))
        expected = expected_gate * gaussian + (1.0 - expected_gate) * dense
        self.assertTrue(torch.allclose(fused, expected, atol=1.0e-6))

        fused.square().mean().backward()
        self.assertGreater(float(gaussian.grad.abs().sum().detach()), 0.0)
        self.assertGreater(float(dense.grad.abs().sum().detach()), 0.0)
        self.assertGreater(float(block.gate_head[-1].weight.grad.abs().sum().detach()), 0.0)
        self.assertGreater(float(block.residual_head[-1].weight.grad.abs().sum().detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
