import unittest

import torch

from experiment_utils import JacobianFoldingLoss
from losses import NCC_vxm


class GAMLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def test_device_safe_ncc(self):
        image = torch.randn(1, 1, 8, 8, 8)
        loss = NCC_vxm(win=[5, 5, 5])(image, image)
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss), -0.99)

    def test_jacobian_loss_is_zero_for_translation(self):
        flow = torch.zeros(1, 3, 6, 6, 6, requires_grad=True)
        translated = flow + torch.tensor(
            [1.0, -0.5, 0.25]
        ).view(1, 3, 1, 1, 1)
        loss = JacobianFoldingLoss()(translated)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)

    def test_jacobian_loss_detects_axis_folding(self):
        coordinate = torch.arange(6.0).view(1, 1, 6, 1, 1)
        flow = torch.zeros(1, 3, 6, 6, 6, requires_grad=True)
        folded = flow.clone()
        folded[:, 0:1] = -2.0 * coordinate
        loss = JacobianFoldingLoss()(folded)
        self.assertGreater(float(loss), 0.5)
        loss.backward()
        self.assertIsNotNone(flow.grad)
        self.assertTrue(torch.isfinite(flow.grad).all())


if __name__ == "__main__":
    unittest.main()
