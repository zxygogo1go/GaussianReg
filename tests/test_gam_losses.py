import unittest

import torch

from gaussian_anatomy import AnisotropicGaussianTokenizer3D
from losses import (
    AnchorFlowConsistencyLoss,
    GaussianTokenRegularization,
    NCC_vxm,
    TransportCostLoss,
)


class GAMLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def test_device_safe_ncc(self):
        image = torch.randn(1, 1, 8, 8, 8)
        loss = NCC_vxm(win=[5, 5, 5])(image, image)
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss.detach()), -0.99)

    def test_token_and_transport_losses(self):
        tokenizer = AnisotropicGaussianTokenizer3D(2, (4, 4, 4), token_dim=8, num_tokens=8, num_types=3)
        tokens = tokenizer(torch.randn(1, 2, 4, 4, 4), return_attention=True)
        token_loss = GaussianTokenRegularization()(tokens)
        plan = torch.rand(1, 8, 8)
        cost = torch.rand(1, 8, 8)
        transport_loss = TransportCostLoss()(plan, cost)
        self.assertTrue(torch.isfinite(token_loss))
        self.assertTrue(torch.isfinite(transport_loss))

    def test_anchor_loss_respects_dhw_sampling_order(self):
        flow = torch.zeros(1, 3, 5, 5, 5)
        flow[:, 0] = 2.0
        fixed_mu = torch.tensor([[[0.0, 0.0, 0.0]]])
        target = torch.tensor([[[2.0, 0.0, 0.0]]])
        confidence = torch.ones(1, 1, 1)
        matching = AnchorFlowConsistencyLoss()(flow, fixed_mu, target, confidence)
        wrong = AnchorFlowConsistencyLoss()(flow, fixed_mu, torch.zeros_like(target), confidence)
        self.assertLess(float(matching.detach()), 0.01)
        self.assertGreater(float(wrong.detach()), 0.5)


if __name__ == "__main__":
    unittest.main()
