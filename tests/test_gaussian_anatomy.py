import unittest

import torch

from gaussian_anatomy import (
    AnisotropicGaussianTokenizer3D,
    GaussianAnatomyMatcher3D,
    UnbalancedSinkhorn,
    pairwise_bures_wasserstein,
)


class GaussianAnatomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def test_tokenizer_shapes_spd_and_attention_partition(self):
        tokenizer = AnisotropicGaussianTokenizer3D(
            in_ch=4,
            spatial_size=(4, 5, 6),
            token_dim=8,
            num_tokens=8,
            num_types=3,
        )
        feature = torch.randn(2, 4, 4, 5, 6)
        tokens = tokenizer(feature, return_attention=True)
        self.assertEqual(tokens.mu.shape, (2, 8, 3))
        self.assertEqual(tokens.cov.shape, (2, 8, 3, 3))
        self.assertEqual(tokens.feat.shape, (2, 8, 8))
        self.assertEqual(tokens.anatomy.shape, (2, 8, 3))
        self.assertEqual(tokens.visibility.shape, (2, 8, 1))
        self.assertEqual(tokens.attention.shape, (2, 8, 120))
        self.assertTrue(torch.allclose(tokens.attention.sum(-1), torch.ones(2, 8), atol=1.0e-5))
        eigvals = torch.linalg.eigvalsh(tokens.cov.float())
        self.assertTrue(torch.isfinite(eigvals).all())
        self.assertGreater(float(eigvals.min().detach()), 0.0)

    def test_bures_cost_is_separated_and_zero_for_identical_gaussians(self):
        mu = torch.tensor([[[0.0, 0.0, 0.0], [0.5, -0.25, 0.1]]])
        eye = torch.eye(3).reshape(1, 1, 3, 3)
        cov = eye.repeat(1, 2, 1, 1) * 0.04
        center, covariance, full = pairwise_bures_wasserstein(mu, cov, mu, cov, chunk_size=1)
        self.assertTrue(torch.allclose(torch.diagonal(center, dim1=1, dim2=2), torch.zeros(1, 2), atol=1.0e-6))
        self.assertTrue(torch.allclose(torch.diagonal(covariance, dim1=1, dim2=2), torch.zeros(1, 2), atol=2.0e-5))
        self.assertTrue(torch.allclose(full, center + covariance, atol=1.0e-6))

    def test_unbalanced_sinkhorn_is_finite_nonnegative_and_differentiable(self):
        cost = torch.rand(2, 5, 6, requires_grad=True)
        a = torch.softmax(torch.randn(2, 5), dim=-1)
        b = torch.softmax(torch.randn(2, 6), dim=-1)
        plan = UnbalancedSinkhorn(iterations=15)(cost, a, b)
        self.assertEqual(plan.shape, (2, 5, 6))
        self.assertTrue(torch.isfinite(plan).all())
        self.assertTrue((plan >= 0).all())
        plan.sum().backward()
        self.assertIsNotNone(cost.grad)
        self.assertTrue(torch.isfinite(cost.grad).all())

    def test_matcher_outputs_and_gradients(self):
        matcher = GaussianAnatomyMatcher3D(
            in_ch=4,
            spatial_size=(4, 4, 4),
            num_tokens=8,
            token_dim=8,
            num_types=3,
            bures_chunk=4,
            raster_voxel_chunk=32,
        )
        moving = torch.randn(1, 4, 4, 4, 4, requires_grad=True)
        fixed = torch.randn(1, 4, 4, 4, 4, requires_grad=True)
        output = matcher(moving, fixed, return_aux=True)
        self.assertEqual(output["flow"].shape, (1, 3, 4, 4, 4))
        self.assertEqual(output["confidence"].shape, (1, 1, 4, 4, 4))
        self.assertEqual(output["context"].shape, (1, 11, 4, 4, 4))
        for key in ("flow", "confidence", "context", "transport", "cost"):
            self.assertTrue(torch.isfinite(output[key]).all(), key)
        loss = output["flow"].square().mean() + output["cost"].mean()
        loss.backward()
        gradients = (
            matcher.tokenizer.queries.grad,
            matcher.tokenizer.anchors.grad,
            matcher.tokenizer.key_proj.weight.grad,
            matcher.tokenizer.type_head.weight.grad,
        )
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum().detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
