import unittest

import torch

from gaussian_anatomy import (
    CompactGaussianTokenizer3D,
    GaussianAnatomyCorrespondenceModule,
)


class GaussianAnatomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)

    def test_compact_tokenizer_shapes_and_diagonal_moments(self):
        tokenizer = CompactGaussianTokenizer3D(
            in_ch=4,
            spatial_size=(4, 5, 6),
            token_dim=8,
            num_tokens=8,
        )
        feature = torch.randn(2, 4, 4, 5, 6)
        tokens = tokenizer(feature, return_attention=True)
        self.assertEqual(tokens.mu.shape, (2, 8, 3))
        self.assertEqual(tokens.variance.shape, (2, 8, 3))
        self.assertEqual(tokens.feat.shape, (2, 8, 8))
        self.assertEqual(tokens.attention.shape, (2, 8, 120))
        self.assertTrue(
            torch.allclose(
                tokens.attention.sum(-1),
                torch.ones(2, 8),
                atol=1.0e-5,
            )
        )
        self.assertTrue(torch.isfinite(tokens.variance).all())
        self.assertGreater(float(tokens.variance.min()), 0.0)
        self.assertFalse(hasattr(tokens, "visibility"))
        self.assertFalse(hasattr(tokens, "anatomy"))
        self.assertFalse(hasattr(tokens, "cov"))

    def test_correspondence_context_and_gradients(self):
        module = GaussianAnatomyCorrespondenceModule(
            in_ch=4,
            spatial_size=(4, 4, 4),
            num_tokens=8,
            token_dim=8,
            context_ch=5,
            raster_voxel_chunk=32,
        )
        moving = torch.randn(
            1,
            4,
            4,
            4,
            4,
            requires_grad=True,
        )
        fixed = torch.randn_like(moving, requires_grad=True)
        output = module(moving, fixed, return_aux=True)
        self.assertEqual(output["context"].shape, (1, 5, 4, 4, 4))
        self.assertEqual(output["correspondence"].shape, (1, 8, 8))
        self.assertTrue(
            torch.allclose(
                output["correspondence"].sum(-1),
                torch.ones(1, 8),
                atol=1.0e-5,
            )
        )
        for key in ("context", "correspondence", "match_cost"):
            self.assertTrue(torch.isfinite(output[key]).all(), key)
        for removed_key in ("flow", "confidence", "transport", "anchor_conf"):
            self.assertNotIn(removed_key, output)

        loss = (
            output["context"].square().mean()
            + 0.01 * output["match_cost"].mean()
        )
        loss.backward()
        gradients = (
            module.tokenizer.queries.grad,
            module.tokenizer.key_proj.weight.grad,
            module.tokenizer.value_proj.weight.grad,
            module.residual_proj.weight.grad,
        )
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
