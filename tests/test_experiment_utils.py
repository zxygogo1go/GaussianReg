import unittest

import torch

from experiment_utils import (
    RegistrationObjective,
    bootstrap_mean_ci,
    build_model,
    learning_rate_factor,
    warp_volume,
)


class ExperimentUtilityTests(unittest.TestCase):
    def test_warp_and_statistical_helpers(self):
        source = torch.arange(5.0).view(1, 1, 1, 1, 5).expand(1, 1, 3, 3, 5)
        identity = warp_volume(source, torch.zeros(1, 3, 3, 3, 5))
        self.assertTrue(torch.allclose(identity, source, atol=1.0e-5))
        factor = learning_rate_factor(epoch=9, epochs=10, warmup_epochs=2, minimum_factor=0.05)
        self.assertAlmostEqual(factor, 0.05)
        estimate = bootstrap_mean_ci([1.0, 2.0, 3.0], samples=100, seed=1)
        self.assertEqual(estimate["mean"], 2.0)
        self.assertEqual(estimate["n"], 3)

    def test_full_objective_is_finite_and_reaches_gaussian_module(self):
        config = {
            "data": {"shape_dhw": [32, 32, 32]},
            "model": {
                "channel_scale": 2,
                "num_k": 3,
                "token_dim": 8,
                "token_num_l5": 8,
                "token_num_l4": 8,
                "num_types": 3,
                "fusion_hidden_channels": 8,
            },
        }
        model = build_model(config)
        objective = RegistrationObjective({})
        moving = torch.randn(1, 1, 32, 32, 32)
        fixed = torch.randn_like(moving)
        output = model(moving, fixed, return_aux=True)
        terms = objective(output, moving, fixed)
        self.assertTrue(all(bool(torch.isfinite(value)) for value in terms.values()))
        terms["total"].backward()
        # The 32^3 smoke model has only 2^3 level-5 voxels, so its 8-anchor
        # spatial attention is intentionally saturated. Query gradients are
        # covered by the dedicated non-saturated tokenizer test; here verify
        # that the assembled objective reaches the anatomy correspondence head.
        gradient = model.gacm5.tokenizer.type_head.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
