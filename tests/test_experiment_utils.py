import unittest

import torch

from experiment_utils import (
    BaselineRegistrationObjective,
    BaselineSACBNet,
    RegistrationObjective,
    bootstrap_mean_ci,
    build_model,
    build_objective,
    config_architecture,
    learning_rate_factor,
    warp_volume,
)
from model import SACB_Net


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
                "token_num_l4": 8,
                "context_channels": 5,
                "residual_hidden_channels": 8,
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
        gradient = model.gacm4.residual_proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertEqual(
            set(terms),
            {
                "similarity",
                "smoothness",
                "deep_similarity",
                "jacobian",
                "total",
            },
        )

    def test_baseline_builder_preserves_original_state_and_common_objective(self):
        config = {
            "data": {"shape_dhw": [32, 32, 32]},
            "model": {
                "architecture": "sacb",
                "channel_scale": 2,
                "num_k": 3,
                "kmeans_max_iter": 2,
                "kmeans_tolerance": 1.0e-4,
            },
            "loss": {"similarity": 1.0, "smoothness": 0.3, "ncc_window": 9},
        }
        self.assertEqual(config_architecture(config), "sacb")
        model = build_model(config)
        objective = build_objective(config)
        self.assertIsInstance(model, BaselineSACBNet)
        self.assertIsInstance(objective, BaselineRegistrationObjective)

        original = SACB_Net(inshape=(32, 32, 32), ch_scale=2, num_k=3)
        self.assertEqual(set(model.state_dict()), set(original.state_dict()))
        for name, value in original.state_dict().items():
            self.assertEqual(model.state_dict()[name].shape, value.shape, name)

        moving = torch.randn(1, 1, 32, 32, 32)
        fixed = torch.randn_like(moving)
        output = model(moving, fixed, return_aux=True)
        self.assertEqual(set(output), {"warped", "flow"})
        terms = objective(output, moving, fixed)
        self.assertEqual(set(terms), {"similarity", "smoothness", "total"})
        self.assertTrue(all(bool(torch.isfinite(value)) for value in terms.values()))
        terms["total"].backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))

    def test_default_architecture_remains_gam_for_existing_configs(self):
        self.assertEqual(config_architecture({}), "gam_sacb")
        self.assertIsInstance(build_objective({}), RegistrationObjective)


if __name__ == "__main__":
    unittest.main()
