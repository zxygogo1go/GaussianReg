import unittest

import torch

from experiment_utils import (
    BaselineRegistrationObjective,
    BaselineSACBNet,
    bootstrap_mean_ci,
    build_model,
    build_objective,
    config_architecture,
    learning_rate_factor,
    warp_volume,
)
from gaussian_native import GaussianNativeObjective, GaussianNativeRegistration
from model import SACB_Net
from train_registration import _augment_pair


class ExperimentUtilityTests(unittest.TestCase):
    def test_warp_and_statistical_helpers(self):
        source = torch.arange(5.0).view(1, 1, 1, 1, 5).expand(1, 1, 3, 3, 5)
        identity = warp_volume(source, torch.zeros(1, 3, 3, 3, 5))
        self.assertTrue(torch.allclose(identity, source, atol=1.0e-5))
        factor = learning_rate_factor(
            epoch=9,
            epochs=10,
            warmup_epochs=2,
            minimum_factor=0.05,
        )
        self.assertAlmostEqual(factor, 0.05)
        estimate = bootstrap_mean_ci([1.0, 2.0, 3.0], samples=100, seed=1)
        self.assertEqual(estimate["mean"], 2.0)
        self.assertEqual(estimate["n"], 3)

    def test_baseline_builder_preserves_original_state(self):
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

    def test_default_architecture_is_gaussian_native(self):
        self.assertEqual(config_architecture({}), "gaussian_native")
        self.assertIsInstance(build_objective({}), GaussianNativeObjective)
        config = {
            "data": {
                "shape_dhw": [32, 32, 32],
                "spacing_dhw": [1.5, 1.5, 1.5],
            },
            "model": {
                "root_grid_shape": [2, 2, 2],
                "feature_dim": 24,
                "hidden_dim": 32,
                "graph_heads": 4,
                "graph_neighbors": 4,
                "graph_blocks_per_level": 1,
                "samples_per_axis": 2,
                "pyramid_factors": [8, 4, 2],
                "sinkhorn_iterations": 3,
                "parent_candidates": 2,
                "velocity_hidden_dim": 48,
                "raster_chunk": 16,
                "integration_steps": 3,
            },
        }
        model = build_model(config)
        self.assertIsInstance(model, GaussianNativeRegistration)
        self.assertEqual(model.architecture_revision, "gaussian_native_v3")
        self.assertTrue(model.correspondence.calibration_gradient)
        self.assertEqual(model.correspondence.coordinate_mode, "canonical")
        self.assertTrue(
            model.velocity_synthesis.rasterizer.use_canonical_basis
        )

    def test_pair_augmentation_preserves_shape_and_range(self):
        torch.manual_seed(9)
        moving = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 4, 4, 4)
        fixed = 1.0 - moving
        augmented_moving, augmented_fixed = _augment_pair(
            moving,
            fixed,
            {
                "enabled": True,
                "reverse_pair_probability": 1.0,
                "shared_flip_probability": 1.0,
                "intensity_probability": 1.0,
                "gamma_range": [0.9, 1.1],
                "scale_range": [0.9, 1.1],
                "shift_range": [-0.05, 0.05],
                "noise_std_range": [0.0, 0.01],
            },
        )
        self.assertEqual(augmented_moving.shape, moving.shape)
        self.assertEqual(augmented_fixed.shape, fixed.shape)
        self.assertGreaterEqual(float(augmented_moving.min()), 0.0)
        self.assertLessEqual(float(augmented_moving.max()), 1.0)
        self.assertFalse(torch.equal(augmented_moving, moving))

    def test_shared_intensity_augmentation_preserves_pair_photometry(self):
        torch.manual_seed(17)
        volume = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 4, 4, 4)
        augmented_moving, augmented_fixed = _augment_pair(
            volume,
            volume.clone(),
            {
                "enabled": True,
                "reverse_pair_probability": 0.0,
                "shared_flip_probability": 0.0,
                "intensity_probability": 1.0,
                "intensity_pair_mode": "shared",
                "gamma_range": [0.8, 1.2],
                "scale_range": [0.9, 1.1],
                "shift_range": [-0.1, 0.1],
                "noise_std_range": [0.0, 0.0],
            },
        )
        self.assertTrue(torch.equal(augmented_moving, augmented_fixed))
        self.assertFalse(torch.equal(augmented_moving, volume))


if __name__ == "__main__":
    unittest.main()
