import unittest

import torch

from experiment_utils import build_model, build_objective
from gaussian_native import GaussianNativeObjective, GaussianNativeRegistration
from gaussian_native.integration import ScalingAndSquaring, compose_displacements
from metrics import jacobian_metrics


def small_config():
    return {
        "data": {
            "shape_dhw": [32, 32, 32],
            "spacing_dhw": [1.5, 1.5, 1.5],
        },
        "model": {
            "architecture": "gaussian_native",
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
        "loss": {"ncc_window": 5},
    }


class GaussianNativeModelTests(unittest.TestCase):
    def test_principal_ablation_switches_construct_and_run(self):
        config = small_config()
        config["model"].update(
            {
                "covariance_mode": "diagonal",
                "motion_mode": "translation",
                "integration_mode": "direct",
                "dustbin_mass": 0.0,
            }
        )
        model = build_model(config)
        volume = torch.rand(1, 1, 32, 32, 32)
        with torch.no_grad():
            output = model(volume, volume, return_aux=True)
        self.assertEqual(float(output["flow"].abs().max()), 0.0)
        for level in output["fixed_decomposition"]["levels"]:
            off_diagonal = level.covariance_mm2 - torch.diag_embed(
                torch.diagonal(level.covariance_mm2, dim1=-2, dim2=-1)
            )
            self.assertEqual(float(off_diagonal.abs().max()), 0.0)

    def test_zero_initialization_is_identity_and_diffeomorphic(self):
        model = GaussianNativeRegistration(
            inshape=(32, 32, 32),
            spacing_dhw=(1.5, 1.5, 1.5),
            root_grid_shape=(2, 2, 2),
            feature_dim=24,
            hidden_dim=32,
            graph_heads=4,
            graph_neighbors=4,
            graph_blocks_per_level=1,
            samples_per_axis=2,
            pyramid_factors=(8, 4, 2),
            sinkhorn_iterations=3,
            parent_candidates=2,
            velocity_hidden_dim=48,
            raster_chunk=16,
            integration_steps=3,
        )
        volume = torch.rand(1, 1, 32, 32, 32)
        with torch.no_grad():
            warped, flow = model(volume, volume)
        self.assertEqual(tuple(flow.shape), (1, 3, 32, 32, 32))
        self.assertEqual(float(flow.abs().max()), 0.0)
        self.assertTrue(torch.allclose(warped, volume, atol=2.0e-5))
        jacobian = jacobian_metrics(flow)
        self.assertEqual(jacobian["negative_jacobian_ratio"], 0.0)
        self.assertEqual(jacobian["minimum_jacobian"], 1.0)

    def test_full_objective_backward_reaches_motion_and_decomposition(self):
        model = build_model(small_config())
        objective = build_objective(small_config())
        self.assertIsInstance(model, GaussianNativeRegistration)
        self.assertIsInstance(objective, GaussianNativeObjective)
        moving = torch.rand(1, 1, 32, 32, 32)
        fixed = torch.rand_like(moving)
        output = model(moving, fixed, return_aux=True)
        terms = objective(output, moving, fixed)
        self.assertEqual(
            set(terms),
            {"similarity", "representation", "correspondence", "deformation", "total"},
        )
        self.assertTrue(all(torch.isfinite(value) for value in terms.values()))
        terms["total"].backward()
        motion_gradient = model.velocity_head.network[-1].weight.grad
        geometry_gradient = model.decomposer.root_geometry.geometry.weight.grad
        self.assertIsNotNone(motion_gradient)
        self.assertIsNotNone(geometry_gradient)
        self.assertGreater(float(motion_gradient.abs().sum()), 0.0)
        self.assertGreater(float(geometry_gradient.abs().sum()), 0.0)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_scaling_and_squaring_inverse_composition(self):
        integrator = ScalingAndSquaring(steps=5)
        velocity = torch.zeros(1, 3, 12, 12, 12)
        velocity[:, 2] = 0.25
        forward = integrator(velocity)
        inverse = integrator(-velocity)
        residual = compose_displacements(forward, inverse)
        self.assertLess(float(residual.abs().max()), 1.0e-4)


if __name__ == "__main__":
    unittest.main()
