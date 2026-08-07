import unittest

import torch

from gaussian_native.geometry import GaussianLevel, covariance_from_scale_rotation
from gaussian_native.integration import compose_displacements
from gaussian_native.small_organ_refinement import (
    SmallOrganAdaptiveGaussianRefiner,
    analytic_measurements,
    sample_volume_at_physical_points,
)


def _fine_level(feature_dim=12):
    axes = torch.tensor((4.0, 12.0))
    dd, hh, ww = torch.meshgrid(axes, axes, axes, indexing="ij")
    centers = torch.stack((dd, hh, ww), dim=-1).reshape(1, -1, 3)
    nodes = centers.shape[1]
    scales = torch.full_like(centers, 4.0)
    rotations = torch.eye(3).view(1, 1, 3, 3).expand(1, nodes, -1, -1)
    covariance, precision = covariance_from_scale_rotation(scales, rotations)
    features = torch.randn(1, nodes, feature_dim)
    level = GaussianLevel(
        centers_mm=centers,
        covariance_mm2=covariance,
        precision_mm2=precision,
        scales_mm=scales,
        rotations=rotations,
        mass=torch.full((1, nodes), 1.0 / float(nodes)),
        features=features,
        appearance=torch.zeros(1, nodes, 7),
        parent_index=None,
        anchor_centers_mm=centers,
        anchor_scales_mm=scales,
    )
    match = {
        "transport_delta_mm": torch.zeros_like(centers),
        "matched_feature": features + 0.05 * torch.randn_like(features),
        "match_evidence": torch.full((1, nodes), 0.8),
        "support_entropy": torch.full((1, nodes), 0.2),
        "matched_mass_fraction": torch.full((1, nodes), 0.9),
    }
    return level, match


class SmallOrganAdaptiveGaussianRefinerTests(unittest.TestCase):
    def test_analytic_measurement_and_physical_sampling_shapes(self):
        volume = torch.rand(1, 1, 16, 16, 16)
        measurements = analytic_measurements(
            volume,
            torch.tensor([[1.0, 1.0, 1.0]]),
        )
        self.assertEqual(tuple(measurements.shape), (1, 7, 16, 16, 16))
        points = torch.tensor([[[0.0, 0.0, 0.0], [15.0, 15.0, 15.0]]])
        sampled = sample_volume_at_physical_points(
            measurements,
            points,
            torch.tensor([[15.0, 15.0, 15.0]]),
        )
        self.assertEqual(tuple(sampled.shape), (1, 2, 7))
        self.assertTrue(torch.isfinite(sampled).all())

    def test_zero_initialization_preserves_global_deformation(self):
        refiner = SmallOrganAdaptiveGaussianRefiner(
            feature_dim=12,
            selected_parents=4,
            children_per_parent=9,
            descriptor_dim=8,
            hidden_dim=16,
            synthesis_factor=2,
            integration_steps=2,
            raster_chunk=4,
        )
        level, match = _fine_level()
        moving = torch.rand(1, 1, 16, 16, 16)
        fixed = torch.rand_like(moving)
        base_flow = 0.05 * torch.randn(1, 3, 16, 16, 16)
        output = refiner(
            moving,
            fixed,
            base_flow,
            level,
            match,
            torch.tensor([[1.0, 1.0, 1.0]]),
            torch.tensor([[15.0, 15.0, 15.0]]),
        )
        self.assertEqual(tuple(output["priority"].shape), (1, 8))
        self.assertEqual(tuple(output["child_centers_mm"].shape), (1, 4, 9, 3))
        self.assertEqual(tuple(output["local_transport"].shape), (1, 4, 9, 9))
        self.assertEqual(tuple(output["local_flow"].shape), (1, 3, 16, 16, 16))
        self.assertEqual(float(output["local_velocity_vox"].abs().max()), 0.0)
        self.assertEqual(float(output["local_flow"].abs().max()), 0.0)
        composed = compose_displacements(output["local_flow"], base_flow)
        self.assertTrue(torch.allclose(composed, base_flow, atol=1.0e-6))

    def test_priority_and_velocity_heads_receive_gradients(self):
        refiner = SmallOrganAdaptiveGaussianRefiner(
            feature_dim=12,
            selected_parents=4,
            children_per_parent=9,
            descriptor_dim=8,
            hidden_dim=16,
            synthesis_factor=2,
            integration_steps=2,
            raster_chunk=4,
        )
        level, match = _fine_level()
        moving = torch.rand(1, 1, 16, 16, 16)
        fixed = torch.roll(moving, shifts=1, dims=4)
        base_flow = torch.zeros(1, 3, 16, 16, 16)
        output = refiner(
            moving,
            fixed,
            base_flow,
            level,
            match,
            torch.tensor([[1.0, 1.0, 1.0]]),
            torch.tensor([[15.0, 15.0, 15.0]]),
        )
        loss = (
            (output["local_flow"] - 0.1).square().mean()
            + output["priority"].mean()
        )
        loss.backward()
        self.assertIsNotNone(refiner.velocity_head[-1].weight.grad)
        self.assertGreater(
            float(refiner.velocity_head[-1].weight.grad.abs().sum()),
            0.0,
        )
        self.assertIsNotNone(refiner.priority_head[-1].weight.grad)
        self.assertGreater(
            float(refiner.priority_head[-1].weight.grad.abs().sum()),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
