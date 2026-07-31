import unittest

import torch

from gaussian_native.refinement import GaussianGuidedResidualPyramid


class GaussianResidualPyramidTests(unittest.TestCase):
    def test_zero_initialization_preserves_gaussian_velocity(self):
        pyramid = GaussianGuidedResidualPyramid(
            factors=(8, 4, 2),
            channels=(8, 8, 8),
            blocks_per_stage=1,
            maximum_residual_vox=(1.0, 1.0, 1.0),
            integration_steps=3,
        )
        moving = torch.rand(1, 1, 32, 32, 32)
        fixed = torch.rand_like(moving)
        level_fields = [
            torch.zeros(1, 3, 8, 8, 8)
            for _ in range(3)
        ]
        output = pyramid(
            moving,
            fixed,
            level_fields,
            torch.tensor([[1.5, 1.5, 1.5]]),
        )
        self.assertEqual(
            [tuple(flow.shape[2:]) for flow in output["pyramid_flow"]],
            [(4, 4, 4), (8, 8, 8), (16, 16, 16)],
        )
        self.assertEqual(
            tuple(output["flow"].shape),
            (1, 3, 32, 32, 32),
        )
        self.assertEqual(float(output["flow"].abs().max()), 0.0)
        for residual in output["pyramid_residual_velocity_vox"]:
            self.assertEqual(float(residual.abs().max()), 0.0)

    def test_residual_heads_receive_dense_flow_gradient(self):
        pyramid = GaussianGuidedResidualPyramid(
            factors=(8, 4, 2, 1),
            channels=(8, 8, 8, 8),
            blocks_per_stage=(1, 1, 1, 1),
            maximum_residual_vox=(1.0, 1.0, 1.0, 0.5),
            integration_steps=3,
            use_gradient_features=True,
        )
        moving = torch.rand(1, 1, 32, 32, 32)
        fixed = torch.rand_like(moving)
        level_fields = [
            torch.zeros(1, 3, 8, 8, 8)
            for _ in range(3)
        ]
        output = pyramid(
            moving,
            fixed,
            level_fields,
            torch.tensor([[1.5, 1.5, 1.5]]),
        )
        target = torch.ones_like(output["flow"])
        loss = (output["flow"] - target).square().mean()
        loss.backward()
        for stage in pyramid.stages:
            gradient = stage.output.weight.grad
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
