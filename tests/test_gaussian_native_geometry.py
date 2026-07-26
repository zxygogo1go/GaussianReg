import unittest

import torch

from gaussian_native.decomposition import HierarchicalGaussianDecomposer
from gaussian_native.geometry import (
    covariance_from_scale_rotation,
    rotation_6d_to_matrix,
)


class GaussianNativeGeometryTests(unittest.TestCase):
    def test_rotation_and_covariance_are_valid(self):
        representation = torch.randn(2, 7, 6, requires_grad=True)
        rotation = rotation_6d_to_matrix(representation)
        identity = rotation.transpose(-1, -2) @ rotation
        expected = torch.eye(3).view(1, 1, 3, 3)
        self.assertTrue(torch.allclose(identity, expected, atol=1.0e-5))
        self.assertTrue(torch.all(torch.linalg.det(rotation) > 0.999))

        scales = torch.rand(2, 7, 3, requires_grad=True) + 0.5
        covariance, precision = covariance_from_scale_rotation(scales, rotation)
        eigenvalues = torch.linalg.eigvalsh(covariance)
        self.assertTrue(torch.all(eigenvalues > 0.0))
        product = covariance @ precision
        self.assertTrue(torch.allclose(product, expected, atol=1.0e-4))
        (covariance.square().mean() + precision.square().mean()).backward()
        self.assertIsNotNone(representation.grad)
        self.assertTrue(torch.isfinite(representation.grad).all())

    def test_hierarchy_preserves_mass_and_parent_structure(self):
        decomposer = HierarchicalGaussianDecomposer(
            root_grid_shape=(2, 2, 2),
            feature_dim=24,
            hidden_dim=32,
            pyramid_factors=(8, 4, 2),
            samples_per_axis=2,
            raster_chunk=16,
        )
        volume = torch.rand(1, 1, 32, 32, 32)
        result = decomposer(
            volume,
            torch.tensor([[1.5, 1.5, 1.5]]),
            compute_reconstruction=True,
        )
        levels = result["levels"]
        self.assertEqual([level.centers_mm.shape[1] for level in levels], [8, 32, 128])
        for level in levels:
            self.assertTrue(torch.allclose(level.mass.sum(dim=1), torch.ones(1), atol=1.0e-6))
            self.assertTrue(torch.all(torch.linalg.eigvalsh(level.covariance_mm2) > 0.0))
            self.assertIsNotNone(level.reconstruction)
            self.assertIsNotNone(level.coverage)
            self.assertTrue(torch.isfinite(level.reconstruction).all())
        for child, parent in zip(levels[1:], levels[:-1]):
            grouped = child.mass.reshape(1, parent.mass.shape[1], 4).sum(dim=-1)
            self.assertTrue(torch.allclose(grouped, parent.mass, atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
