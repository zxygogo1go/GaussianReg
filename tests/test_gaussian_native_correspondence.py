import unittest
from dataclasses import replace

import torch

from gaussian_native.correspondence import HierarchicalGaussianCorrespondence
from gaussian_native.decomposition import HierarchicalGaussianDecomposer
from gaussian_native.encoding import HierarchicalGaussianEncoder


class GaussianNativeCorrespondenceTests(unittest.TestCase):
    def _represent(self, volume):
        decomposer = self.decomposer(
            volume,
            torch.tensor([[1.5, 1.5, 1.5]]),
            compute_reconstruction=False,
        )
        levels = self.encoder(decomposer["levels"], decomposer["extent_mm"])
        return levels, decomposer["extent_mm"]

    def setUp(self):
        torch.manual_seed(4)
        self.decomposer = HierarchicalGaussianDecomposer(
            root_grid_shape=(2, 2, 2),
            feature_dim=24,
            hidden_dim=32,
            pyramid_factors=(8, 4, 2),
            samples_per_axis=2,
            raster_chunk=16,
        )
        self.encoder = HierarchicalGaussianEncoder(
            feature_dim=24,
            heads=4,
            neighbors=4,
            blocks_per_level=1,
        )
        self.matcher = HierarchicalGaussianCorrespondence(
            feature_dim=24,
            sinkhorn_iterations=5,
            parent_candidates=2,
        )

    def test_partial_transport_and_hierarchy_are_finite(self):
        fixed, extent = self._represent(torch.rand(1, 1, 32, 32, 32))
        moving, _ = self._represent(torch.rand(1, 1, 32, 32, 32))
        results = self.matcher(fixed, moving, extent)
        self.assertEqual([result["plan"].shape[1:] for result in results], [(8, 8), (32, 32), (128, 128)])
        self.assertIsNone(results[0]["candidate_mask"])
        self.assertIsNotNone(results[1]["candidate_mask"])
        self.assertTrue(results[1]["candidate_mask"].any())
        for result in results:
            self.assertTrue(torch.isfinite(result["full_plan"]).all())
            self.assertTrue(torch.isfinite(result["cycle_error"]))
            self.assertTrue(torch.isfinite(result["transport_cost"]))
            self.assertTrue(torch.all(result["matched_mass_fraction"] >= 0.0))
            self.assertTrue(torch.all(result["matched_mass_fraction"] <= 1.0))
            total = result["full_plan"].sum(dim=(1, 2))
            self.assertTrue(torch.allclose(total, torch.ones_like(total), atol=1.0e-4))
        loss = sum(result["transport_cost"] + result["cycle_error"] for result in results)
        loss.backward()
        gradients = [parameter.grad for parameter in self.matcher.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_identity_calibrated_transport_is_zero_for_identical_inputs(self):
        fixed, extent = self._represent(torch.rand(1, 1, 32, 32, 32))
        results = self.matcher(fixed, fixed, extent)
        for result in results:
            self.assertTrue(
                torch.allclose(
                    result["transport_delta_mm"],
                    torch.zeros_like(result["transport_delta_mm"]),
                    atol=1.0e-6,
                )
            )

    def test_identity_calibration_recovers_global_gaussian_translation(self):
        fixed, extent = self._represent(torch.rand(1, 1, 32, 32, 32))
        shift = torch.tensor([[[2.0, -1.0, 0.5]]])
        moving = [
            replace(level, centers_mm=level.centers_mm + shift)
            for level in fixed
        ]
        matcher = HierarchicalGaussianCorrespondence(
            feature_dim=24,
            position_weight=0.0,
            sinkhorn_iterations=5,
            parent_candidates=2,
            identity_calibration=True,
        )
        results = matcher(fixed, moving, extent)
        for result in results:
            self.assertTrue(
                torch.allclose(
                    result["transport_delta_mm"],
                    shift.expand_as(result["transport_delta_mm"]),
                    atol=2.0e-4,
                )
            )

    def test_v3_calibration_cancels_identity_value_and_gradient(self):
        fixed, extent = self._represent(torch.rand(1, 1, 32, 32, 32))
        matcher = HierarchicalGaussianCorrespondence(
            feature_dim=24,
            position_weight=0.6,
            sinkhorn_iterations=5,
            parent_candidates=2,
            identity_calibration=True,
            calibration_gradient=True,
            coordinate_mode="canonical",
            mutual_transport=True,
        )
        results = matcher(fixed, fixed, extent)
        identity_delta = sum(
            result["transport_delta_mm"].sum()
            for result in results
        )
        identity_delta.backward()
        self.assertLess(float(identity_delta.detach().abs()), 1.0e-6)
        for level_matcher in matcher.matchers:
            gradient = level_matcher.feature_projection.weight.grad
            self.assertIsNotNone(gradient)
            self.assertLess(float(gradient.abs().max()), 1.0e-6)

    def test_v3_transport_is_invariant_to_learned_center_drift(self):
        fixed, extent = self._represent(torch.rand(1, 1, 32, 32, 32))
        drift = torch.tensor([[[8.0, -4.0, 2.0]]])
        moving = [
            replace(level, centers_mm=level.centers_mm + drift)
            for level in fixed
        ]
        matcher = HierarchicalGaussianCorrespondence(
            feature_dim=24,
            position_weight=0.6,
            sinkhorn_iterations=5,
            parent_candidates=2,
            identity_calibration=True,
            calibration_gradient=True,
            coordinate_mode="canonical",
            mutual_transport=True,
        )
        results = matcher(fixed, moving, extent)
        for result in results:
            self.assertTrue(
                torch.allclose(
                    result["transport_delta_mm"],
                    torch.zeros_like(result["transport_delta_mm"]),
                    atol=1.0e-6,
                )
            )

    def test_v3_feature_correspondence_moves_between_canonical_anchors(self):
        fixed, extent = self._represent(torch.rand(1, 1, 32, 32, 32))
        moving = list(fixed)
        moving[0] = replace(
            fixed[0],
            features=torch.roll(fixed[0].features, shifts=1, dims=1),
        )
        matcher = HierarchicalGaussianCorrespondence(
            feature_dim=24,
            temperature=0.02,
            position_weight=0.0,
            dustbin_mass=0.0,
            sinkhorn_iterations=8,
            parent_candidates=8,
            identity_calibration=True,
            calibration_gradient=True,
            coordinate_mode="canonical",
            mutual_transport=True,
        )
        results = matcher(fixed, moving, extent)
        displacement = torch.linalg.vector_norm(
            results[0]["transport_delta_mm"],
            dim=-1,
        )
        self.assertGreater(float(displacement.detach().mean()), 1.0)

    def test_v4_stopped_anatomical_calibration_preserves_translation(self):
        fixed, extent = self._represent(torch.rand(1, 1, 32, 32, 32))
        shift = torch.tensor([[[3.0, -2.0, 1.0]]])
        moving = [
            replace(level, centers_mm=level.centers_mm + shift)
            for level in fixed
        ]
        matcher = HierarchicalGaussianCorrespondence(
            feature_dim=24,
            temperature=0.2,
            position_weight=0.0,
            dustbin_mass=0.0,
            sinkhorn_iterations=8,
            parent_candidates=8,
            identity_calibration=True,
            calibration_gradient=False,
            coordinate_mode="learned",
            mutual_transport=False,
            detach_geometry_cost=True,
        )
        results = matcher(fixed, moving, extent)
        for result in results:
            self.assertTrue(
                torch.allclose(
                    result["transport_delta_mm"],
                    shift.expand_as(result["transport_delta_mm"]),
                    atol=2.0e-4,
                )
            )

    def test_temperature_setter_updates_all_levels(self):
        self.matcher.set_temperature(0.17)
        self.assertAlmostEqual(self.matcher.temperature, 0.17)
        self.assertTrue(
            all(
                abs(level_matcher.temperature - 0.17) < 1.0e-12
                for level_matcher in self.matcher.matchers
            )
        )
        with self.assertRaises(ValueError):
            self.matcher.set_temperature(0.0)

    def test_v5_row_softmax_has_strict_mask_and_shared_calibration_support(self):
        fixed, extent = self._represent(torch.rand(1, 1, 32, 32, 32))
        moving, _ = self._represent(torch.rand(1, 1, 32, 32, 32))
        matcher = HierarchicalGaussianCorrespondence(
            feature_dim=24,
            temperature=0.1,
            position_weight=0.12,
            dustbin_mass=0.0,
            sinkhorn_iterations=3,
            parent_candidates=2,
            identity_calibration=True,
            calibration_gradient=False,
            coordinate_mode="learned",
            mutual_transport=False,
            detach_geometry_cost=True,
            appearance_weight=0.75,
            transport_mode="row_softmax",
            shared_calibration_candidates=True,
            include_identity_candidate=True,
        )
        results = matcher(fixed, moving, extent)
        for index, result in enumerate(results):
            plan = result["plan"]
            row_mass = plan.sum(dim=2)
            self.assertTrue(
                torch.allclose(
                    row_mass,
                    fixed[index].mass,
                    atol=1.0e-6,
                )
            )
            self.assertTrue(torch.isfinite(result["support_entropy"]).all())
            self.assertTrue(torch.all(result["support_entropy"] >= 0.0))
            self.assertTrue(torch.all(result["support_entropy"] <= 1.0))
            if index:
                mask = result["candidate_mask"]
                reference_mask = result["identity_candidate_mask"]
                self.assertTrue(torch.equal(mask, reference_mask))
                self.assertEqual(float(plan.masked_select(~mask).abs().max()), 0.0)
                diagonal = torch.diagonal(mask, dim1=1, dim2=2)
                self.assertTrue(bool(diagonal.all()))

    def test_v5_uniform_plan_has_zero_match_evidence(self):
        fixed, extent = self._represent(torch.rand(1, 1, 32, 32, 32))
        fixed = [
            replace(
                level,
                features=torch.zeros_like(level.features),
                appearance=torch.zeros_like(level.appearance),
            )
            for level in fixed
        ]
        shift = torch.tensor([[[2.0, -1.0, 0.5]]])
        moving = [
            replace(level, centers_mm=level.centers_mm + shift)
            for level in fixed
        ]
        matcher = HierarchicalGaussianCorrespondence(
            feature_dim=24,
            temperature=0.1,
            position_weight=0.0,
            scale_weight=0.0,
            dustbin_mass=0.0,
            parent_candidates=2,
            identity_calibration=True,
            calibration_gradient=False,
            coordinate_mode="learned",
            detach_geometry_cost=True,
            appearance_weight=0.75,
            transport_mode="row_softmax",
            shared_calibration_candidates=True,
            include_identity_candidate=True,
        )
        results = matcher(fixed, moving, extent)
        for result in results:
            self.assertTrue(
                torch.allclose(
                    result["support_entropy"],
                    torch.ones_like(result["support_entropy"]),
                    atol=1.0e-5,
                )
            )
            self.assertLess(float(result["match_evidence"].abs().max()), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
