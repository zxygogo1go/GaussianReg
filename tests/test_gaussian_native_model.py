import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch

from experiment_utils import build_model, build_objective, cuda_autocast
from gaussian_native import GaussianNativeObjective, GaussianNativeRegistration
from gaussian_native.integration import ScalingAndSquaring, compose_displacements
from gaussian_native.velocity import GaussianVelocityHead
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
    def test_v13_end_to_end_composition_and_backward(self):
        config = small_config()
        config["model"].update(
            {
                "architecture_revision": "gaussian_native_v13",
                "geometry_mode": "anchored",
                "transport_mode": "unbalanced_sinkhorn",
                "marginal_relaxation": 0.9,
                "mutual_transport": True,
                "refinement_factors": [8, 4, 2, 1],
                "refinement_channels": [8, 8, 8, 8],
                "refinement_blocks_per_stage": [1, 1, 1, 1],
                "refinement_maximum_residual_vox": [1.0, 1.0, 1.0, 0.5],
                "small_organ_selected_parents": 4,
                "small_organ_children_per_parent": 9,
                "small_organ_descriptor_dim": 8,
                "small_organ_hidden_dim": 16,
                "small_organ_raster_chunk": 4,
            }
        )
        model = build_model(config)
        moving = torch.rand(1, 1, 32, 32, 32)
        fixed = torch.rand_like(moving)
        output = model(moving, fixed, return_aux=True)
        small = output["small_organ_refinement"]
        self.assertEqual(tuple(output["flow"].shape), (1, 3, 32, 32, 32))
        self.assertEqual(float(small["local_flow"].abs().max()), 0.0)
        self.assertTrue(
            torch.allclose(output["flow"], output["global_flow"], atol=1.0e-5)
        )
        loss = (output["warped"] - fixed).square().mean()
        loss = loss + small["priority"].mean()
        loss.backward()
        self.assertGreater(
            float(model.small_organ_refiner.velocity_head[-1].weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(model.small_organ_refiner.priority_head[-1].weight.grad.abs().sum()),
            0.0,
        )

    def test_v3_canonical_rasterization_ignores_learned_geometry_drift(self):
        model = build_model(small_config())
        volume = torch.rand(1, 1, 32, 32, 32)
        with torch.no_grad():
            decomposition = model.decomposer(
                volume,
                torch.tensor([[1.5, 1.5, 1.5]]),
                compute_reconstruction=False,
            )
        level = decomposition["levels"][0]
        nodes = int(level.centers_mm.shape[1])
        translation = torch.randn(1, nodes, 3)
        motion = SimpleNamespace(
            translation_mm=translation,
            linear=torch.zeros(1, nodes, 3, 3),
        )
        rasterizer = model.velocity_synthesis.rasterizer
        field, _ = rasterizer(
            level,
            motion,
            output_shape=(8, 8, 8),
            extent_mm=decomposition["extent_mm"],
        )
        drifted = replace(
            level,
            centers_mm=level.centers_mm + 25.0,
            mass=torch.softmax(torch.randn_like(level.mass), dim=1),
        )
        drifted_field, _ = rasterizer(
            drifted,
            motion,
            output_shape=(8, 8, 8),
            extent_mm=decomposition["extent_mm"],
        )
        self.assertTrue(torch.allclose(field, drifted_field, atol=1.0e-6))

    def test_v2_child_transport_is_additive_not_hard_centered(self):
        feature_dim = 8
        head = GaussianVelocityHead(
            feature_dim=feature_dim,
            hidden_dim=16,
            children_per_parent=4,
            motion_mode="translation",
            hierarchy_mode="soft_residual",
            direct_displacement_fractions=(1.0, 1.0, 1.0),
        )
        counts = (1, 4, 16)
        parent_indices = (
            None,
            torch.zeros(4, dtype=torch.long),
            torch.arange(4).repeat_interleave(4),
        )
        levels = []
        matches = []
        absolute_deltas = []
        root_delta = torch.tensor([[[2.0, 0.0, 0.0]]])
        middle_delta = root_delta[:, parent_indices[1]] + torch.tensor(
            [[[0.0, 1.0, 0.0]]]
        )
        fine_delta = middle_delta[:, parent_indices[2]] + torch.tensor(
            [[[0.0, 0.0, 0.5]]]
        )
        absolute_deltas.extend((root_delta, middle_delta, fine_delta))
        for count, parent_index, delta in zip(
            counts,
            parent_indices,
            absolute_deltas,
        ):
            centers = torch.zeros(1, count, 3)
            scales = torch.full_like(centers, 10.0)
            covariance = torch.eye(3).reshape(1, 1, 3, 3).expand(1, count, -1, -1)
            features = torch.zeros(1, count, feature_dim)
            levels.append(
                SimpleNamespace(
                    centers_mm=centers,
                    scales_mm=scales,
                    precision_mm2=covariance,
                    features=features,
                    parent_index=parent_index,
                )
            )
            matches.append(
                {
                    "matched_center_mm": centers,
                    "matched_scale_mm": scales,
                    "matched_covariance_mm2": covariance,
                    "matched_feature": features,
                    "transport_delta_mm": delta,
                }
            )
        parameters = head(levels, matches)
        expected = (
            torch.tensor([2.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 0.5]),
        )
        for motion, target in zip(parameters, expected):
            self.assertTrue(
                torch.allclose(
                    motion.direct_translation_mm,
                    target.reshape(1, 1, 3).expand_as(
                        motion.direct_translation_mm
                    ),
                    atol=1.0e-6,
                )
            )
        self.assertGreater(
            float(
                parameters[1]
                .translation_mm.detach()
                .mean(dim=1)
                .abs()
                .sum()
            ),
            0.0,
        )

        bounded_head = GaussianVelocityHead(
            feature_dim=feature_dim,
            hidden_dim=16,
            children_per_parent=4,
            motion_mode="translation",
            hierarchy_mode="soft_residual",
            direct_displacement_fractions=(1.0, 1.0, 1.0),
            direct_displacement_limits_mm=(2.0, 1.0, 0.5),
            learned_translation_fractions=(0.0, 0.0, 0.0),
        )
        large_matches = []
        for match in matches:
            large_match = dict(match)
            large_match["transport_delta_mm"] = (
                10.0 * match["transport_delta_mm"]
            )
            large_matches.append(large_match)
        bounded = bounded_head(levels, large_matches)
        for motion, limit in zip(bounded, (2.0, 1.0, 0.5)):
            norm = torch.linalg.vector_norm(
                motion.direct_translation_mm,
                dim=-1,
            )
            self.assertLessEqual(float(norm.detach().max()), limit + 1.0e-5)

    def test_v5_uniform_child_match_cannot_cancel_parent_motion(self):
        feature_dim = 8
        head = GaussianVelocityHead(
            feature_dim=feature_dim,
            hidden_dim=16,
            children_per_parent=4,
            motion_mode="translation",
            hierarchy_mode="soft_residual",
            direct_displacement_fractions=(1.0, 1.0, 1.0),
            direct_displacement_limits_mm=(20.0, 20.0, 20.0),
            learned_translation_fractions=(0.0, 0.0, 0.0),
            use_match_evidence=True,
        )
        counts = (1, 4, 16)
        parent_indices = (
            None,
            torch.zeros(4, dtype=torch.long),
            torch.arange(4).repeat_interleave(4),
        )
        absolute_deltas = (
            torch.tensor([[[2.0, 0.0, 0.0]]]),
            torch.tensor([[[3.0, 1.0, 0.0]]]).expand(1, 4, 3),
            torch.tensor([[[4.0, 2.0, 1.0]]]).expand(1, 16, 3),
        )
        levels = []
        matches = []
        for level_index, (count, parent_index, delta) in enumerate(
            zip(counts, parent_indices, absolute_deltas)
        ):
            centers = torch.zeros(1, count, 3)
            scales = torch.full_like(centers, 10.0)
            covariance = (
                torch.eye(3)
                .reshape(1, 1, 3, 3)
                .expand(1, count, -1, -1)
            )
            features = torch.zeros(1, count, feature_dim)
            levels.append(
                SimpleNamespace(
                    centers_mm=centers,
                    scales_mm=scales,
                    precision_mm2=covariance,
                    features=features,
                    parent_index=parent_index,
                )
            )
            matches.append(
                {
                    "matched_center_mm": centers + delta,
                    "matched_scale_mm": scales,
                    "matched_covariance_mm2": covariance,
                    "matched_feature": features,
                    "transport_delta_mm": delta,
                    "match_evidence": torch.ones(1, count)
                    if level_index == 0
                    else torch.zeros(1, count),
                }
            )
        parameters = head(levels, matches)
        self.assertTrue(
            torch.allclose(
                parameters[0].direct_translation_mm,
                absolute_deltas[0],
                atol=1.0e-6,
            )
        )
        self.assertEqual(
            float(parameters[1].direct_translation_mm.abs().max()),
            0.0,
        )
        self.assertEqual(
            float(parameters[2].direct_translation_mm.abs().max()),
            0.0,
        )

    def test_v6_square_root_evidence_restores_child_residual_capacity(self):
        feature_dim = 8
        head = GaussianVelocityHead(
            feature_dim=feature_dim,
            hidden_dim=16,
            children_per_parent=4,
            motion_mode="translation",
            hierarchy_mode="soft_residual",
            direct_displacement_fractions=(1.0, 1.0, 1.0),
            direct_displacement_limits_mm=(20.0, 20.0, 20.0),
            learned_translation_fractions=(0.0, 0.0, 0.0),
            use_match_evidence=True,
            match_evidence_power=0.5,
        )
        parent_indices = (
            None,
            torch.zeros(4, dtype=torch.long),
            torch.arange(4).repeat_interleave(4),
        )
        absolute_deltas = (
            torch.tensor([[[2.0, 0.0, 0.0]]]),
            torch.tensor([[[2.0, 2.0, 0.0]]]).expand(1, 4, 3),
            torch.tensor([[[2.0, 2.0, 1.0]]]).expand(1, 16, 3),
        )
        evidences = (
            torch.ones(1, 1),
            torch.full((1, 4), 0.25),
            torch.zeros(1, 16),
        )
        levels = []
        matches = []
        for count, parent_index, delta, evidence in zip(
            (1, 4, 16),
            parent_indices,
            absolute_deltas,
            evidences,
        ):
            centers = torch.zeros(1, count, 3)
            scales = torch.full_like(centers, 10.0)
            covariance = (
                torch.eye(3)
                .reshape(1, 1, 3, 3)
                .expand(1, count, -1, -1)
            )
            features = torch.zeros(1, count, feature_dim)
            levels.append(
                SimpleNamespace(
                    centers_mm=centers,
                    scales_mm=scales,
                    precision_mm2=covariance,
                    features=features,
                    parent_index=parent_index,
                )
            )
            matches.append(
                {
                    "matched_center_mm": centers + delta,
                    "matched_scale_mm": scales,
                    "matched_covariance_mm2": covariance,
                    "matched_feature": features,
                    "transport_delta_mm": delta,
                    "match_evidence": evidence,
                }
            )
        parameters = head(levels, matches)
        expected_child = torch.tensor([0.0, 1.0, 0.0])
        self.assertTrue(
            torch.allclose(
                parameters[1].direct_translation_mm,
                expected_child.reshape(1, 1, 3).expand(1, 4, 3),
                atol=1.0e-6,
            )
        )
        self.assertEqual(
            float(parameters[2].direct_translation_mm.abs().max()),
            0.0,
        )

    def test_v5_similarity_gradient_reaches_learned_correspondence(self):
        config = small_config()
        config["model"].update(
            {
                "architecture_revision": "gaussian_native_v5",
                "transport_mode": "row_softmax",
                "appearance_weight": 0.75,
                "dustbin_mass": 0.0,
                "motion_mode": "translation",
                "direct_displacement_limits_mm": [8.0, 3.0, 1.5],
            }
        )
        config["loss"].update(
            {
                "similarity": 1.0,
                "representation": 0.0,
                "correspondence": 0.0,
                "deformation": 0.0,
                "ngf_weight": 0.0,
            }
        )
        model = build_model(config)
        objective = build_objective(config)
        moving = torch.rand(1, 1, 32, 32, 32)
        fixed = torch.rand_like(moving)
        output = model(moving, fixed, return_aux=True)
        terms = objective(output, moving, fixed)
        terms["total"].backward()
        projection_gradient = (
            model.correspondence.matchers[0].feature_projection.weight.grad
        )
        encoder_gradients = [
            parameter.grad
            for parameter in model.encoder.parameters()
            if parameter.grad is not None
        ]
        self.assertIsNotNone(projection_gradient)
        self.assertGreater(float(projection_gradient.abs().sum()), 0.0)
        self.assertTrue(encoder_gradients)
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in encoder_gradients),
            0.0,
        )

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for the autocast cache regression",
    )
    def test_v7_bfloat16_updates_pair_residual_scorer(self):
        config = small_config()
        config["model"].update(
            {
                "architecture_revision": "gaussian_native_v7",
                "geometry_mode": "anchored",
                "transport_mode": "row_softmax",
                "correspondence_score_mode": "appearance_residual",
                "appearance_weight": 1.0,
                "feature_residual_weight": 0.2,
                "max_feature_residual_logit": 2.0,
                "pair_score_hidden_dim": 16,
                "dustbin_mass": 0.0,
                "motion_mode": "translation",
                "include_identity_candidate": False,
                "direct_displacement_limits_mm": [8.0, 3.0, 1.5],
            }
        )
        config["loss"].update(
            {
                "similarity": 1.0,
                "representation": 0.0,
                "correspondence": 0.0,
                "deformation": 0.0,
                "ngf_weight": 0.0,
            }
        )
        device = torch.device("cuda:0")
        model = build_model(config).to(device).train()
        objective = build_objective(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        moving = torch.rand(1, 1, 32, 32, 32, device=device)
        fixed = torch.rand_like(moving)
        before = [
            matcher.pair_residual_score[-1].weight.detach().clone()
            for matcher in model.correspondence.matchers
        ]
        with cuda_autocast(True, "bfloat16"):
            output = model(moving, fixed, return_aux=True)
            terms = objective(output, moving, fixed)
        terms["total"].backward()
        for matcher in model.correspondence.matchers:
            gradient = matcher.pair_residual_score[-1].weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)
        optimizer.step()
        for previous, matcher in zip(
            before,
            model.correspondence.matchers,
        ):
            current = matcher.pair_residual_score[-1].weight.detach()
            self.assertGreater(float((current - previous).abs().sum()), 0.0)

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
