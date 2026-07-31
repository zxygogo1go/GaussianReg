import unittest

import torch

from experiment_utils import (
    BaselineRegistrationObjective,
    BaselineSACBNet,
    appearance_weight_for_epoch,
    bootstrap_mean_ci,
    build_model,
    build_objective,
    config_architecture,
    configure_model_for_epoch,
    correspondence_temperature_for_epoch,
    feature_residual_weight_for_epoch,
    learning_rate_factor,
    warp_volume,
)
from gaussian_native import GaussianNativeObjective, GaussianNativeRegistration
from model import SACB_Net
from train_registration import (
    _augment_pair,
    _augment_pair_with_segmentations,
    _collapse_warning,
    _configure_training_stage,
    _fail_fast_reason,
    _make_synthetic_deformation_pair,
    _sample_flow_mm_at_centers,
    _supervised_anatomy_factor,
    _supervised_anatomy_loss,
    _synthetic_deformation_probability,
)


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

    def test_v4_builder_uses_anatomical_matching_and_stable_raster_basis(self):
        config = {
            "data": {
                "shape_dhw": [32, 32, 32],
                "spacing_dhw": [1.5, 1.5, 1.5],
            },
            "model": {
                "architecture_revision": "gaussian_native_v4",
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
                "match_temperature": 0.2,
                "match_temperature_end": 0.1,
                "match_temperature_anneal_epochs": 5,
            },
        }
        model = build_model(config)
        self.assertEqual(model.architecture_revision, "gaussian_native_v4")
        self.assertFalse(model.correspondence.calibration_gradient)
        self.assertEqual(model.correspondence.coordinate_mode, "learned")
        self.assertFalse(model.correspondence.matchers[0].mutual_transport)
        self.assertTrue(model.correspondence.matchers[0].detach_geometry_cost)
        self.assertTrue(model.velocity_synthesis.rasterizer.use_canonical_basis)
        self.assertAlmostEqual(configure_model_for_epoch(model, config, 1), 0.2)
        self.assertAlmostEqual(model.correspondence.temperature, 0.2)
        self.assertAlmostEqual(configure_model_for_epoch(model, config, 5), 0.1)
        self.assertAlmostEqual(model.correspondence.temperature, 0.1)

    def test_v5_builder_uses_appearance_row_matching_and_evidence(self):
        config = {
            "data": {
                "shape_dhw": [32, 32, 32],
                "spacing_dhw": [1.5, 1.5, 1.5],
            },
            "model": {
                "architecture_revision": "gaussian_native_v5",
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
                "transport_mode": "row_softmax",
                "appearance_weight": 0.75,
                "dustbin_mass": 0.0,
                "motion_mode": "translation",
            },
        }
        model = build_model(config)
        self.assertEqual(model.architecture_revision, "gaussian_native_v5")
        self.assertTrue(model.correspondence.shared_calibration_candidates)
        self.assertTrue(model.correspondence.include_identity_candidate)
        for matcher in model.correspondence.matchers:
            self.assertEqual(matcher.transport_mode, "row_softmax")
            self.assertAlmostEqual(matcher.appearance_weight, 0.75)
            self.assertTrue(matcher.detach_geometry_cost)
            self.assertFalse(matcher.mutual_transport)
        self.assertTrue(model.velocity_head.use_match_evidence)
        self.assertEqual(model.velocity_head.motion_mode, "translation")
        self.assertTrue(model.velocity_synthesis.rasterizer.use_canonical_basis)

    def test_v6_builder_removes_forced_identity_and_restores_fine_motion(self):
        config = {
            "data": {
                "shape_dhw": [32, 32, 32],
                "spacing_dhw": [1.5, 1.5, 1.5],
            },
            "model": {
                "architecture_revision": "gaussian_native_v6",
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
                "transport_mode": "row_softmax",
                "appearance_weight": 0.75,
                "appearance_weight_end": 0.25,
                "appearance_weight_anneal_epochs": 5,
                "dustbin_mass": 0.0,
                "include_identity_candidate": False,
                "match_evidence_power": 0.5,
                "motion_mode": "translation",
                "direct_displacement_fractions": [0.75, 1.0, 0.75],
            },
        }
        model = build_model(config)
        self.assertEqual(model.architecture_revision, "gaussian_native_v6")
        self.assertTrue(model.correspondence.shared_calibration_candidates)
        self.assertFalse(model.correspondence.include_identity_candidate)
        self.assertAlmostEqual(model.velocity_head.match_evidence_power, 0.5)
        self.assertEqual(
            model.velocity_head.direct_displacement_fractions,
            (0.75, 1.0, 0.75),
        )
        configure_model_for_epoch(model, config, 1)
        self.assertAlmostEqual(model.correspondence.appearance_weight, 0.75)
        configure_model_for_epoch(model, config, 5)
        self.assertAlmostEqual(model.correspondence.appearance_weight, 0.25)

    def test_v7_builder_uses_anchored_residual_pair_matching(self):
        config = {
            "data": {
                "shape_dhw": [32, 32, 32],
                "spacing_dhw": [1.5, 1.5, 1.5],
            },
            "model": {
                "architecture_revision": "gaussian_native_v7",
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
                "geometry_mode": "anchored",
                "transport_mode": "row_softmax",
                "correspondence_score_mode": "appearance_residual",
                "appearance_weight": 1.0,
                "feature_residual_weight": 0.1,
                "feature_residual_weight_end": 1.0,
                "feature_residual_weight_anneal_epochs": 5,
                "dustbin_mass": 0.0,
                "include_identity_candidate": False,
                "match_evidence_power": 0.5,
                "motion_mode": "translation",
            },
        }
        model = build_model(config)
        self.assertEqual(model.architecture_revision, "gaussian_native_v7")
        self.assertEqual(model.decomposer.geometry_mode, "anchored")
        self.assertIsNone(model.decomposer.root_geometry)
        self.assertEqual(model.correspondence.coordinate_mode, "canonical")
        self.assertFalse(model.correspondence.include_identity_candidate)
        for matcher in model.correspondence.matchers:
            self.assertEqual(matcher.score_mode, "appearance_residual")
            self.assertIsNotNone(matcher.pair_residual_score)
        configure_model_for_epoch(model, config, 1)
        self.assertAlmostEqual(
            model.correspondence.feature_residual_weight,
            0.1,
        )
        configure_model_for_epoch(model, config, 5)
        self.assertAlmostEqual(
            model.correspondence.feature_residual_weight,
            1.0,
        )

    def test_v8_correspondence_warmup_freezes_only_velocity_head(self):
        config = {
            "data": {
                "shape_dhw": [32, 32, 32],
                "spacing_dhw": [1.5, 1.5, 1.5],
            },
            "model": {
                "architecture_revision": "gaussian_native_v8",
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
                "geometry_mode": "anchored",
                "transport_mode": "row_softmax",
                "correspondence_score_mode": "appearance_residual",
                "appearance_weight": 1.0,
                "feature_residual_weight": 0.35,
                "dustbin_mass": 0.0,
                "include_identity_candidate": False,
                "motion_mode": "translation",
            },
            "optimization": {
                "correspondence_warmup_epochs": 2,
                "freeze_velocity_head_during_correspondence_warmup": True,
            },
        }
        model = build_model(config)
        self.assertEqual(model.architecture_revision, "gaussian_native_v8")
        self.assertEqual(model.decomposer.geometry_mode, "anchored")
        warmup = _configure_training_stage(model, config, 1)
        self.assertEqual(warmup["name"], "correspondence_warmup")
        self.assertFalse(warmup["velocity_head_trainable"])
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in model.velocity_head.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.correspondence.parameters()
            )
        )
        joint = _configure_training_stage(model, config, 3)
        self.assertEqual(joint["name"], "joint")
        self.assertTrue(joint["velocity_head_trainable"])
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.velocity_head.parameters()
            )
        )

    def test_v9_builder_uses_contextual_matching_capacity(self):
        config = {
            "data": {
                "shape_dhw": [32, 32, 32],
                "spacing_dhw": [1.5, 1.5, 1.5],
            },
            "model": {
                "architecture_revision": "gaussian_native_v9",
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
                "geometry_mode": "anchored",
                "transport_mode": "row_softmax",
                "correspondence_score_mode": "contextual_residual",
                "appearance_weight": 0.85,
                "feature_residual_weight": 0.15,
                "pair_score_hidden_dim": 32,
                "pair_context_dim": 32,
                "pair_score_heads": 4,
                "pair_fusion_hidden_dim": 48,
                "dustbin_mass": 0.0,
                "include_identity_candidate": False,
                "motion_mode": "translation",
            },
            "synthetic_deformation": {
                "enabled": True,
                "warmup_epochs": 2,
                "probability_after_warmup": 0.25,
                "active_until_epoch": 4,
            },
        }
        model = build_model(config)
        self.assertEqual(
            model.architecture_revision,
            "gaussian_native_v9",
        )
        for matcher in model.correspondence.matchers:
            self.assertEqual(
                matcher.score_mode,
                "contextual_residual",
            )
            self.assertEqual(matcher.context_dim, 32)
            self.assertEqual(matcher.pair_score_heads, 4)
        warmup = _configure_training_stage(model, config, 1)
        self.assertEqual(
            warmup["name"],
            "synthetic_deformation_warmup",
        )
        self.assertTrue(warmup["velocity_head_trainable"])
        self.assertEqual(
            warmup["synthetic_deformation_probability"],
            1.0,
        )
        mixed = _configure_training_stage(model, config, 3)
        self.assertEqual(
            mixed["name"],
            "joint_with_synthetic_deformation",
        )
        self.assertEqual(
            mixed["synthetic_deformation_probability"],
            0.25,
        )

    def test_synthetic_deformation_schedule_and_pair_are_exact(self):
        schedule = {
            "enabled": True,
            "warmup_epochs": 2,
            "probability_after_warmup": 0.25,
            "active_until_epoch": 4,
        }
        self.assertEqual(
            _synthetic_deformation_probability(schedule, 1),
            1.0,
        )
        self.assertEqual(
            _synthetic_deformation_probability(schedule, 3),
            0.25,
        )
        self.assertEqual(
            _synthetic_deformation_probability(schedule, 5),
            0.0,
        )
        torch.manual_seed(23)
        moving = torch.rand(1, 1, 16, 16, 16)
        fixed = torch.rand_like(moving)
        spacing = torch.tensor([1.5, 1.5, 1.5])
        synthetic_moving, synthetic_fixed, target = (
            _make_synthetic_deformation_pair(
                moving,
                fixed,
                spacing,
                {
                    "control_grid_shape": [3, 3, 3],
                    "local_velocity_mm_range": [1.0, 1.0],
                    "translation_velocity_mm_range": [0.0, 0.0],
                    "integration_steps": 3,
                },
            )
        )
        reconstructed = warp_volume(synthetic_moving, target)
        self.assertTrue(
            torch.allclose(
                synthetic_fixed,
                reconstructed,
                atol=1.0e-5,
            )
        )
        self.assertGreater(float(target.abs().max()), 0.0)

        constant_flow = torch.zeros(1, 3, 8, 8, 8)
        constant_flow[:, 0] = 2.0
        centers = torch.tensor(
            [[[0.0, 0.0, 0.0], [10.5, 10.5, 10.5]]]
        )
        sampled = _sample_flow_mm_at_centers(
            constant_flow,
            spacing,
            centers,
            torch.tensor([[10.5, 10.5, 10.5]]),
        )
        expected = torch.tensor(
            [[[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]
        )
        self.assertTrue(torch.allclose(sampled, expected))

    def test_v10_builder_enables_true_residual_pyramid(self):
        config = {
            "data": {
                "shape_dhw": [32, 32, 32],
                "spacing_dhw": [1.5, 1.5, 1.5],
            },
            "model": {
                "architecture_revision": "gaussian_native_v10",
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
                "geometry_mode": "anchored",
                "transport_mode": "row_softmax",
                "correspondence_score_mode": "contextual_residual",
                "appearance_weight": 0.8,
                "feature_residual_weight": 0.1,
                "pair_score_hidden_dim": 32,
                "pair_context_dim": 32,
                "pair_score_heads": 4,
                "pair_fusion_hidden_dim": 48,
                "dustbin_mass": 0.0,
                "include_identity_candidate": False,
                "motion_mode": "translation",
                "refinement_factors": [8, 4, 2],
                "refinement_channels": [8, 8, 8],
                "refinement_blocks_per_stage": 1,
                "refinement_maximum_residual_vox": [1.0, 1.0, 1.0],
            },
        }
        model = build_model(config)
        self.assertEqual(
            model.architecture_revision,
            "gaussian_native_v10",
        )
        self.assertIsNotNone(model.residual_pyramid)
        moving = torch.rand(1, 1, 32, 32, 32)
        output = model(moving, moving.clone(), return_aux=True)
        self.assertEqual(output["pyramid_factors"], (8, 4, 2))
        self.assertEqual(len(output["pyramid_flow"]), 3)
        self.assertEqual(tuple(output["flow"].shape[2:]), (32, 32, 32))

    def test_response_aware_supervised_anatomy_loss(self):
        moving_seg = torch.zeros(1, 1, 16, 16, 16, dtype=torch.long)
        fixed_seg = torch.zeros_like(moving_seg)
        moving_seg[:, :, 4:9, 4:9, 4:9] = 1
        fixed_seg.copy_(moving_seg)
        flow = torch.zeros(1, 3, 16, 16, 16)
        output = {
            "flow": flow,
            "inverse_flow": flow.clone(),
            "pyramid_flow": [
                torch.zeros(1, 3, 2, 2, 2),
                torch.zeros(1, 3, 4, 4, 4),
                torch.zeros(1, 3, 8, 8, 8),
            ],
        }
        config = {
            "enabled": True,
            "labels": [1, 2],
            "deep_supervision_weights": [0.1, 0.2, 0.3, 0.4],
            "weight_factor_start": 0.2,
            "weight_factor_end": 1.0,
            "weight_ramp_epochs": 3,
        }
        terms = _supervised_anatomy_loss(
            output,
            moving_seg,
            fixed_seg,
            torch.tensor([[True, False]]),
            config,
        )
        self.assertLess(float(terms["supervised_dice"]), 1.0e-5)
        self.assertLess(
            float(terms["supervised_boundary"]),
            1.0e-5,
        )
        self.assertLess(
            float(terms["supervised_centroid"]),
            1.0e-5,
        )
        self.assertLess(
            float(terms["supervised_inverse_dice"]),
            1.0e-5,
        )
        self.assertAlmostEqual(
            float(terms["supervised_final_dice"]),
            1.0,
            places=5,
        )
        self.assertAlmostEqual(
            _supervised_anatomy_factor(config, 1),
            0.2,
        )
        self.assertAlmostEqual(
            _supervised_anatomy_factor(config, 3),
            1.0,
        )

    def test_v11_builder_adds_full_resolution_gradient_stage(self):
        config = {
            "data": {
                "shape_dhw": [32, 32, 32],
                "spacing_dhw": [1.5, 1.5, 1.5],
            },
            "model": {
                "architecture_revision": "gaussian_native_v11",
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
                "geometry_mode": "anchored",
                "transport_mode": "row_softmax",
                "correspondence_score_mode": "contextual_residual",
                "appearance_weight": 0.8,
                "feature_residual_weight": 0.1,
                "pair_score_hidden_dim": 32,
                "pair_context_dim": 32,
                "pair_score_heads": 4,
                "pair_fusion_hidden_dim": 48,
                "dustbin_mass": 0.0,
                "include_identity_candidate": False,
                "motion_mode": "translation",
                "refinement_factors": [8, 4, 2, 1],
                "refinement_channels": [8, 8, 8, 8],
                "refinement_blocks_per_stage": [1, 1, 1, 1],
                "refinement_maximum_residual_vox": [
                    1.0,
                    1.0,
                    1.0,
                    0.5,
                ],
                "refinement_use_gradient_features": True,
            },
        }
        model = build_model(config)
        self.assertEqual(
            model.architecture_revision,
            "gaussian_native_v11",
        )
        self.assertEqual(model.residual_pyramid.factors, (8, 4, 2, 1))
        self.assertTrue(
            model.residual_pyramid.stages[-1].use_gradient_features
        )
        moving = torch.rand(1, 1, 32, 32, 32)
        output = model(moving, moving.clone(), return_aux=True)
        self.assertEqual(len(output["pyramid_flow"]), 4)
        self.assertEqual(
            tuple(output["pyramid_flow"][-1].shape[2:]),
            (32, 32, 32),
        )

    def test_correspondence_temperature_schedule_boundaries(self):
        config = {
            "model": {
                "match_temperature": 0.2,
                "match_temperature_end": 0.1,
                "match_temperature_anneal_epochs": 5,
            }
        }
        self.assertAlmostEqual(
            correspondence_temperature_for_epoch(config, 1),
            0.2,
        )
        self.assertAlmostEqual(
            correspondence_temperature_for_epoch(config, 3),
            0.15,
        )
        self.assertAlmostEqual(
            correspondence_temperature_for_epoch(config, 5),
            0.1,
        )
        self.assertAlmostEqual(
            correspondence_temperature_for_epoch(config, 50),
            0.1,
        )
        with self.assertRaises(ValueError):
            correspondence_temperature_for_epoch(config, 0)

    def test_appearance_weight_schedule_boundaries(self):
        config = {
            "model": {
                "appearance_weight": 0.75,
                "appearance_weight_end": 0.25,
                "appearance_weight_anneal_epochs": 5,
            }
        }
        self.assertAlmostEqual(appearance_weight_for_epoch(config, 1), 0.75)
        self.assertAlmostEqual(appearance_weight_for_epoch(config, 3), 0.5)
        self.assertAlmostEqual(appearance_weight_for_epoch(config, 5), 0.25)
        self.assertAlmostEqual(appearance_weight_for_epoch(config, 50), 0.25)
        with self.assertRaises(ValueError):
            appearance_weight_for_epoch(config, 0)

    def test_feature_residual_weight_schedule_boundaries(self):
        config = {
            "model": {
                "feature_residual_weight": 0.1,
                "feature_residual_weight_end": 1.0,
                "feature_residual_weight_anneal_epochs": 5,
            }
        }
        self.assertAlmostEqual(
            feature_residual_weight_for_epoch(config, 1),
            0.1,
        )
        self.assertAlmostEqual(
            feature_residual_weight_for_epoch(config, 3),
            0.55,
        )
        self.assertAlmostEqual(
            feature_residual_weight_for_epoch(config, 5),
            1.0,
        )
        self.assertAlmostEqual(
            feature_residual_weight_for_epoch(config, 50),
            1.0,
        )
        with self.assertRaises(ValueError):
            feature_residual_weight_for_epoch(config, 0)

    def test_identity_collapse_warning_requires_joint_signature(self):
        train = {
            "match_entropy_l0": 0.01,
            "diagonal_probability_l0": 0.99,
            "transport_delta_l0_mm": 0.01,
        }
        validation = {"ncc_improvement": 0.01}
        self.assertEqual(_collapse_warning(9, train, validation), "")
        self.assertIn("identity", _collapse_warning(10, train, validation))
        healthy_motion = dict(train)
        healthy_motion["transport_delta_l0_mm"] = 2.0
        self.assertEqual(
            _collapse_warning(10, healthy_motion, validation),
            "",
        )

    def test_fail_fast_monitor_detects_repeated_negative_validation(self):
        history = [
            {
                "train": {
                    "support_entropy_l0": 0.9,
                    "row_max_probability_l0": 0.1,
                },
                "validation": {
                    "ncc_after": 0.18,
                    "ncc_improvement": -0.03,
                    "dice_improvement": -0.01,
                    "negative_jacobian_ratio": 0.0,
                    "p95_displacement_mm": 5.0,
                },
            }
            for _ in range(3)
        ]
        config = {
            "monitoring": {
                "fail_fast_enabled": True,
                "fail_fast_start_epoch": 3,
                "fail_fast_patience": 3,
                "recent_mean_ncc_floor": -0.02,
            }
        }
        self.assertIn(
            "mean NCC improvement",
            _fail_fast_reason(3, history, config),
        )
        config["monitoring"]["fail_fast_enabled"] = False
        self.assertEqual(_fail_fast_reason(3, history, config), "")

    def test_fail_fast_monitor_detects_fine_gaussian_drift(self):
        history = [
            {
                "train": {
                    "anchor_offset_l2": 1.2,
                    "support_entropy_l0": 0.5,
                    "row_max_probability_l0": 0.2,
                },
                "validation": {
                    "ncc_after": 0.3,
                    "ncc_improvement": 0.01,
                    "dice_improvement": 0.0,
                    "negative_jacobian_ratio": 0.0,
                    "p95_displacement_mm": 5.0,
                },
            }
        ]
        config = {
            "monitoring": {
                "fail_fast_enabled": True,
                "maximum_anchor_offset_l2": 1.0,
            }
        }
        self.assertIn(
            "anchor offset",
            _fail_fast_reason(1, history, config),
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

    def test_supervised_flip_keeps_images_and_labels_aligned(self):
        moving = torch.zeros(1, 1, 4, 4, 4)
        fixed = torch.zeros_like(moving)
        moving[:, :, 1, 2, 0] = 1.0
        fixed[:, :, 2, 1, 1] = 1.0
        moving_seg = moving.long()
        fixed_seg = fixed.long()
        (
            augmented_moving,
            augmented_fixed,
            augmented_moving_seg,
            augmented_fixed_seg,
        ) = _augment_pair_with_segmentations(
            moving,
            fixed,
            {
                "enabled": True,
                "reverse_pair_probability": 0.0,
                "shared_flip_probability": 1.0,
                "intensity_probability": 0.0,
            },
            moving_seg,
            fixed_seg,
        )
        self.assertTrue(
            torch.equal(
                augmented_moving.round().long(),
                augmented_moving_seg,
            )
        )
        self.assertTrue(
            torch.equal(
                augmented_fixed.round().long(),
                augmented_fixed_seg,
            )
        )
        self.assertTrue(
            torch.equal(
                augmented_moving,
                torch.flip(moving, dims=(-1,)),
            )
        )


if __name__ == "__main__":
    unittest.main()
