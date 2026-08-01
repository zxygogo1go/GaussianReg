"""One-step production-shape forward/backward and memory audit."""

from __future__ import annotations

import argparse
import json
import time

import torch

from experiment_utils import (
    build_model,
    build_objective,
    cuda_autocast,
    load_json,
    output_diagnostics,
    resolve_device,
    set_reproducibility,
)
from train_registration import (
    _make_synthetic_deformation_pair,
    _supervised_anatomy_factor,
    _supervised_anatomy_loss,
    _synthetic_supervision,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/gaussian_native_v12_hntsmrg24.json",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_json(args.config)
    device = resolve_device(args.device)
    seed = int(config.get("seed", 2026))
    set_reproducibility(seed)
    shape = tuple(
        int(value)
        for value in dict(config.get("data", {})).get(
            "shape_dhw",
            (128, 160, 160),
        )
    )
    optimization = dict(config.get("optimization", {}))
    amp = bool(optimization.get("amp", True)) and device.type == "cuda"
    amp_dtype = str(optimization.get("amp_dtype", "bfloat16")).strip().lower()
    amp_cache_enabled = bool(
        optimization.get("amp_cache_enabled", False)
    )
    model = build_model(config).to(device).train()
    objective = build_objective(config).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    moving = torch.rand((1, 1, *shape), generator=generator, device=device)
    fixed = torch.rand((1, 1, *shape), generator=generator, device=device)
    synthetic_config = dict(
        config.get("synthetic_deformation", {})
    )
    target_flow = None
    if bool(synthetic_config.get("enabled", False)):
        moving, fixed, target_flow = _make_synthetic_deformation_pair(
            moving,
            fixed,
            model.spacing_dhw,
            synthetic_config,
        )
    supervised_config = dict(
        config.get("supervised_anatomy", {})
    )
    moving_seg = None
    fixed_seg = None
    response_valid = None
    if bool(supervised_config.get("enabled", False)):
        moving_seg = torch.zeros(
            (1, 1, *shape),
            device=device,
            dtype=torch.long,
        )
        fixed_seg = torch.zeros_like(moving_seg)
        d0, h0, w0 = (value // 3 for value in shape)
        d1, h1, w1 = (2 * value // 3 for value in shape)
        moving_seg[
            :,
            :,
            d0:d1,
            h0:h1,
            w0:w1,
        ] = 1
        fixed_seg[
            :,
            :,
            d0 + 1:d1 + 1,
            h0:h1,
            w0:w1,
        ] = 1
        moving_seg[
            :,
            :,
            d0:d1,
            h0 // 2:h0,
            w0 // 2:w0,
        ] = 2
        fixed_seg[
            :,
            :,
            d0:d1,
            h0 // 2 + 1:h0 + 1,
            w0 // 2:w0,
        ] = 2
        response_valid = torch.ones(
            1,
            2,
            device=device,
            dtype=torch.bool,
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with cuda_autocast(
        amp,
        amp_dtype,
        cache_enabled=amp_cache_enabled,
    ):
        output = model(moving, fixed, return_aux=True)
        terms = objective(output, moving, fixed)
        if target_flow is not None:
            terms = dict(terms)
            synthetic_terms = _synthetic_supervision(
                output,
                target_flow,
                model.spacing_dhw,
            )
            terms.update(synthetic_terms)
            terms["total"] = (
                terms["total"]
                + float(
                    synthetic_config.get(
                        "flow_loss_weight",
                        0.0,
                    )
                )
                * terms["synthetic_flow"]
                + float(
                    synthetic_config.get(
                        "correspondence_loss_weight",
                        0.0,
                    )
                )
                * terms["synthetic_correspondence"]
            )
        if moving_seg is not None:
            supervised_terms = _supervised_anatomy_loss(
                output,
                moving_seg,
                fixed_seg,
                response_valid,
                supervised_config,
            )
            terms.update(supervised_terms)
            factor = _supervised_anatomy_factor(
                supervised_config,
                epoch=1,
            )
            terms["total"] = (
                terms["total"]
                + factor
                * (
                    float(
                        supervised_config.get(
                            "dice_loss_weight",
                            0.0,
                        )
                    )
                    * terms["supervised_dice"]
                    + float(
                        supervised_config.get(
                            "boundary_loss_weight",
                            0.0,
                        )
                    )
                    * terms["supervised_boundary"]
                    + float(
                        supervised_config.get(
                            "centroid_loss_weight",
                            0.0,
                        )
                    )
                    * terms["supervised_centroid"]
                    + float(
                        supervised_config.get(
                            "inverse_dice_loss_weight",
                            0.0,
                        )
                    )
                    * terms["supervised_inverse_dice"]
                )
            )
    terms["total"].backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    finite = bool(torch.isfinite(terms["total"]).all()) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    pair_residual_gradient_l1 = []
    for matcher in getattr(
        getattr(model, "correspondence", None),
        "matchers",
        (),
    ):
        scorer = getattr(matcher, "pair_residual_score", None)
        if scorer is None:
            continue
        gradient = scorer[-1].weight.grad
        pair_residual_gradient_l1.append(
            None
            if gradient is None
            else float(gradient.detach().float().abs().sum().cpu())
        )
    pair_residual_trainable = (
        not pair_residual_gradient_l1
        or all(
            value is not None and value > 0.0
            for value in pair_residual_gradient_l1
        )
    )
    result = {
        "device": str(device),
        "shape_dhw": list(shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "gradient_tensors": len(gradients),
        "finite": finite,
        "amp_cache_enabled": amp_cache_enabled,
        "pair_residual_gradient_l1": pair_residual_gradient_l1,
        "pair_residual_trainable": pair_residual_trainable,
        "seconds": seconds,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
            if device.type == "cuda"
            else 0.0
        ),
        "losses": {
            name: float(value.detach().float().cpu())
            for name, value in terms.items()
        },
        "diagnostics": {
            name: value
            for name, value in output_diagnostics(output).items()
            if name == "velocity_vox_abs"
            or name.startswith(
                (
                    "match_entropy_",
                    "support_entropy_",
                    "match_evidence_",
                    "motion_evidence_",
                    "feature_residual_logit_",
                    "context_attention_concentration_",
                    "row_max_probability_",
                    "support_size_",
                    "matched_mass_",
                    "real_transport_mass_",
                    "unmatched_fixed_mass_",
                    "unmatched_moving_mass_",
                    "marginal_error_",
                    "diagonal_probability_",
                    "transport_delta_",
                    "direct_translation_",
                    "learned_translation_",
                    "pyramid_residual_",
                    "pyramid_flow_",
                )
            )
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not finite:
        raise FloatingPointError("production smoke test produced non-finite values")
    if not pair_residual_trainable:
        raise RuntimeError(
            "pair residual scorer is disconnected from the training objective"
        )


if __name__ == "__main__":
    main()
