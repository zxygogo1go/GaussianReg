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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/gaussian_native_v5_hntsmrg24.json",
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
    model = build_model(config).to(device).train()
    objective = build_objective(config).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    moving = torch.rand((1, 1, *shape), generator=generator, device=device)
    fixed = torch.rand((1, 1, *shape), generator=generator, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with cuda_autocast(amp, amp_dtype):
        output = model(moving, fixed, return_aux=True)
        terms = objective(output, moving, fixed)
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
    result = {
        "device": str(device),
        "shape_dhw": list(shape),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "gradient_tensors": len(gradients),
        "finite": finite,
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
                    "row_max_probability_",
                    "support_size_",
                    "diagonal_probability_",
                    "transport_delta_",
                    "direct_translation_",
                    "learned_translation_",
                )
            )
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not finite:
        raise FloatingPointError("production smoke test produced non-finite values")


if __name__ == "__main__":
    main()
