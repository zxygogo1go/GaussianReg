# Anatomy-aware plug-in local residual refinement

This package separates the former MUSA+ Stage-3 method from the DIR-MUSA
three-stage pipeline. It can refine the output of any 3-D registration model
that provides a warped moving image and an additive DVF.

## Interface contract

- Tensor layout: `(B, C, D, H, W)`.
- The image pair is expected to be normalized consistently, normally to `[0, 1]`.
- `base_dvf` has three channels and uses the same voxel units, axis order, and
  fixed-grid sampling convention as the supplied warp operator.
- Small-structure and bone masks are one-channel tensors. The warped masks must
  be produced with the base DVF before refinement.
- The default network consumes seven channels and predicts an additive residual:
  `refined_dvf = base_dvf + roi_gate * residual_scale * raw_residual_dvf`.

## Direct use with an existing base result

```python
from musa.anatomy_refinement import (
    AnatomyAwareLocalRefiner,
    LocalRefinementConfig,
    RefinementInput,
)

refiner = AnatomyAwareLocalRefiner(
    config=LocalRefinementConfig(input_mode="full")
).to(device)

result = refiner(
    RefinementInput(
        fixed_image=fixed,
        warped_moving_image=base_warped,
        base_dvf=base_dvf,
        fixed_small_mask=fixed_small,
        warped_small_mask=base_warped_small,
        fixed_bone_mask=fixed_bone,
        warped_bone_mask=base_warped_bone,
        moving_image=moving,
    )
)

refined_dvf = result.refined_dvf
```

`result` also exposes the difficulty, ROI radius/gate, residual scale, dynamic
loss weights, raw/scaled/gated residual DVFs, and anatomy regularization maps.

## Wrap a backbone

For a backbone returning `(warped_moving, dvf)`:

```python
from musa.anatomy_refinement import RegistrationWithLocalRefinement

model = RegistrationWithLocalRefinement(
    backbone=base_model,
    refiner=refiner,
    warp=spatial_transformer,
    freeze_backbone=True,
)

output = model(
    moving,
    fixed,
    moving_small,
    fixed_small,
    moving_bone,
    fixed_bone,
)
```

Use `base_output_order="dvf-warped"` for reversed tuple output, or pass a
`base_output_adapter` callback for dictionaries and custom model APIs.

## Loading an existing Stage-3 checkpoint

The trainable layer names are unchanged, so existing checkpoints remain valid:

```python
payload = torch.load(checkpoint_path, map_location=device)
state_dict = payload.get("model_state_dict", payload)
refiner.load_state_dict(state_dict)
```

The checkpoint must use the same U-Net filters and input policy used for
training. `full` uses fixed small-structure and bone masks. For deployment
without fixed segmentation, train and evaluate a `no-fixed-seg` checkpoint.

## Training utilities

`musa.anatomy_refinement.losses` provides the same per-pair Dice, local image,
weighted smoothness, weighted magnitude, and Jacobian losses used by the
original training script. CRS remains an optional training-data strategy and
is not required at inference.
