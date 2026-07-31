# Implementation Guide — Gaussian-Native Registration V11
> Updated: 2026-07-31 | Strategy: evidence-driven v10 refinement | Status: APPROVED_FOR_IMPLEMENTATION

## 1. Motivation

V10 converged normally and reached validation Dice 0.61505 at epoch 120, but
GTVp remained the limiting class (0.46194 versus 0.75007 for GTVn). Dataset
audit showed that paired GTVp has a fixed-image median volume of 786.5 voxels,
roughly half the GTVn median, and substantially larger longitudinal volume
change. The final learned v10 residual is predicted at factor two, so the
smallest GTVp boundaries are represented by too few cells.

V11 keeps the successful Gaussian representation/correspondence and the v10
factor-8/4/2 residual stages. It changes only the fine-refinement and supervised
training components needed to address the measured failure mode.

## 2. Model change

### `gaussian_native/refinement.py`

`GaussianGuidedResidualPyramid` will accept at least three decreasing pyramid
factors instead of exactly three. The first three stages receive the three
Gaussian velocity components; later stages refine the accumulated stationary
velocity without adding a nonexistent Gaussian level.

V11 uses factors `8/4/2/1`, channels `48/40/32/16`, block counts `3/3/3/2`,
and residual limits `1.5/1.0/0.75/0.35` stage voxels. Unit-aware velocity
upsampling remains unchanged.

`ResidualVelocityStage` gains an optional image-gradient input. With the flag
enabled, fixed and currently warped moving gradient magnitudes are concatenated
with the existing fixed image, warped moving image, intensity difference, and
three-channel current flow. The input channel count is therefore eight.

The full-resolution output head remains zero-initialized. At initialization,
V11 exactly preserves the three-stage Gaussian-guided v10 deformation.

### `gaussian_native/model.py` and `experiment_utils.py`

Add architecture revision `gaussian_native_v11`, pass the four-stage
configuration, and expose the fourth residual/flow diagnostic. V10 behavior
must remain unchanged.

## 3. Supervision change

### `train_registration.py`

`_masked_soft_dice` will accept per-label positive weights. V11 uses
`GTVp:GTVn = 1.5:1.0`; weights apply to Dice and boundary Dice losses, while
reported validation metrics remain unweighted.

`_supervised_anatomy_loss` will:

1. avoid duplicating the final full-resolution flow when the last pyramid
   stage is already factor one;
2. add a normalized soft-centroid distance for long-range small-target
   alignment;
3. add a final inverse-direction soft Dice using `inverse_flow`;
4. retain response-aware masking, so a label contributes only when present at
   both timepoints.

The configured total anatomy term is:

`Dice + 0.25 * boundary Dice + 0.15 * centroid + 0.20 * inverse Dice`.

Shared left-right flipping will transform moving image, fixed image, moving
segmentation, and fixed segmentation identically. Pair reversal remains
disabled because the experiment direction is preRT to midRT.

## 4. Training configuration

Create `configs/gaussian_native_v11_hntsmrg24.json` from v10 with:

- full-resolution gradient-aware refinement;
- 150 epochs;
- five-epoch LR warmup and 15-epoch anatomy ramp;
- shared flip probability 0.5;
- weight decay `2e-5`;
- Dice-based checkpoint selection and topology constraints unchanged.

The primary comparison is v10 versus v11 under the same split and seed.
Required ablations are v11 without the factor-one stage, without label
weighting, and without centroid/inverse supervision.

## 5. Validation requirements

- Existing tests remain green.
- A four-stage zero-initialized pyramid returns an identity flow when Gaussian
  velocities are zero.
- Dense flow gradients reach all four residual heads.
- Identical labels yield zero weighted Dice, boundary, centroid, and inverse
  losses.
- Shared spatial augmentation keeps image/segmentation geometry identical.
- Production-shape A100 smoke is finite and fits within 40 GiB.
- One real training batch has finite nonzero gradients.

## 6. Design validation

- Experiment coverage: v10 comparison and each new mechanism have explicit
  configuration switches.
- Logical consistency: factor-one flow is in full-resolution voxel units and
  is not counted twice by deep supervision.
- Completeness: model, loss, augmentation, configuration, diagnostics, tests,
  README, and development log are all included.
