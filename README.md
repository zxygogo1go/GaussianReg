# Gaussian-native diffeomorphic registration for longitudinal head-and-neck MRI

HaN-Seg, Head-Neck-CBCT-CT, and SegRap2023 preprocessing/training commands are
in [`docs/external_head_neck_training.md`](docs/external_head_neck_training.md).

This repository contains a new registration model whose core representation,
correspondence, and coarse deformation parameters are Gaussian primitives.
Current revisions add a bounded residual image pyramid for fine deformation.
SACB-Net is retained only as a controlled baseline and is not part of the new
model.

```text
Moving / Fixed volume
        ↓
Gaussian scale-space and derivative measurements
        ↓
Hierarchical anchored Gaussian decomposition
64 → 256 → 1024 primitives
        ↓
Gaussian graph encoding
        ↓
Coarse-to-fine appearance-anchored Gaussian correspondence
        ↓
Per-Gaussian local affine stationary velocity
        ↓
Multi-scale Gaussian SVF synthesis
        ↓
Gaussian-guided coarse-to-fine residual SVF pyramid
factor 8 → warp → factor 4 → warp → factor 2 → warp
        ↓
gradient-aware factor 1 boundary refinement
        ↓
Full-resolution scaling-and-squaring
        ↓
Dense diffeomorphic deformation
```

The implementation is organized around two research modules:

1. **Hierarchical Gaussian Representation and Correspondence (HGRC)** encodes
   fixed, mass-conserving Gaussian anatomy hierarchies and learns explicit
   coarse-to-fine Gaussian-to-Gaussian correspondence. The production path
   uses strict sparse candidate support and a bounded learned residual on top
   of a fixed appearance/geometric matching base; Sinkhorn and an unmatched
   dustbin remain ablations.
2. **Gaussian-Guided Residual Diffeomorphic Pyramid (GGRP)** predicts
   translation, rotation, and bounded strain for every Gaussian, rasterizes
   the three Gaussian velocity components, and then performs sequential
   image-warped residual SVF refinement at factors 8, 4, and 2. V11 adds a
   gradient-aware factor-one stage before the accumulated full-resolution
   stationary velocity is integrated into a diffeomorphism.

There is no SACB branch, Gaussian/dense gate, or standalone confidence module
in the new prediction path. Revisions v10/v11 deliberately add a conventional
voxel residual refiner after Gaussian correspondence; it is therefore a
Gaussian-guided hybrid rather than a strictly Gaussian-only model. Revision v9
is retained as the strict Gaussian-only unsupervised ablation.

## Environment

```bash
conda create -n gaussian-native python=3.9
conda activate gaussian-native
pip install -r requirements.txt
```

The code remains compatible with the original PyTorch 1.13/CUDA 11.7
environment. The production configuration uses bfloat16 autocast on an A100;
covariance, transport, rasterization, and integration calculations are kept in
float32. AMP weight caching is explicitly disabled because the same Gaussian
matcher is first reused by no-gradient self-calibration and then by the
gradient-bearing cross-image path; caching the first cast would disconnect the
learned residual scorer.

## HNTS-MRG24 preprocessing

The main experiment is within-patient longitudinal T2 MRI registration. Raw
preRT after rigid/affine prealignment is moving and midRT is fixed. The
challenge-provided deformably registered preRT image is excluded. Tumor labels
are `1=GTVp` and `2=GTVn`. Revisions v10/v11 use the paired labels during
training; a class contributes only when present at both timepoints.
Consequently they must be reported as segmentation-supervised and compared
with supervised baselines. Revision v9 remains the label-free comparison.

Preprocessing creates 1.5 mm isotropic `(D,H,W)=(128,160,160)` volumes,
performs robust MRI normalization and geometry QA, and writes patient-disjoint
train/validation/test manifests.

```bash
python prepare_hntsmrg24.py \
  --source-root /path/to/HNTSMRG24_train \
  --output-root /path/to/HNTSMRG24_gaussian_native_preprocessed \
  --num-workers 2
```

Do not use `--allow-failures` for a final paper experiment without reporting
every exclusion recorded in `dataset_summary.json`.

## Gaussian-native training

Run from the repository root on one selected A100. This is an explicit command;
no shell launch wrapper is required.

The current experimental revision is v12. It retains v11's contextual matcher
and factor-8/4/2/1 residual image pyramid, but replaces forced row-softmax
matching with bidirectional KL-relaxed Sinkhorn transport and an explicit
unmatched dustbin. Match evidence is the product of support concentration and
the transported real mass, so unmatched longitudinal anatomy cannot generate a
full-strength displacement merely because its remaining candidates are sharp.

V12 uses a Gaussian-first curriculum. Epochs 1--15 use known synthetic
diffeomorphic pairs, the zero-initialized dense residual pyramid is frozen
through epoch 20, and real-pair GTV supervision starts only at epoch 31. The
anatomy factor ramps from 0.05 to 0.40 instead of reaching 1.0. Moving and fixed
T2 intensities are augmented independently. These changes target the v11
failure mode in which training Dice reached 0.90 while validation Dice had
already plateaued near epoch 22.

First run one production-shape forward/backward memory audit:

```bash
CUDA_VISIBLE_DEVICES=0 python smoke_gaussian_native.py \
  --config configs/gaussian_native_v12_hntsmrg24.json \
  --device cuda:0
```

Proceed only if `"finite": true`. Record `peak_gpu_memory_mb` with the paper
experiment metadata.

```bash
CUDA_VISIBLE_DEVICES=0 python train_gaussian_native.py \
  --config configs/gaussian_native_v12_hntsmrg24.json \
  --data-root /path/to/HNTSMRG24_gaussian_native_preprocessed \
  --train-manifest /path/to/HNTSMRG24_gaussian_native_preprocessed/manifests/train.csv \
  --validation-manifest /path/to/HNTSMRG24_gaussian_native_preprocessed/manifests/validation.csv \
  --output-dir runs/gaussian_native_v12_hntsmrg24_seed2026 \
  --device cuda:0
```

The v12 model contains 64/256/1024 Gaussian primitives and 5,380,790 trainable
parameters. A full `(128,160,160)` A100 forward/backward smoke test uses about
10.3 GiB peak allocated GPU memory. Training uses:

- bidirectional multi-scale LNCC and normalized-gradient similarity;
- an anchored, mass-conserving Gaussian hierarchy without learned geometry
  predictors;
- fixed-base plus multi-head contextual Gaussian scores;
- bidirectional unbalanced Sinkhorn, an explicit dustbin, and partial-mass
  motion attenuation;
- sequential factor-8/4/2/1 residual SVF refinement after image warping;
- gradient features in every residual stage, including full resolution;
- delayed weak GTVp-weighted Dice, boundary, centroid, and inverse supervision;
- SVF smoothness, inverse consistency, and a Jacobian safety barrier;
- independent longitudinal MRI intensity and synchronized image/label flip
  augmentation.

Each validation record includes NCC and Dice before/after registration,
improvements, displacement, and topology. The console also reports coarse
support-normalized matching entropy, deterministic match evidence, row maximum,
effective motion evidence, diagonal probability, and calibrated transport
displacement. v9 additionally logs the synthetic-pair fraction, synthetic flow
and transport losses, endpoint error, contextual-attention concentration, and
mean absolute learned residual logit. v10--v12 additionally log residual
velocity and accumulated flow magnitude at every pyramid stage. V12 logs real
transport mass, fixed/moving unmatched mass, relaxed marginal error, and the
weighted Dice, boundary, centroid, and inverse-Dice terms. Every residual
velocity head is exactly zero before the first update and should then become
nonzero.
Clearly harmful or stalled runs are stopped by configured fail-fast rules and
retain a `failed_epoch_XXXX.pt` checkpoint with the exact reason.

The v12 best checkpoint is selected by validation mean Dice and written as
`best_validation_dice.pt` only when Dice improves over the unregistered pair,
NCC degradation is below the configured tolerance, and the negative Jacobian
ratio is at most 1%. Resume only with the same configuration and manifest
hashes:

```bash
CUDA_VISIBLE_DEVICES=0 python train_gaussian_native.py \
  --config configs/gaussian_native_v12_hntsmrg24.json \
  --data-root /path/to/HNTSMRG24_gaussian_native_preprocessed \
  --train-manifest /path/to/HNTSMRG24_gaussian_native_preprocessed/manifests/train.csv \
  --validation-manifest /path/to/HNTSMRG24_gaussian_native_preprocessed/manifests/validation.csv \
  --output-dir runs/gaussian_native_v12_hntsmrg24_seed2026 \
  --device cuda:0 \
  --resume runs/gaussian_native_v12_hntsmrg24_seed2026/latest.pt
```

## Evaluation

Evaluate the held-out test set after validation-based model selection:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_gaussian_native.py \
  --checkpoint runs/gaussian_native_v12_hntsmrg24_seed2026/best_validation_dice.pt \
  --data-root /path/to/HNTSMRG24_gaussian_native_preprocessed \
  --manifest /path/to/HNTSMRG24_gaussian_native_preprocessed/manifests/test.csv \
  --output-dir results/gaussian_native_v12_hntsmrg24_seed2026 \
  --device cuda:0 \
  --save-predictions
```

The evaluator writes patient-level CSV/JSON and bootstrap 95% confidence
intervals for NCC, Dice, HD95, ASSD, displacement, runtime, and Jacobian
topology. A tumor class is eligible only when present in both original
timepoints. If an eligible structure is lost after warping, it receives Dice 0
and an image-diagonal surface-distance penalty.

## Controlled SACB-Net baseline

The original SACB-Net can still be trained and evaluated with the same data
split and reporting protocol:

```bash
CUDA_VISIBLE_DEVICES=0 python train_sacb_baseline.py \
  --config configs/sacb_baseline_hntsmrg24.json \
  --data-root /path/to/HNTSMRG24_gaussian_native_preprocessed \
  --train-manifest /path/to/HNTSMRG24_gaussian_native_preprocessed/manifests/train.csv \
  --validation-manifest /path/to/HNTSMRG24_gaussian_native_preprocessed/manifests/validation.csv \
  --output-dir runs/sacb_baseline_hntsmrg24_seed2026 \
  --device cuda:0
```

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_sacb_baseline.py \
  --checkpoint runs/sacb_baseline_hntsmrg24_seed2026/best_validation_ncc.pt \
  --data-root /path/to/HNTSMRG24_gaussian_native_preprocessed \
  --manifest /path/to/HNTSMRG24_gaussian_native_preprocessed/manifests/test.csv \
  --output-dir results/sacb_baseline_hntsmrg24_seed2026 \
  --device cuda:0 \
  --save-predictions
```

## Verification

```bash
python -m unittest discover -s tests -v
```

The focused suite covers full SPD geometry, mass-conserving hierarchy,
partial transport, parent-conditioned matching, Gaussian velocity synthesis,
identity initialization, scaling-and-squaring inverse composition, full-model
gradients, true residual-pyramid propagation, response-aware soft-label
supervision, DHW flow direction, medical metrics, and dataset behavior.

## SACB-Net citation

```bibtex
@InProceedings{Cheng_2025_CVPR,
    author    = {Cheng, Xinxing and Zhang, Tianyang and Lu, Wenqi and Meng, Qingjie and Frangi, Alejandro F. and Duan, Jinming},
    title     = {SACB-Net: Spatial-awareness Convolutions for Medical Image Registration},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {5227-5237}
}
```
