# Gaussian-native diffeomorphic registration for longitudinal head-and-neck MRI

This repository contains a new registration model whose learned
representation, correspondence, and deformation parameters are all Gaussian
primitives. SACB-Net is retained only as a controlled baseline and is not part
of the new model.

```text
Moving / Fixed volume
        ↓
Gaussian scale-space and derivative measurements
        ↓
Hierarchical anisotropic Gaussian decomposition
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
Scaling-and-squaring
        ↓
Dense diffeomorphic deformation
```

The implementation is organized around two research modules:

1. **Hierarchical Gaussian Representation and Correspondence (HGRC)** learns
   full-covariance, mass-conserving Gaussian anatomy hierarchies and explicit
   coarse-to-fine correspondence. The production path uses strict sparse
   candidate support; Sinkhorn and an unmatched dustbin remain ablations.
2. **Gaussian Stationary Velocity Field Generator (GSVF)** predicts
   translation, rotation, and bounded strain for every Gaussian, rasterizes
   coarse-to-fine residual velocities, and integrates the SVF into a
   diffeomorphism.

There is no SACB branch, learned dense-flow head, Gaussian/dense gate, or
standalone confidence module in the new prediction path.

## Environment

```bash
conda create -n gaussian-native python=3.9
conda activate gaussian-native
pip install -r requirements.txt
```

The code remains compatible with the original PyTorch 1.13/CUDA 11.7
environment. The production configuration uses bfloat16 autocast on an A100;
covariance, transport, rasterization, and integration calculations are kept in
float32.

## HNTS-MRG24 preprocessing

The main experiment is within-patient longitudinal T2 MRI registration. Raw
preRT after rigid/affine prealignment is moving and midRT is fixed. The
challenge-provided deformably registered preRT image is excluded. Tumor labels
(`1=GTVp`, `2=GTVn`) are reserved for validation and testing.

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

The current production revision is v6. It retains v5's masked row-softmax and
shared fixed-to-moving/fixed-to-fixed support, but removes the forced identity
parent candidate. Normalized Gaussian intensity/derivative measurements anchor
early matching and are cosine-annealed from weight 0.75 to 0.25 over 30
epochs, allowing learned anatomy features to take over. Candidate-normalized
entropy remains a deterministic ambiguity statistic, but its square root
weights motion so uncertain child residuals are not suppressed linearly.
Middle/fine direct residual capacity is increased while the root contribution
is slightly reduced. Canonical anchors remain the stable velocity
rasterization basis, and a stronger representation anchor prevents the
fine-Gaussian drift observed in v5. Matching temperature is annealed from 0.12
to 0.07 over 60 epochs. Earlier revisions remain available only for
reproducing prior experiments.

First run one production-shape forward/backward memory audit:

```bash
CUDA_VISIBLE_DEVICES=0 python smoke_gaussian_native.py \
  --config configs/gaussian_native_v6_hntsmrg24.json \
  --device cuda:0
```

Proceed only if `"finite": true`. Record `peak_gpu_memory_mb` with the paper
experiment metadata.

```bash
CUDA_VISIBLE_DEVICES=0 python train_gaussian_native.py \
  --config configs/gaussian_native_v6_hntsmrg24.json \
  --data-root /path/to/HNTSMRG24_gaussian_native_preprocessed \
  --train-manifest /path/to/HNTSMRG24_gaussian_native_preprocessed/manifests/train.csv \
  --validation-manifest /path/to/HNTSMRG24_gaussian_native_preprocessed/manifests/validation.csv \
  --output-dir runs/gaussian_native_v6_hntsmrg24_seed2026 \
  --device cuda:0
```

The production model contains 64/256/1024 Gaussian primitives and about 2.08
million trainable parameters. Training uses:

- bidirectional multi-scale LNCC and normalized-gradient similarity;
- Gaussian reconstruction, coverage, and hierarchy containment;
- appearance-anchored sparse Gaussian correspondence, trained through image
  similarity rather than a self-minimizing transport-cost loss;
- SVF smoothness, inverse consistency, and a Jacobian safety barrier;
- shared left-right flipping and shared MRI intensity augmentation.

Each validation record includes NCC and Dice before/after registration,
improvements, displacement, and topology. The console also reports coarse
support-normalized matching entropy, deterministic match evidence, row maximum,
effective motion evidence, diagonal probability, and calibrated transport
displacement. Clearly harmful or stalled runs are stopped by configured
fail-fast rules and retain a `failed_epoch_XXXX.pt` checkpoint with the exact
reason.

The best checkpoint is selected by validation NCC, not tumor Dice, and is
written only when NCC improves over the unregistered pair and the negative
Jacobian ratio is at most 1%. Resume only with the same configuration and
manifest hashes:

```bash
CUDA_VISIBLE_DEVICES=0 python train_gaussian_native.py \
  --config configs/gaussian_native_v6_hntsmrg24.json \
  --data-root /path/to/HNTSMRG24_gaussian_native_preprocessed \
  --train-manifest /path/to/HNTSMRG24_gaussian_native_preprocessed/manifests/train.csv \
  --validation-manifest /path/to/HNTSMRG24_gaussian_native_preprocessed/manifests/validation.csv \
  --output-dir runs/gaussian_native_v6_hntsmrg24_seed2026 \
  --device cuda:0 \
  --resume runs/gaussian_native_v6_hntsmrg24_seed2026/latest.pt
```

## Evaluation

Evaluate the held-out test set after validation-based model selection:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_gaussian_native.py \
  --checkpoint runs/gaussian_native_v6_hntsmrg24_seed2026/best_validation_ncc.pt \
  --data-root /path/to/HNTSMRG24_gaussian_native_preprocessed \
  --manifest /path/to/HNTSMRG24_gaussian_native_preprocessed/manifests/test.csv \
  --output-dir results/gaussian_native_v6_hntsmrg24_seed2026 \
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
gradients, DHW flow direction, medical metrics, and dataset behavior.

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
