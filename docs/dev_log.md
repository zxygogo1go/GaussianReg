# Development Log — Gaussian-Native Registration
> Created: 2026-07-31 | Last updated: 2026-08-01
> Implementation guide: `docs/implementation.md`

## Progress

| Module | Files | Status | Notes |
|---|---|---|---|
| V10 result diagnosis | server metrics and label statistics | Done | Best Dice 0.61505 at epoch 120; GTVp is limiting |
| V11 refinement | `gaussian_native/refinement.py`, model builder | Done | Four-stage unit/GPU/real-batch checks passed |
| V11 supervision | `train_registration.py` | Done | Loss and synchronized augmentation checks passed |
| V11 configuration | `configs/gaussian_native_v11_hntsmrg24.json` | Done | 150 epochs; explicit command, no launch script |
| Verification | unit tests and A100 smoke | Done | 62 tests; production smoke; two real batches |
| V12 partial transport | `gaussian_native/correspondence.py` | Done | Bidirectional KL-relaxed Sinkhorn with unmatched mass |
| V12 Gaussian-first curriculum | `train_registration.py` | Done | Synthetic pretrain, delayed refinement and weak labels |
| V12 configuration | `configs/gaussian_native_v12_hntsmrg24.json` | Done | Explicit command; 64 tests and A100 smoke passed |

## Log

### 2026-08-01 — V12 partial correspondence and curriculum

- Diagnosed v11 as numerically stable but overfit: best validation Dice was
  0.61962 at epoch 110, versus 0.61920 at epoch 22, while training Dice rose
  from 0.744 to 0.884.
- Replaced production row-softmax with bidirectional KL-relaxed Sinkhorn and an
  explicit dustbin. Motion evidence now includes actual matched mass.
- Added real transport mass, fixed/moving unmatched mass, and marginal error
  diagnostics at every Gaussian level.
- Added a Gaussian-first schedule: 15 synthetic-only epochs, residual-pyramid
  freeze through epoch 20, and weak anatomy supervision beginning at epoch 31.
- Enabled independent moving/fixed intensity augmentation.
- Passed 64/64 server tests. Production A100 smoke was finite with 5,380,790
  parameters, 10,539.4 MiB peak allocation, and nonzero gradients in all three
  contextual pair scorers. Initial real transport mass was 0.688/0.706/0.717.

### 2026-07-31 — V11 design

- V10 completed 200 epochs without numerical or topology failure.
- Validation Dice improved from 0.55827 to 0.61505; best epoch was 120.
- GTVp/GTVn validation Dice was 0.46194/0.75007.
- Training-label audit found fixed-volume medians of 786.5/1563 voxels for
  GTVp/GTVn, motivating full-resolution small-target refinement.
- Approved V11 scope: factor-one gradient-aware refinement, GTVp-weighted
  supervision, centroid/inverse constraints, and synchronized flipping.

### 2026-07-31 — Residual pyramid core

- Generalized the residual pyramid from exactly three to at least three
  decreasing scales.
- Kept Gaussian velocity injection at exactly the first three scales and
  enabled later image-only residual stages.
- Added optional fixed/warped gradient-magnitude inputs and per-stage block
  counts.
- Validation is pending until V11 model wiring and tests are complete.

### 2026-07-31 — V11 model wiring

- Registered `gaussian_native_v11` without changing v10 behavior.
- Added list-valued per-stage block configuration and the gradient-feature
  switch to the model builder.
- Exposed the existing per-stage diagnostic path to the new fourth stage.

### 2026-07-31 — Small-target supervision

- Added positive per-label weighting to soft Dice and boundary Dice.
- Added normalized soft-centroid distance and inverse-direction Dice.
- Avoided duplicate deep supervision when the last stage is already full
  resolution.
- Added geometry-synchronized image/segmentation reversal and flipping;
  production V11 enables flipping but continues to reject pair reversal.
- Extended smoke and training totals with the new supervised terms.

### 2026-07-31 — V11 experiment configuration

- Added the complete four-stage V11 configuration.
- Set GTVp/GTVn loss weights to 1.5/1.0 and enabled centroid/inverse terms.
- Enabled synchronized left-right flipping at probability 0.5.
- Limited training to 150 epochs because V10 validation peaked at epoch 120.
- Updated the default smoke configuration and explicit run instructions.

### 2026-07-31 — Verification and documentation

- Passed 62/62 server unit tests, including v10 compatibility, four-stage
  gradient propagation, full-resolution de-duplication, and label-aligned flip.
- Production A100 smoke: finite, 5,380,790 parameters, 359 gradient tensors,
  10,192.9 MiB peak allocation, and 2.46 seconds.
- Two real samples completed optimization with all supervised terms finite;
  the fourth residual became nonzero after the first update.
- Updated README, implementation specification, smoke default, and explicit
  run instructions for V11.
- Moved label-weight positivity validation to CPU configuration values to avoid
  unnecessary CUDA synchronization inside every deep-supervision scale.

## Known issues

- Full training has not yet been run; validation improvement over V10 remains
  an experimental hypothesis.

## Run instructions

### Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 /home/student3/miniconda3/envs/SACB/bin/python \
  smoke_gaussian_native.py \
  --config configs/gaussian_native_v12_hntsmrg24.json \
  --device cuda:0
```

- `--config` selects the complete V12 architecture and training objective.
- `--device` selects the visible A100.
- The command performs one production-shape forward/backward pass and prints
  finiteness, gradients, parameter count, runtime, and peak allocated memory.

### Training

```bash
CUDA_VISIBLE_DEVICES=0 nohup \
  /home/student3/miniconda3/envs/SACB/bin/python \
  train_gaussian_native.py \
  --config configs/gaussian_native_v12_hntsmrg24.json \
  --data-root /home/student3/data2t/TouJing/HNTSMRG24_gam_preprocessed \
  --train-manifest /home/student3/data2t/TouJing/HNTSMRG24_gam_preprocessed/manifests/train.csv \
  --validation-manifest /home/student3/data2t/TouJing/HNTSMRG24_gam_preprocessed/manifests/validation.csv \
  --output-dir /home/student3/data2t/TouJing/GaussianReg/runs/gaussian_native_v12_hntsmrg24_seed2026 \
  --device cuda:0 \
  > /home/student3/data2t/TouJing/GaussianReg/train_gaussian_native_v12_hntsmrg24_seed2026.log 2>&1 &
```

- The manifests define the patient-disjoint train/validation split.
- The output directory receives `metrics.jsonl`, `latest.pt`, periodic
  checkpoints, TensorBoard data, and `best_validation_dice.pt`.
- The log contains step/epoch losses, Dice/NCC improvement, Gaussian matching,
  four residual-pyramid diagnostics, and topology metrics.
