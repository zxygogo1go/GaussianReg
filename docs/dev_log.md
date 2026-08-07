# Development Log — Gaussian-Native Registration
> Created: 2026-07-31 | Last updated: 2026-08-07
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
| V13 SAGR design | `docs/implementation.md` | Done | Label-free inference, adaptive child Gaussians, local residual SVF composition |
| V13 SAGR core | `gaussian_native/small_organ_refinement.py` | Done | Priority, child densification, local transport and residual SVF implemented |
| V13 model integration | `gaussian_native/model.py`, `losses.py`, `experiment_utils.py` | Done | Composition, regularization, builder and diagnostics complete |
| V13 training integration | `train_registration.py` | Done | Mixed labels, training-only priority loss, schedule and logging complete |
| V13 configurations | `configs/gaussian_native_v13_*.json` | Done | HNTS/HaN-Seg/SegRap SAGR; unlabeled CBCT SAGR disabled |
| V13 tests | `tests/test_gaussian_native_small_organ_refinement.py`, `tests/test_gaussian_native_model.py`, `tests/test_experiment_utils.py` | Done | Full CPU suite: 77 cases, 3 optional-dependency skips, no failures |
| V13 smoke | `smoke_gaussian_native.py` | Implemented | A100 production run pending after GitHub synchronization |
| V13 documentation | `README.md`, `docs/external_head_neck_training.md` | Done | Explicit V13 smoke/train/evaluation commands added |
| V13 SAGR implementation | model/training/config/tests | Done locally | Runtime acceptance requires server unit tests and A100 smoke |

## Log

### 2026-08-07 — V13 local implementation complete

- Completed the model, loss, builder, training, configuration, tests, smoke,
  implementation specification, README, and external-dataset commands.
- Confirmed by static review that local and global deformations use the correct
  fixed-grid composition order and that the inverse uses reverse order.
- Made the no-adaptive-priority ablation use fixed nodes and unit gates, so it
  no longer receives a hidden learned-priority effect.
- Added selected-target hit rate/recall and SAGR priority/local-flow/direct-gain
  values to JSON/TensorBoard and the per-epoch console log.
- Resolved all four V13 configurations through their inheritance chain and
  constructed the models: labeled V13 has 5,597,405 parameters; the
  SAGR-disabled CBCT configuration preserves the 5,380,790-parameter V12 path.
- JSON inheritance and Python syntax checks pass locally. The full x86 CPU
  suite ran through Rosetta: 77 cases, 3 expected optional-dependency/CUDA
  skips, and no failures. Production-shape forward/backward remains required
  on the A100.

### 2026-08-06 — SAGR core module

- Added fixed analytic factor-two image measurements and physical-point
  sampling without a learned dense image encoder.
- Added finest-Gaussian adaptive priority, fixed-budget top-K selection, and
  symmetric child Gaussian densification.
- Added within-parent mutual child correspondence, zero-initialized bounded
  child velocity prediction, Gaussian rasterization, and residual SVF
  scaling-and-squaring.
- Kept all segmentation information outside the model forward interface.
- Verification remains pending until the V13 model integration and tests are
  complete.

### 2026-08-06 — V13 model integration

- Registered `gaussian_native_v13` while preserving all V1--V12 branches.
- Composed the integrated local residual with the global deformation and
  constructed the inverse in reverse composition order.
- Added local SVF smoothness/energy regularization and SAGR diagnostics.
- Added complete JSON-to-model parameter wiring for controlled ablations.
- Syntax checks passed; forward/backward verification remains pending.

### 2026-08-06 — V13 training integration

- Added a stable union of global anatomy labels and SAGR-only small-organ
  labels, with loss-specific channel and validity selection.
- Added physical-radius dilation and balanced BCE plus soft-Dice supervision
  for finest-Gaussian priority logits.
- Ensured labels remain outside `model.forward`; synthetic samples skip all
  segmentation-derived losses.
- Appended the composed final flow to deep supervision so local Gaussian
  velocities receive organ-alignment gradients.
- Added curriculum freezing/ramping and TensorBoard/JSON metrics for SAGR.
- Static syntax validation passed; unit and real-batch verification remain.

### 2026-08-06 — V13 dataset configurations

- Added SAGR production settings for HNTS-MRG24, HaN-Seg, and SegRap2023.
- HaN-Seg priority supervision targets Arytenoid, cochleae, lacrimal glands,
  optic chiasm/nerves, and pituitary.
- SegRap2023 priority supervision targets chiasm, cochleae, ET bones, IACs,
  lenses, optic nerves, pituitary, tympanic cavities, and vestibular
  semicircular structures.
- Kept Head-Neck-CBCT-CT SAGR disabled because its current preprocessed copy
  contains no labels with which to validate the claimed small-organ endpoint.
- Added five-way deep supervision weights so the composed V13 flow receives
  the strongest anatomy gradient.

### 2026-08-06 — V13 test coverage

- Added analytic-measurement and physical-sampling shape tests.
- Added exact zero-initialization preservation and SAGR head-gradient tests.
- Added mixed loaded-label selection and training-only priority-loss tests.
- Added V13 builder and Gaussian-pretrain freeze/ramp tests.
- Execution is pending because the local Apple-Silicon Python installations do
  not include a compatible PyTorch; syntax validation continues locally and
  production tests will run in the existing server environment.

### 2026-08-06 — V13 production smoke path

- Switched the default production smoke configuration to V13.
- Added synthetic small-target labels only inside the smoke loss path, while
  preserving the image-only model forward call.
- Added explicit nonzero-gradient checks for both the SAGR priority head and
  residual velocity head, plus all new SAGR diagnostics in the JSON output.
- Production memory and runtime remain pending on the 40 GiB A100.

### 2026-08-06 — V13 run documentation

- Updated the architecture narrative from two modules to three and documented
  training-only label supervision versus image-only inference.
- Replaced primary HNTS-MRG24, HaN-Seg, and SegRap train/evaluation commands
  with explicit V13 commands; no shell launch script was added.
- Added the architecture-compatible V13 CBCT-CT command with SAGR explicitly
  disabled because that dataset has no small-organ labels.

### 2026-08-06 — V13 SAGR design locked

- Reframed the user's successful MUSA stage-three small-organ refiner as a
  Gaussian-native module rather than adding another full-volume U-Net.
- Locked image-only inference: fixed small-organ labels are permitted only in
  the training loss that teaches Gaussian priority.
- Selected a fixed-budget top-K design with nine child Gaussians per selected
  finest-level parent and within-parent local Gaussian correspondence.
- Required a bounded residual SVF, scaling-and-squaring, and deformation
  composition with the V12 global transform.
- Preserved the untracked `小器官精修模块/` prototype without importing or
  modifying it.

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

- V13 has not yet completed its server unit test, production smoke, or full
  training; improvement over V12 remains an experimental hypothesis.

## Run instructions

### Unit tests

```bash
CUDA_VISIBLE_DEVICES=0 /home/student3/miniconda3/envs/SACB/bin/python \
  -m unittest discover -s tests -v
```

### Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 /home/student3/miniconda3/envs/SACB/bin/python \
  smoke_gaussian_native.py \
  --config configs/gaussian_native_v13_hntsmrg24.json \
  --device cuda:0
```

- `--config` selects the complete V13 architecture, SAGR, and objectives.
- `--device` selects the visible A100.
- The command performs one production-shape forward/backward pass and prints
  finiteness, gradients, parameter count, runtime, and peak allocated memory.

### Training

```bash
CUDA_VISIBLE_DEVICES=0 nohup \
  /home/student3/miniconda3/envs/SACB/bin/python \
  train_gaussian_native.py \
  --config configs/gaussian_native_v13_hntsmrg24.json \
  --data-root /home/student3/data2t/TouJing/HNTSMRG24_gam_preprocessed \
  --train-manifest /home/student3/data2t/TouJing/HNTSMRG24_gam_preprocessed/manifests/train.csv \
  --validation-manifest /home/student3/data2t/TouJing/HNTSMRG24_gam_preprocessed/manifests/validation.csv \
  --output-dir /home/student3/data2t/TouJing/GaussianReg/runs/gaussian_native_v13_hntsmrg24_seed2026 \
  --device cuda:0 \
  > /home/student3/data2t/TouJing/GaussianReg/train_gaussian_native_v13_hntsmrg24_seed2026.log 2>&1 &
```

- The manifests define the patient-disjoint train/validation split.
- The output directory receives `metrics.jsonl`, `latest.pt`, periodic
  checkpoints, TensorBoard data, and `best_validation_dice.pt`.
- The log contains step/epoch losses, Dice/NCC improvement, global and local
  Gaussian matching, SAGR priority/coverage/local flow, four residual-pyramid
  diagnostics, and topology metrics.

### HNTS-MRG24 evaluation

```bash
CUDA_VISIBLE_DEVICES=0 /home/student3/miniconda3/envs/SACB/bin/python \
  evaluate_gaussian_native.py \
  --checkpoint /home/student3/data2t/TouJing/GaussianReg/runs/gaussian_native_v13_hntsmrg24_seed2026/best_validation_dice.pt \
  --data-root /home/student3/data2t/TouJing/HNTSMRG24_gam_preprocessed \
  --manifest /home/student3/data2t/TouJing/HNTSMRG24_gam_preprocessed/manifests/test.csv \
  --output-dir /home/student3/data2t/TouJing/GaussianReg/results/gaussian_native_v13_hntsmrg24_seed2026 \
  --device cuda:0 \
  --save-predictions
```

The explicit HaN-Seg, SegRap2023, and Head-Neck-CBCT-CT training and evaluation
commands are maintained in `docs/external_head_neck_training.md`.
