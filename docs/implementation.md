# Implementation Guide — Gaussian-Native Registration V13 with SAGR
> Updated: 2026-08-07 | Strategy: extend the validated V12 backbone | Status: IMPLEMENTED_PENDING_GPU_VALIDATION
> Extended: Small-Organ-Adaptive Gaussian Refinement (SAGR)

## 1. Objective and constraints

V13 retains the V12 hierarchical Gaussian decomposition, contextual partial
transport, Gaussian velocity synthesis, and four-stage residual pyramid. It
adds a third, explicitly Gaussian refinement stage for anatomy that is smaller
than the factor-four Gaussian representation can reliably resolve.

The implementation has five non-negotiable constraints:

1. inference receives only the moving and fixed images;
2. fixed segmentations supervise training but never enter `model.forward`;
3. local refinement is represented by densified Gaussian primitives rather
   than a second full-volume U-Net;
4. the local output is a bounded stationary velocity field (SVF);
5. the integrated local deformation is composed with the global deformation,
   not added as a raw displacement.

> The user confirmed this design after observing that a self-designed MUSA
> stage-three local refiner substantially improves small-organ registration.
> V13 transfers that validated failure-mode intervention into the
> Gaussian-native model without target-label leakage.

## 2. Data flow and tensor contract

```text
moving/fixed [B,1,D,H,W]
  -> V12 global Gaussian registration
  -> global flow/inverse flow [B,3,D,H,W]
  -> fixed and globally warped moving analytic factor-two measurements
  -> finest fixed Gaussian priority logits [B,N]
  -> top-K parent Gaussian selection
  -> deterministic child Gaussian densification [B,K,M,3]
  -> within-parent child Gaussian correspondence [B,K,M,M]
  -> bounded child residual velocities [B,K,M,3] in mm
  -> Gaussian rasterization -> local residual SVF [B,3,D,H,W]
  -> scaling-and-squaring -> local flow/inverse flow
  -> deformation composition -> final flow/inverse flow
  -> final warped/inverse-warped images
```

`N` is the finest V12 Gaussian count (1024 in production), `K` is a fixed
compute budget, and `M=9` is one centre plus eight three-dimensional corner
children. Top-K changes where compute is spent but not the tensor allocation
budget, so runtime remains predictable.

## 3. File changes

| File | Operation | Responsibility |
|---|---|---|
| `gaussian_native/small_organ_refinement.py` | NEW | Gaussian priority, densification, local correspondence, residual SVF synthesis and composition inputs |
| `gaussian_native/model.py` | MODIFIED | Register V13 and invoke SAGR after the V12 global deformation |
| `gaussian_native/losses.py` | MODIFIED | Regularize the local residual SVF while evaluating topology on the composed final flow |
| `gaussian_native/__init__.py` | MODIFIED | Export the SAGR module and document V13 |
| `experiment_utils.py` | MODIFIED | Parse V13/SAGR configuration and emit diagnostics |
| `train_registration.py` | MODIFIED | Load the union of global/small labels and apply training-only Gaussian-priority supervision |
| `configs/gaussian_native_v13_*.json` | NEW | Dataset-specific V13 training configurations |
| `tests/test_gaussian_native_small_organ_refinement.py` | NEW | Unit tests for zero-init, shapes, composition, and gradients |
| `tests/test_gaussian_native_model.py` | MODIFIED | End-to-end V13 composition and backward test |
| `tests/test_experiment_utils.py` | MODIFIED | V13 builder and mixed-label supervision tests |
| `README.md` | MODIFIED | Explicit V13 smoke/train/evaluation commands |
| `docs/external_head_neck_training.md` | MODIFIED | V13 commands for HaN-Seg, SegRap2023, and CBCT-CT |
| `docs/dev_log.md` | MODIFIED | Implementation decisions, verification, and run instructions |

The original untracked `小器官精修模块/` directory is kept unchanged as the
MUSA-stage prototype and is not imported by production code.

## 4. SAGR implementation

### 4.1 `analytic_measurements(volume, spacing_dhw)`

- Input: `[B,1,d,h,w]` float image and `[B,3]` effective spacing.
- Output: `[B,7,d,h,w]` containing intensity, three physical first
  derivatives, gradient magnitude, Laplacian, and local variance.
- The operation is fixed and contains no learned dense convolution.

### 4.2 `sample_volume_at_physical_points(volume, points_mm, extent_mm)`

- Input volume: `[B,C,d,h,w]`.
- Input points: `[B,...,3]` in physical DHW millimetres.
- Output: `[B,...,C]` through trilinear `grid_sample`.
- This function is also used by the training-only priority-label loss.

### 4.3 `SmallOrganAdaptiveGaussianRefiner`

Constructor parameters include `feature_dim`, `selected_parents`,
`children_per_parent`, `descriptor_dim`, `hidden_dim`, physical search-radius
limits, child-scale fraction, maximum residual millimetres, synthesis factor,
matching temperature, position weight, raster chunk, cutoff sigma, and
integration steps.

`forward(moving, fixed, base_flow, fixed_level, match,
spacing_dhw, extent_mm) -> dict` performs:

1. warp moving with the global flow;
2. calculate factor-two analytic measurements;
3. predict one priority logit per finest fixed Gaussian from fixed/matched
   Gaussian features, match geometry/evidence, and local image residual; the
   learned correction is zero-initialized so early selection follows the
   deterministic residual prior;
4. choose top-K parents and split each into nine physical child Gaussians;
5. encode fixed and globally warped-moving child descriptors;
6. calculate a mutual local transport matrix inside each parent neighbourhood;
7. convert the barycentric child displacement into a bounded residual
   translation, with a zero-initialized direct gain and output head;
8. rasterize the child velocities, upsample the physical SVF, and integrate
   forward/inverse residual deformations;
9. return local fields and diagnostics for composition by the parent model.

At initialization, the local SVF is exactly zero and V13 exactly reproduces
the V12 deformation. Child velocity norms are capped in millimetres.

## 5. Model integration

`GaussianNativeRegistration` accepts architecture revision
`gaussian_native_v13`. For V13 it constructs `small_organ_refiner` after the
existing `residual_pyramid`.

The global and local transformations obey:

```text
phi_final     = phi_global o phi_local
phi_final_inv = phi_local_inv o phi_global_inv
```

For the repository's fixed-grid sampling convention this is implemented with
`compose_displacements(local_flow, global_flow)` and
`compose_displacements(global_inverse_flow, local_inverse_flow)`.

The public non-auxiliary return remains `(warped, flow)`. Auxiliary output adds
`global_flow`, `global_inverse_flow`, `local_residual_velocity_mm`,
`local_residual_velocity_vox`, `local_residual_flow`, and a
`small_organ_refinement` dictionary. Existing V1--V12 behavior remains
unchanged.

## 6. Training-only small-organ supervision

The training dataset loads the ordered union of:

- `supervised_anatomy.labels`, used by the existing global anatomy loss;
- `small_organ_refinement.supervision_labels`, used only to supervise Gaussian
  priority.

For binary-channel datasets, each loss selects its requested channels from the
union explicitly. `response_valid` is selected in the same order, preventing
missing-label leakage.

`_small_organ_priority_loss` unions valid fixed small-organ masks, dilates the
union by a configured physical radius, samples it at the finest Gaussian
centres, and applies balanced BCE plus soft Dice to the priority logits. The
mask is not passed into the model and is not available during validation or
inference.

The final composed flow is always appended to anatomy deep supervision even
when the V12 pyramid already contains a factor-one flow. This is required for
small-organ Dice gradients to reach SAGR.

## 7. Configurations

- HNTS-MRG24: supervise priority around GTVp as a small-target stress test.
- HaN-Seg: supervise Arytenoid, cochleae, lacrimal glands, optic chiasm,
  optic nerves, and pituitary.
- SegRap2023: supervise chiasm, cochleae, ET bones, IACs, lenses, optic nerves,
  pituitary, tympanic cavities, and vestibular semicircular structures.
- Head-Neck-CBCT-CT: keep SAGR disabled because this dataset copy has no
  anatomical labels; V13 remains architecture-compatible without claiming a
  small-organ result on an unobservable endpoint.

Primary ablations are: V12; V13 without adaptive priority (fixed nodes and
unit refinement gates); V13
without densification (one child); V13 without local correspondence; V13 with
raw DVF addition; and full V13. The raw-addition variant is experiment-only and
must not become the default implementation.

## 8. Validation requirements

- All existing tests pass unchanged for V1--V12.
- A V13 zero-initialized SAGR produces exactly the global V12 flow.
- Model forward never accepts segmentations.
- Priority loss can select a label subset from a larger loaded-label union.
- Child centres, local transport, SVF, composed flow, and inverse flow have
  finite expected shapes.
- A backward pass reaches the SAGR output head and priority head.
- The final displacement is bounded and Jacobian metrics remain finite.
- Production A100 smoke for both `128x160x160` and `192x160x160` stays below
  40 GiB before full training is authorized.

## 9. Results and logging

Existing `metrics.jsonl`, TensorBoard, checkpoints, and evaluation JSON remain
compatible. New training diagnostics include priority mean/maximum, selected
priority, local transport entropy, direct gain, local SVF magnitude, local
flow magnitude, and densified Gaussian coverage. Training also records
priority loss, target prevalence, selected-target hit rate, and selected-target
recall when labels are available.

## 10. Implementation order

```text
documentation
  -> small_organ_refinement.py
  -> model/loss/builder integration
  -> training-only priority supervision
  -> V13 configs
  -> tests and smoke
  -> README and external-data commands
  -> complete local test suite
  -> production A100 smoke and real-batch verification on the server
```

## 11. Design validation

- Experiment coverage: every claimed component has a configuration switch or
  a controlled ablation path.
- Logical consistency: all local translations use physical millimetres; SVF
  conversion accounts for voxel spacing; final transformations are composed.
- Fair inference: neither fixed nor moving segmentation is consumed by
  `model.forward`.
- Completeness: model, supervision, diagnostics, configuration, tests,
  documentation, and explicit run commands are included.
