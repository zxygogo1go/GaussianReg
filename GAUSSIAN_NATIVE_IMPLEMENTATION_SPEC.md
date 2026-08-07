# Gaussian-Native Diffeomorphic Registration: Implementation Specification

## 1. Scope

The model registers a moving longitudinal head-and-neck T2 MRI to a fixed T2
MRI. Revisions v1--v9 use Gaussian primitives for all learned intermediate
states and deformation degrees of freedom. Dense voxel tensors are used only
for:

- fixed Gaussian scale-space measurements of the input;
- deterministic Gaussian reconstruction/velocity rasterization;
- scaling-and-squaring integration;
- final image and label resampling.

The new model has no SACB, Gaussian/dense gate, or learned confidence module.
Revision v10 adds a three-stage voxel CNN that predicts bounded residual
stationary velocities after each coarse-to-fine image warp. V11 adds a
gradient-aware fourth stage at full resolution. These revisions are
Gaussian-guided hybrid registration models; v9 remains the strict
Gaussian-only ablation. V12 introduces partial bidirectional correspondence
and a Gaussian-first curriculum. V13 adds a final Small-Organ-Adaptive
Gaussian Refinement (SAGR) stage that returns the local intervention to
Gaussian primitives.

## 2. Coordinate convention

- Array and vector component order is `(D,H,W)`.
- Gaussian centers, scales, covariance, and local velocities are represented
  in physical millimetres.
- The final stationary velocity is converted to voxel units before numerical
  integration.
- The output flow is a fixed-grid to moving-image sampling displacement:

  \[
  I_{\mathrm{warp}}(x)=I_{\mathrm{moving}}(x+u(x)).
  \]

## 3. Module I: Hierarchical Gaussian Representation and Correspondence

### 3.1 Gaussian primitive

At hierarchy level \(s\), primitive \(i\) is:

\[
G_i^s=(\mu_i^s,\Sigma_i^s,m_i^s,z_i^s,a_i^s).
\]

- \(\mu_i^s\in\mathbb R^3\): center in millimetres.
- \(\Sigma_i^s\in\mathbb S_{++}^3\): full SPD covariance.
- \(m_i^s\): normalized representation mass.
- \(z_i^s\): learned anatomy feature.
- \(a_i^s\): Gaussian-window intensity/derivative measurements.

The implementation supports a learned covariance parameterization:

\[
\Sigma=R\,\mathrm{diag}(\sigma_D^2,\sigma_H^2,\sigma_W^2)R^\top+\epsilon I,
\]

where \(R\) is produced by the continuous 6D rotation representation and
scales are strictly positive.

Production v7/v8 instead fixes centres, axis scales, identity rotations, and
uniform mass to a canonical hierarchy. This removes image-dependent geometric
degrees of freedom from the production matcher while retaining explicit SPD
covariance tensors and Gaussian feature/appearance measurements.

### 3.2 Gaussian scale-space

The model applies a fixed separable Gaussian filter before every factor-two
downsampling. The three production levels use factors 16, 8, and 4.

At each level it measures:

- intensity;
- three first spatial derivatives;
- gradient magnitude;
- Laplacian;
- local variance.

No trainable voxel convolution is used.

### 3.3 Anchored mass-conserving hierarchy

- Root lattice: `4×4×4 = 64` primitives.
- Each root produces four children: `256` primitives.
- Each middle primitive produces four children: `1024` primitives.

Children use fixed tetrahedral offsets inside each canonical parent. In
production v7/v8, local Gaussian-window measurements change the appearance and
learned feature of a primitive but cannot move or resize it. The earlier
anatomy-adaptive geometry predictor remains only for revision ablations.

Mass is exactly conserved:

\[
\sum_{c\in\mathcal C(p)}m_c=m_p,\qquad
\sum_i m_i^s=1.
\]

Because production geometry is fixed, centre/scale/mass collapse is impossible
by construction. Reconstruction, coverage, and parent containment remain
reported representation diagnostics for historical ablations but have zero
group weight in the v7/v8 training objective.

### 3.4 Gaussian graph encoder

Each level uses geometry-defined k-nearest-neighbour attention. Edge features
contain:

- relative center in the query Gaussian frame;
- log axis-scale ratios;
- normalized local distance.

Moving and fixed volumes share the complete decomposer and graph encoder.
Parent features are explicitly propagated into the child level.

### 3.5 Coarse-to-fine Gaussian correspondence

The root level performs global all-to-all transport. For each finer level, the
top parent transports define the admissible child correspondence mask.

Production revisions v7/v8 use a fixed appearance/geometric base and a bounded
learned residual. The appearance descriptor is the detached Gaussian-window
intensity/derivative vector, standardized independently in each volume and
L2-normalized. For pair \((i,j)\), the base cost is:

\[
C^{base}_{ij}
 =(1-\cos(\hat a_i^f,\hat a_j^m))
+\lambda_p\left\|
\frac{\mu_i^f-\mu_j^m}{e}
\right\|_2^2
+\lambda_s\left\|\log\sigma_i^f-\log\sigma_j^m\right\|_2^2.
\]

The learned pair scorer receives eight values: appearance similarity, encoded
feature similarity, three signed normalized centre offsets, normalized
distance, log-scale cost, and the absolute difference of the first
standardized appearance channel. Its residual is bounded:

\[
r_{ij}=r_{\max}\tanh f_\theta(x_{ij}),\qquad
\ell_{ij}=-C^{base}_{ij}/T+\lambda_r r_{ij}.
\]

The final layer of \(f_\theta\) is initialized to zero and \(r_{\max}=2\).
The residual multiplier \(\lambda_r\) follows the configured training ramp
(0.35 to 1.0 over 20 epochs in v8). Therefore v7/v8 initially reproduce or
remain close to the deterministic base matcher, and learned features can
refine it while the base term remains present.

Production v9 replaces the eight-value MLP with a bidirectional contextual
matcher. Separate multi-head query/key projections produce an initial
fixed-to-moving affinity. Row- and column-normalized attention then aggregate
moving context for every fixed Gaussian and fixed context for every moving
Gaussian. A shared fusion network combines each node with its transported
context. Refined multi-head similarities are concatenated with the initial
similarities, appearance correlation, signed position, distance, per-axis
log-scale difference, scale cost, appearance difference, and log-mass
difference. A zero-initialized bounded output layer still makes the initial
transport equal to the fixed base matcher. This changes the learned matching
capacity from 963 pair-MLP parameters in v8 to a 1,639,206-parameter
correspondence subsystem in v9.

Production v7--v9 apply row-softmax on the admissible support. Fine-level support
contains only the strongest transported parents; the same-index parent is not
inserted unconditionally. Forbidden entries are explicitly zeroed and
renormalized, so sparse support cannot be undone by a marginal constraint.
Fixed-to-moving and fixed-to-fixed calibration use the exact same support.
Sinkhorn, dustbin, and mutual transport remain implementation ablations but
are not in v7--v9.

The transport yields matched moving features, centers, scales, and covariances
for every fixed Gaussian. Expected transport cost is not optimized in v7--v9
because it admits uniform or feature-collapse shortcuts; registration
similarity supplies the task gradient through the learned portion of the
matching score.

## 4. Module II: Gaussian Stationary Velocity Field Generator

### 4.1 Local affine stationary velocity

For a fixed Gaussian:

\[
v_i(x)=t_i+\left(\Omega(\omega_i)+S_i\right)(x-\mu_i).
\]

- \(t_i\in\mathbb R^3\): translation.
- \(\omega_i\in\mathbb R^3\): bounded rotation vector.
- \(S_i\in\mathbb S^3\): six bounded symmetric strain parameters.

The motion head consumes the fixed feature, transported moving feature, center
offset, log-scale ratio, and relative covariance.

The learned affine-residual layer is zero-initialized. Direct calibrated
transport can still provide nonzero initial motion for different image pairs,
while identical inputs have exactly zero calibrated transport displacement.

### 4.2 Anatomy-calibrated residual motion hierarchy

Revisions v7--v9 compute fixed-base, residual-refined fixed-to-moving
correspondence in the canonical Gaussian frame. Its direct displacement is:

\[
\Delta_i =
\sum_j q^{FM}_{ij}\mu^M_j -
\operatorname{stopgrad}\left(\sum_j q^{FF}_{ij}\mu^F_j\right).
\]

The stopped fixed-to-fixed reference removes soft-correspondence barycentre
bias and keeps identical-input direct displacement exactly zero. Since the v7--v9
geometry is shared and canonical, motion comes from differences between the
cross-image and self-image transport distributions, not from predicted centre
drift. Mutual row/column plan multiplication is not used.

There is no geometry predictor in v7--v9, so matching costs cannot be lowered by
moving the primitives. A low position weight is only a soft anatomical
locality prior, and the matching temperature is cosine-annealed from 0.12 to
0.08.

The root level predicts coarse motion. Middle and fine levels use the
calibrated transport displacement relative to their parent. For candidate
support \(S_i\), ambiguity is measured as:

\[
h_i = \frac{-\sum_{j\in S_i}q_{ij}\log q_{ij}}{\log |S_i|},
\qquad e_i=1-h_i,\qquad \tilde e_i=\sqrt{e_i}.
\]

The deterministic factor \(\tilde e_i\) multiplies only the current level's
residual after parent subtraction. Exact uniform ambiguity still contributes
no new residual and cannot cancel reliable parent motion, while partially
informative middle/fine correspondence is no longer suppressed linearly as in
v5. This is a derived transport statistic, not a trainable confidence
predictor. Child residuals are not forcibly centred:

\[
v=v^{0}+\Delta v^{1}+\Delta v^{2}.
\]

This additive hierarchy avoids counting a global translation three times
while preserving nonzero local motion at the middle and fine levels.

The dense SVF is rasterized with the same canonical centres, scales, identity
rotations, and uniform Gaussian mass used for representation and matching.

### 4.3 Gaussian velocity synthesis

For every level:

\[
v^s(x)=
\frac{
\sum_i m_i^s k_i^s(x)
\left[t_i^s+A_i^s(x-\mu_i^s)\right]
}{
\sum_i m_i^s k_i^s(x)+\epsilon
},
\]

with:

\[
k_i(x)=
\exp\left[
-\frac12(x-\mu_i)^\top\Sigma_i^{-1}(x-\mu_i)
\right].
\]

Kernels are truncated at 3.5 standard deviations and evaluated in node
chunks. The production field is synthesized on the factor-four grid, summed
across hierarchy levels, and physically upsampled before integration.

### 4.4 V10 Gaussian-guided residual image pyramid

Revision v10 does not sum all three Gaussian fields and warp only once.
Instead, the root, middle, and fine Gaussian residual fields are injected
sequentially at image factors 8, 4, and 2. At stage \(s\), accumulated
stationary velocity \(v_{s-1}\) is integrated and used to warp the moving
image. A residual CNN then predicts:

\[
\delta v_s =
\alpha_s\tanh f_s\left(
I^F_s,\,
I^M_s\circ\exp(v_{s-1}),\,
I^F_s-I^M_s\circ\exp(v_{s-1}),\,
\exp(v_{s-1})
\right).
\]

The stage update is additive in stationary-velocity space:

\[
v_s =
\operatorname{up}(v_{s-1}) + v_s^{G} + \delta v_s.
\]

Scale-aware upsampling converts displacement units correctly between pyramid
levels. Residual bounds are 1.5, 1.0, and 0.75 voxels at factors 8, 4, and 2.
Each residual output convolution is initialized to zero, so v10 starts from
the Gaussian deformation rather than an unrelated dense field.

### 4.5 V11 gradient-aware full-resolution refinement

V11 appends a factor-one residual stage without inventing a fourth Gaussian
level. The first three stages still receive the root/middle/fine Gaussian
velocity components; the fourth stage refines their accumulated stationary
velocity. Its input adds fixed and warped-moving forward-difference gradient
magnitudes:

\[
x_1 = \left[
I^F,\,
I^M\circ\exp(v_2),\,
I^F-I^M\circ\exp(v_2),\,
\exp(v_2),\,
\|\nabla I^F\|,\,
\|\nabla(I^M\circ\exp(v_2))\|
\right].
\]

The factor-one stage has 16 channels, two residual blocks, and a 0.35-voxel
velocity bound. Its output head is zero-initialized. Therefore adding the stage
does not perturb the initial v10 deformation, while training can learn
sub-voxel boundary corrections that are not representable at factor two.

### 4.6 Diffeomorphic integration

The physical velocity is converted to DHW voxel units and integrated using
seven scaling-and-squaring steps:

\[
\phi=\exp(v),\qquad \phi^{-1}=\exp(-v).
\]

The model returns both forward and inverse flows. Through revision v9, no
learned operation follows Gaussian velocity synthesis. In v10, the accumulated
factor-two stationary velocity is physically upsampled to full resolution and
then integrated. In v11, the factor-one residual is added before the final
integration.

## 5. Training objective

The objective has four reported groups:

1. **Similarity**
   - bidirectional multi-scale local NCC;
   - half-resolution normalized-gradient agreement.
2. **Representation**
   - scale-space reconstruction;
   - Gaussian coverage;
   - parent/child containment;
   - reported for diagnostics, with production v7--v9 group weight zero because
     geometry is fixed.
3. **Correspondence**
   - reported expected transport cost, cycle error, and hierarchy consistency;
   - production v7--v9 group weight is zero to avoid the self-minimizing
     correspondence shortcut.
4. **Deformation**
   - SVF derivative energy;
   - forward/inverse composition;
   - Jacobian safety barrier;
   - small velocity magnitude penalty.

Through revision v9, labels are never loaded by the training dataset.
Validation and test labels are used only for response-aware Dice, HD95, and
ASSD.

Production v8 used a correspondence-first curriculum. During epochs 1--5, the
learned velocity-head parameters are frozen while differentiable direct
Gaussian transport keeps the image-similarity gradient connected to the
encoder and residual pair scorer. From epoch 6 onward, the velocity head is
unfrozen and the complete model is optimized jointly. LR warmup uses the same
five-epoch boundary. This is a training strategy rather than an additional
prediction module.

Production v9 instead uses a supervised synthetic-deformation curriculum
without anatomical labels. A coarse random stationary velocity \(u\), expressed
in millimetres, is trilinearly interpolated, bounded, augmented with a random
translation, and integrated as \(\phi=\exp(u)\). For an observed training image
\(I\), the exact synthetic pair is

\[
I^M=I,\qquad I^F=I\circ\phi.
\]

The predicted dense sampling flow is supervised against \(\phi\), and each
level's calibrated Gaussian transport displacement is supervised against
\(\phi\) sampled at its canonical centres. Both use normalized Smooth-L1
losses. Epochs 1--8 use synthetic pairs exclusively; epochs 9--60 sample them
with probability 0.25 and otherwise use the real longitudinal pair. The
velocity head remains trainable because the dense target supervises the whole
representation--matching--motion chain. This curriculum teaches sensitivity to
known non-identity anatomy while allowing correct same-index correspondence
after affine prealignment.

Revision v10 disables the synthetic curriculum and loads paired GTVp/GTVn
segmentations for training. A class is valid only when it is present in both
the unwarped moving and fixed masks, which prevents response-related
appearance/disappearance from being treated as a registration error.
Trilinearly downsampled one-hot masks supervise the three pyramid flows and the
final full-resolution flow with normalized weights 0.1/0.2/0.3/0.4. The
soft-Dice denominator uses squared probabilities so identical soft masks have
zero loss. A final soft-boundary Dice term receives weight 0.2. The combined
anatomy factor ramps from 0.2 to 1.0 over 15 epochs. This setting is
segmentation-supervised and must not be compared as if it were unsupervised.

Revision v11 retains response-aware masking but weights GTVp/GTVn losses
1.5/1.0 based on the measured small-target failure. The factor-8/4/2/1 Dice
weights are 0.10/0.15/0.25/0.50. In addition to forward Dice and boundary Dice,
the objective includes normalized soft-centroid distance and final inverse
Dice with weights 0.15 and 0.20. Synchronized left-right flipping transforms
both images and both segmentations; longitudinal pair reversal remains
disabled.

Revision v12 addresses forced correspondence and late supervised overfitting.
For each level it solves a bidirectional, KL-relaxed Sinkhorn problem over the
real Gaussian nodes plus an unmatched dustbin. The relaxed plan may depart from
the prescribed row and column marginals. A Gaussian's motion evidence is its
support concentration multiplied by its transported real-mass fraction; a
sharp match with negligible transported mass therefore cannot create a strong
velocity. The implementation reports real transport mass, unmatched mass in
both directions, and marginal error.

V12 also uses a Gaussian-first curriculum. Epochs 1--15 contain only synthetic
pairs with exact dense and Gaussian displacement supervision. The
zero-initialized residual image pyramid is frozen through epoch 20. Real-pair
anatomy supervision starts at epoch 31 and ramps only from 0.05 to 0.40, while
moving and fixed intensities are augmented independently. This curriculum is a
training strategy, not a third prediction module.

Revision v13 retains the complete v12 global path and adds SAGR after its
full-resolution deformation. SAGR predicts image-derived priority on the
finest fixed Gaussian set, selects a fixed compute budget, densifies each
selected primitive into one centre and eight corner children, and performs a
second within-parent Gaussian correspondence against the globally warped
moving image. The child translations are rasterized into a bounded residual
SVF, integrated by scaling-and-squaring, and composed with the global
deformation. Both its direct gain and learned velocity output are initialized
to zero, so v13 exactly reproduces v12 before optimization.

Small-organ masks supervise only the priority logits during training. They are
not accepted by `model.forward` and are absent at validation and inference.
The final composed flow receives the strongest anatomy deep-supervision weight,
while local SVF smoothness, energy, inverse consistency, and the final Jacobian
barrier constrain the refinement.

## 6. V11/V12/V13 experimental configuration

- Input: `128×160×160`, 1.5 mm isotropic.
- Gaussian counts: `64/256/1024`.
- Feature width: 160.
- Graph encoder: three blocks per level, eight heads, 20 neighbours.
- Production geometry: anchored centres/scales, identity rotations, uniform
  mass; no learned geometry predictor.
- Production transport: strict-support row-softmax.
- Contextual pair scorer: eight heads, 256 context channels, 320 fusion
  channels, and 160 pair-score hidden channels.
- Fixed appearance base weight: cosine 0.85 to 0.55 over 50 epochs.
- Learned residual multiplier: cosine 0.10 to 0.40 over 40 epochs.
- Maximum learned residual logit before the multiplier: 1.5.
- Matching temperature: cosine 0.16 to 0.11 over 60 epochs.
- Position weight: 0.03.
- Production dustbin mass: 0.
- Production motion: translation-only local Gaussian residuals.
- Direct residual fractions: 0.45/0.70/0.60 for root/middle/fine levels.
- Match-evidence power: 0.5.
- Local samples: `3×3×3` per primitive.
- Residual image pyramid: factors 8/4/2/1, channels 48/40/32/16, block counts
  3/3/3/2.
- Residual velocity limits: 1.5/1.0/0.75/0.35 stage voxels.
- Gradient-magnitude features: enabled for residual refinement.
- Integration steps: 7.
- Trainable parameters: 5,380,790.
- Production-shape peak A100 allocation: approximately 10.2 GiB.
- Training autocast: bfloat16.
- Response-aware GTVp-weighted Dice, boundary, centroid, and inverse Dice.
- Anatomy-supervision ramp: 15 epochs.
- Shared left-right image/label flip probability: 0.5.
- Synthetic-deformation training: disabled.
- Learning-rate warmup: 5 epochs.
- Training epochs: 150.
- AMP weight cache: disabled to preserve gradients across no-gradient
  fixed-to-fixed calibration and trainable fixed-to-moving matching.
- Geometry, transport, rasterization, integration, and NCC: float32.

V12 changes only the matching and training protocol needed for a controlled
comparison: transport is bidirectional unbalanced Sinkhorn with dustbin mass
0.18, marginal relaxation 0.90, and 16 iterations. It uses the 15/20/31 epoch
synthetic-pretrain/refinement-unfreeze/anatomy-start boundaries described above;
the architecture width and four-stage refinement capacity remain unchanged.

V13 keeps those global settings. HNTS-MRG24 selects 48 finest parents and
creates nine children per parent; HaN-Seg and SegRap2023 select 64 parents.
The resulting production model has 5,597,405 trainable parameters, an increase
of 216,615 over v12.
Local child correspondence operates at factor two, residual translation is
bounded to 2.5--3.0 mm, and the local SVF is integrated with seven
scaling-and-squaring steps. HaN-Seg and SegRap2023 use their configured
small-organ masks only for delayed Gaussian-priority supervision.
Head-Neck-CBCT-CT keeps SAGR disabled because its current preprocessed copy has
no anatomical labels for validating the small-organ endpoint.

## 7. Required comparisons and ablations

Main comparison:

- original SACB-Net;
- strongest retained historical result reported from its archived metrics;
- strict Gaussian-only v9 model;
- segmentation-supervised Gaussian-guided v10 model;
- small-target-refined Gaussian-guided v11 model.
- partial-correspondence Gaussian-first v12 model.
- small-organ-adaptive Gaussian-refined v13 model.

Required ablations:

- one scale versus three-scale hierarchy;
- diagonal versus full covariance;
- forced matching versus partial matching;
- translation only versus SE(3) versus local affine velocity;
- direct displacement versus stationary velocity integration;
- without transport cycle/hierarchy consistency;
- without inverse consistency;
- fixed-base matcher versus fixed-base plus learned residual scorer;
- adaptive versus anchored Gaussian geometry.
- Gaussian-only synthesis versus Gaussian-guided residual pyramid;
- v10 without label supervision versus Dice only versus Dice plus boundary;
- shallow versus three-stage residual refinement.
- v11 without factor-one refinement;
- v11 without GTVp label weighting;
- v11 without centroid/inverse supervision.
- v12 forced row-softmax versus partial Sinkhorn;
- v12 without synthetic Gaussian pretraining;
- v12 without delayed weak anatomy supervision.
- v13 without adaptive priority (fixed nodes and unit refinement gates);
- v13 without child densification;
- v13 without local child correspondence;
- v13 without Gaussian-priority supervision;
- v13 local residual composition versus raw displacement addition as an
  experiment-only ablation.

The implementation exposes `model.motion_mode` (`translation`, `se3`, or
`affine`), `model.integration_mode` (`direct` or `svf`), and a zero
`model.dustbin_mass` for the principal motion/integration/matching ablations.
`model.covariance_mode` selects `diagonal` or `full` Gaussian geometry.

Final claims require patient-level paired testing and at least three random
seeds. Topology improvement alone is not treated as sufficient evidence of
better registration accuracy.
