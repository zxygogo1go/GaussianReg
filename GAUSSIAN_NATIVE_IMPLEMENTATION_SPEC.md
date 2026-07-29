# Gaussian-Native Diffeomorphic Registration: Implementation Specification

## 1. Scope

The model registers a moving longitudinal head-and-neck T2 MRI to a fixed T2
MRI. Learned intermediate states and learned deformation degrees of freedom
are Gaussian primitives. Dense voxel tensors are used only for:

- fixed Gaussian scale-space measurements of the input;
- deterministic Gaussian reconstruction/velocity rasterization;
- scaling-and-squaring integration;
- final image and label resampling.

The new model has no SACB, voxel CNN feature hierarchy, dense displacement
head, Gaussian/dense fusion, confidence gate, or learned dense residual.

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

Production v7/v8 apply row-softmax on the admissible support. Fine-level support
contains only the strongest transported parents; the same-index parent is not
inserted unconditionally. Forbidden entries are explicitly zeroed and
renormalized, so sparse support cannot be undone by a marginal constraint.
Fixed-to-moving and fixed-to-fixed calibration use the exact same support.
Sinkhorn, dustbin, and mutual transport remain implementation ablations but
are not in v7/v8.

The transport yields matched moving features, centers, scales, and covariances
for every fixed Gaussian. Expected transport cost is not optimized in v7/v8
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

Revisions v7/v8 compute fixed-base, residual-refined fixed-to-moving
correspondence in the canonical Gaussian frame. Its direct displacement is:

\[
\Delta_i =
\sum_j q^{FM}_{ij}\mu^M_j -
\operatorname{stopgrad}\left(\sum_j q^{FF}_{ij}\mu^F_j\right).
\]

The stopped fixed-to-fixed reference removes soft-correspondence barycentre
bias and keeps identical-input direct displacement exactly zero. Since the v7/v8
geometry is shared and canonical, motion comes from differences between the
cross-image and self-image transport distributions, not from predicted centre
drift. Mutual row/column plan multiplication is not used.

There is no geometry predictor in v7/v8, so matching costs cannot be lowered by
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

### 4.4 Diffeomorphic integration

The physical velocity is converted to DHW voxel units and integrated using
seven scaling-and-squaring steps:

\[
\phi=\exp(v),\qquad \phi^{-1}=\exp(-v).
\]

The model returns both forward and inverse flows. No learned operation is
applied after Gaussian velocity synthesis.

## 5. Training objective

The objective has four reported groups:

1. **Similarity**
   - bidirectional multi-scale local NCC;
   - half-resolution normalized-gradient agreement.
2. **Representation**
   - scale-space reconstruction;
   - Gaussian coverage;
   - parent/child containment;
   - reported for diagnostics, with production v7/v8 group weight zero because
     geometry is fixed.
3. **Correspondence**
   - reported expected transport cost, cycle error, and hierarchy consistency;
   - production v7/v8 group weight is zero to avoid the self-minimizing
     correspondence shortcut.
4. **Deformation**
   - SVF derivative energy;
   - forward/inverse composition;
   - Jacobian safety barrier;
   - small velocity magnitude penalty.

Labels are never loaded by the training dataset. Validation and test labels are
used only for response-aware Dice, HD95, and ASSD.

Production v8 uses a correspondence-first curriculum. During epochs 1--5, the
learned velocity-head parameters are frozen while differentiable direct
Gaussian transport keeps the image-similarity gradient connected to the
encoder and residual pair scorer. From epoch 6 onward, the velocity head is
unfrozen and the complete model is optimized jointly. LR warmup uses the same
five-epoch boundary. This is a training strategy rather than an additional
prediction module.

## 6. Production configuration

- Input: `128×160×160`, 1.5 mm isotropic.
- Gaussian counts: `64/256/1024`.
- Feature width: 128.
- Graph encoder: three blocks per level, eight heads, 16 neighbours.
- Production geometry: anchored centres/scales, identity rotations, uniform
  mass; no learned geometry predictor.
- Production transport: strict-support row-softmax.
- Fixed appearance base weight: 1.0 throughout training.
- Learned residual multiplier: cosine 0.35 to 1.0 over 20 epochs.
- Maximum learned residual logit before the multiplier: 2.0.
- Matching temperature: cosine 0.12 to 0.07 over 40 epochs.
- Position weight: 0.03.
- Production dustbin mass: 0.
- Production motion: translation-only local Gaussian residuals.
- Direct residual fractions: 0.55/0.85/0.75 for root/middle/fine levels.
- Match-evidence power: 0.5.
- Local samples: `3×3×3` per primitive.
- Integration steps: 7.
- Trainable parameters: 1,906,522.
- Training autocast: bfloat16.
- Correspondence-first/learning-rate warmup: 5 epochs.
- AMP weight cache: disabled to preserve gradients across no-gradient
  fixed-to-fixed calibration and trainable fixed-to-moving matching.
- Geometry, transport, rasterization, integration, and NCC: float32.

## 7. Required comparisons and ablations

Main comparison:

- original SACB-Net;
- strongest retained historical result reported from its archived metrics;
- Gaussian-native model.

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

The implementation exposes `model.motion_mode` (`translation`, `se3`, or
`affine`), `model.integration_mode` (`direct` or `svf`), and a zero
`model.dustbin_mass` for the principal motion/integration/matching ablations.
`model.covariance_mode` selects `diagonal` or `full` Gaussian geometry.

Final claims require patient-level paired testing and at least three random
seeds. Topology improvement alone is not treated as sufficient evidence of
better registration accuracy.
