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

Covariance is parameterized as:

\[
\Sigma=R\,\mathrm{diag}(\sigma_D^2,\sigma_H^2,\sigma_W^2)R^\top+\epsilon I,
\]

where \(R\) is produced by the continuous 6D rotation representation and
scales are strictly positive.

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

### 3.3 Anatomy-adaptive hierarchy

- Root lattice: `4×4×4 = 64` primitives.
- Each root produces four children: `256` primitives.
- Each middle primitive produces four children: `1024` primitives.

Children are initialized at tetrahedral offsets in the parent ellipsoid. Local
Gaussian-window measurements and the parent feature predict bounded changes
to center, scale, rotation, and mass.

Mass is exactly conserved:

\[
\sum_{c\in\mathcal C(p)}m_c=m_p,\qquad
\sum_i m_i^s=1.
\]

Scale-space reconstruction, coverage, and parent containment prevent collapse
into arbitrary tokens or background-only primitives.

### 3.4 Gaussian graph encoder

Each level uses geometry-defined k-nearest-neighbour attention. Edge features
contain:

- relative center in the query Gaussian frame;
- log axis-scale ratios;
- normalized local distance.

Moving and fixed volumes share the complete decomposer and graph encoder.
Parent features are explicitly propagated into the child level.

### 3.5 Coarse-to-fine partial transport

The root level performs global all-to-all transport. For each finer level, the
top parent transports define the admissible child correspondence mask.

The matching cost combines:

\[
C_{ij}
=1-\cos(z_i^f,z_j^m)
+\lambda_p\left\|
\frac{\mu_i^f-\mu_j^m}{e}
\right\|_2^2
+\lambda_s\left\|\log\sigma_i^f-\log\sigma_j^m\right\|_2^2.
\]

Entropic Sinkhorn transport uses Gaussian masses as marginals and adds one
dustbin row/column for unmatched anatomy. The transport yields matched moving
features, centers, scales, and covariances for every fixed Gaussian.

The dustbin is not a standalone confidence head and is never used to gate two
deformation branches.

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

The final motion layer is zero-initialized, so an untrained model is exactly
the identity deformation.

### 4.2 Identity-calibrated residual motion hierarchy

Revision v2 subtracts a fixed-to-fixed self-transport barycentre from the
fixed-to-moving transport barycentre. This removes entropic matching bias,
gives exactly zero direct displacement for identical inputs, and lets
correspondence drive motion without a confidence or scale gate.

The root level predicts coarse motion. Middle and fine levels use the
calibrated transport displacement relative to their parent. Child residuals
are not forcibly centred; their mass-weighted parent mean is regularized
softly in the deformation objective:

\[
v=v^{0}+\Delta v^{1}+\Delta v^{2}.
\]

This additive hierarchy avoids counting a global translation three times
while preserving nonzero local motion at the middle and fine levels.

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
   - parent/child containment.
3. **Correspondence**
   - expected transport cost;
   - transport cycle error;
   - parent/child transport consistency.
4. **Deformation**
   - SVF derivative energy;
   - forward/inverse composition;
   - Jacobian safety barrier;
   - small velocity magnitude penalty.

Labels are never loaded by the training dataset. Validation and test labels are
used only for response-aware Dice, HD95, and ASSD.

## 6. Production configuration

- Input: `128×160×160`, 1.5 mm isotropic.
- Gaussian counts: `64/256/1024`.
- Feature width: 128.
- Graph encoder: three blocks per level, eight heads, 16 neighbours.
- Partial Sinkhorn iterations: 12.
- Local samples: `3×3×3` per primitive.
- Integration steps: 7.
- Trainable parameters: 2,082,046.
- Training autocast: bfloat16.
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
- without inverse consistency.

The implementation exposes `model.motion_mode` (`translation`, `se3`, or
`affine`), `model.integration_mode` (`direct` or `svf`), and a zero
`model.dustbin_mass` for the principal motion/integration/matching ablations.
`model.covariance_mode` selects `diagonal` or `full` Gaussian geometry.

Final claims require patient-level paired testing and at least three random
seeds. Topology improvement alone is not treated as sufficient evidence of
better registration accuracy.
