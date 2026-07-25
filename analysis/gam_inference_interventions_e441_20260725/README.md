# GAM-SACB-Net inference intervention report

## Scope

- Checkpoint: GAM-SACB-Net epoch 441, validation-NCC selected.
- Checkpoint SHA-256: `c8c7126acf211e39175bd90e97696435c8e15a198cd35bad0128fcc95ff66a06`.
- Test manifest SHA-256: `73e6228c238bf74b5823676aa9335a15a5ecf1f17f327466d93d9a8a723e215c`.
- Cohort: 16 patient-disjoint test pairs; 15 have response-aware Dice.
- No weights were updated.
- Effects are intervention minus learned inference, with patient-paired 10,000-sample bootstrap confidence intervals.
- Gaussian spatial roll uses five deterministic seeds and is averaged within each patient before patient-level inference.

## Main results

| Intervention | ΔNCC | ΔDice | Δnegative-Jacobian ratio | Δmean displacement | Interpretation |
|---|---:|---:|---:|---:|---|
| Gaussian displacement zero | +0.00026 `[-0.00065, +0.00120]` | -0.00009 `[-0.00268, +0.00232]` | -0.00011 `[-0.00034, +0.00019]` | -1.35 mm `[-1.94, -0.87]` | Accuracy/topology practically equivalent; deformation is smaller |
| Gaussian joint spatial roll, 5 seeds | -0.00597 `[-0.00917, -0.00345]` | -0.01436 `[-0.02154, -0.00734]` | +0.00004 `[-0.00033, +0.00064]` | -0.58 mm `[-0.78, -0.40]` | Spatially located Gaussian/context information matters |
| Gaussian base only, residual off | -0.00823 `[-0.01384, -0.00344]` | -0.02578 `[-0.05346, -0.00624]` | -0.00048 `[-0.00084, -0.00011]` | -3.70 mm `[-4.52, -2.84]` | Gaussian base flow alone is insufficient |
| Dense base only, residual off | -0.01769 `[-0.02242, -0.01328]` | +0.00046 `[-0.01217, +0.02012]` | +0.01039 `[+0.00170, +0.02572]` | -0.43 mm `[-1.54, +0.67]` | Dense base alone loses NCC and strongly worsens folding |

`Gaussian displacement zero` zeros both GACM flow and its normalized displacement channels at levels 5 and 4, while retaining confidence/covariance/anisotropy context, the trained residual head, and downstream refinements.

`Gaussian joint spatial roll` cyclically rolls GACM flow and all 11 context channels together before conditioned SACB and GCDR. It preserves value distributions and local smoothness while breaking anatomy-location correspondence.

## Gate sweep

The trained residual head remains enabled in this sweep.

| Forced gate | ΔNCC | ΔDice | Δnegative-Jacobian ratio |
|---:|---:|---:|---:|
| 0.00 | -0.02019 | -0.00054 | +0.01398 |
| 0.25 | -0.01117 | +0.00119 | +0.00585 |
| 0.50 | -0.00447 | -0.00089 | +0.00060 |
| 0.75 | -0.00067 | -0.00183 | +0.00002 |
| 1.00 | -0.00084 | -0.01120 | -0.00002 |

Forcing less Gaussian base flow harms NCC and, below gate 0.5, topology, while Dice remains practically unchanged through gate 0.75. Forcing a spatially constant gate of 1.0 has a larger negative Dice mean, suggesting that the remaining learned spatial/dense contribution is not completely dispensable.

## Interpretation

1. The direct Gaussian displacement is not carrying the measured accuracy benefit in this trained checkpoint. Removing it leaves NCC, Dice, HD95, ASSD, and folding practically equivalent while substantially reducing deformation magnitude.
2. The full GACM/GCDR path is not replaceable by arbitrary spatial information. Jointly relocating Gaussian flow and context produces paired harm to NCC and Dice.
3. The trained residual head and downstream refinement are central. Gaussian base flow without the residual significantly harms both NCC and Dice; dense base flow without the residual harms NCC and folding.
4. The most plausible useful signal is spatial geometry context interacting with the residual and downstream dense refinements, rather than direct convex replacement of SACB dense flow by Gaussian flow.
5. These are checkpoint interventions, not retrained architecture ablations. They diagnose co-adapted dependence but do not predict exactly how a simplified model will retrain.

## Design implication

The next trained version should test GACM primarily as a geometry-context provider and make GCDR a bounded, confidence-aware residual corrector. A clean ablation should compare:

1. conditioned SACB dense flow;
2. conditioned dense flow plus geometry residual;
3. geometry residual with no direct Gaussian displacement;
4. the current convex Gaussian/dense fusion.

## Files

- `paired_analysis/paired_intervention_summary.json`: complete metadata, paired statistics, condition definitions, and patient-level values.
- `paired_analysis/paired_intervention_summary.csv`: one row per intervention and metric.
- `paired_analysis/per_patient_deltas.csv`: long-form patient-level paired effects.
- Each condition directory contains its original `evaluation.json` and `per_patient_metrics.csv`.

The practical thresholds used by the analysis are exploratory diagnostic margins, not established clinical MCIDs.
