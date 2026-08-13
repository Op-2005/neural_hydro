# Completion suite — four experiments, 15 runs (2026-08-12)

Computed from stored per-basin `test_metrics.csv`. Paired per-basin ΔNSE vs `L`,
cross-seed mean of per-seed medians, on the forward-connected basins (n=150/seed)
unless stated. Seeds 11/13/17.

## 1. Precipitation decomposition — shared weather vs shared water

| seed | upPrecip Δ | realizable Δ | realizable−upPrecip |
|---|---|---|---|
| 11 | +0.0160 | +0.0340 | +0.0211 |
| 13 | +0.0229 | +0.0298 | +0.0143 |
| 17 | -0.0021 | +0.0149 | +0.0126 |

Pooled: upstream precipitation **+0.0133**, realizable **+0.0264**, difference
**+0.0148** (paired Wilcoxon one-sided p=2.1e-09), positive at all three seeds.

**Verdict: PASS — discharge adds beyond weather, but only about half the gain is
discharge-specific.** Upstream precipitation, a spatial smooth of a forcing the baseline
already receives, recovers **50%** of the deployable gain.

## 2. Distance sweep — proximity dose-response (seed 11)

| condition | mean edge km | Δ NSE |
|---|---|---|
| forward (true graph) | 92 | +0.0462 |
| distance-matched | 101 | +0.0456 |
| swept 175 km | 175 | +0.0332 |
| swept 250 km | 250 | +0.0240 |
| swept 350 km | 350 | +0.0160 |
| swept 500 km | 500 | +0.0065 |
| random rewire | 511 | +0.0139 |

Spearman(distance, Δ) = **-1.000** (p=0.000); the gain falls monotonically from
**+0.0456** at 101 km to **+0.0065** at 500 km, a drop of 0.0391.

**Verdict: PASS — proximity is a dose-response, not a threshold.** Distance is the
operative variable, measured rather than inferred by elimination.

## 3. Nearest-neighbour baseline — the river graph is not needed, and is worse

| condition | connected (n=150) | all 183 | headwaters (n=33) |
|---|---|---|---|
| L_upQ | +0.0431 | +0.0352 | -0.0109 |
| L_upQpred | +0.0262 | +0.0218 | -0.0009 |
| L_upQknn2 | +0.0806 | +0.0781 | +0.0615 |
| L_upQknn4 | +0.0785 | +0.0738 | +0.0503 |

Paired k-NN(k=2) minus true-graph oracle: median **+0.0341**,
p=1.4e-24, k-NN better on 72% of basin-seeds,
significant at every seed individually.

k-NN neighbours average **46.7 km** (k=2) and 62.7 km (k=4) against **91.6 km** for the
true edges, which by the sweep in §2 is exactly why they carry more signal. The effect is
not an averaging-count artifact: k=2 matches or beats k=4, and the graph's mean in-degree
is 4.16.

**Headwaters gain +0.0618 under k-NN (p=4.8e-12)**,
against −0.008 under the graph input, which assigns them a constant zero by construction.

**Verdict: the river network is not merely unnecessary — it is a worse neighbour-selector**
**than plain distance.** Its edge rule (area ratio ≥1.5, elevation-decreasing, ≤150 km)
admits neighbours roughly twice as far away as the nearest gauges.

## 4. Clean random rewire

| control | per-seed | cross-seed mean |
|---|---|---|
| random (contaminated, 27/624 true edges) | ['+0.014', '+0.019', '+0.003'] | +0.0119 |
| random (clean, 0/624) | ['+0.006', '+0.023', '+0.000'] | +0.0096 |

Shift -0.0023. The clean control sits slightly lower, in the predicted direction,
but the change is small relative to the seed spread.

**Verdict: the contamination was immaterial to any conclusion.** The defect is closed and
the clean number is the one to report.

## What these four results change

1. **The headwater result was definitional.** Under k-NN, headwaters gain +0.06; under the
   graph input they receive a constant zero and cannot gain. The paper must retire
   "the gain vanishes where there are no upstream basins" as evidence for anything.
2. **The depth stratification dissolves.** k-NN Δ by depth is flat (+0.062, +0.079, +0.079,
   +0.082, +0.085, +0.061 at depths 0–5). Depth was a proxy for having a defined input.
3. **Proximity is now measured, not inferred.** A monotone decay from 101 to 500 km replaces
   an argument by elimination.
4. **Half the gain is shared weather.** Upstream precipitation recovers 50% of the
   deployable gain with no discharge and no graph.
5. **The river network is a liability for this purpose.** Deleting it and using the nearest
   gauges nearly doubles the gain.

---

## 5. Min-separation k-NN — the leakage objection, tested (2026-08-13)

Nearest-gauge selection optimises only for distance, so it can pick a gauge a few km away: 19 of
366 pairs at k=2 are under 10 km, the nearest 1.6 km. Those may be nested, making the neighbour's
discharge partly a measurement of the target. Re-run with a floor on neighbour separation.

| condition | connected Δ (3-seed mean) | per-seed | paired vs true network |
|---|---|---|---|
| true network | +0.0431 | +0.046 / +0.056 / +0.027 | — |
| k-NN 2, no floor | +0.0806 | +0.077 / +0.087 / +0.077 | +0.0341 (p=1.4e-24, 72%) |
| k-NN 2, ≥10 km | +0.0730 | +0.072 / +0.090 / +0.057 | +0.0300 (p=5.1e-19, 69%) |
| k-NN 2, ≥15 km | +0.0721 | +0.066 / +0.087 / +0.064 | +0.0279 (p=2.3e-17, 68%) |

**Verdict: PASS — the headline survives.** Excluding near-duplicate gauges costs roughly a fifth of
the margin (+0.034 → +0.028) and none of the conclusion. Nesting between very close gauges is not
the source of the nearest-gauge advantage.

Scope: this rules out *near-duplicate* gauges. Catchment-boundary overlap at larger separations is
a different question and remains unmeasured.

## 6. Deployable k-NN — not yet run

The three runs aborted at training start: the feature files were absent in the Colab VM
(`FileNotFoundError` on `upstream_q_pred_knn2_component0_seed{s}_lag1.p`). The builder depends on
each seed's full-span eval (`_Lfullspan_eval_seed{s}/test/model_epoch030/test_results.p`), which is
symlinked in from the previous runs folder; the build did not complete and the notebook proceeded to
train anyway.

All three features have since been **built locally and verified** (183 basins, `date`-named index),
and the notebook's build cell now checks the full-span dependency up front and raises if a feature
does not materialise, rather than falling through to a training run that cannot work. The
experiment is unchanged and still pending.
