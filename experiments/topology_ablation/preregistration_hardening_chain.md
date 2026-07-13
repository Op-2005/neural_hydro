# Pre-registration — Analysis-Only Hardening Chain (significance → mechanism-confound → metric-honesty)

**Date:** 2026-07-12. **Author:** /crs-unleashed session.
**Compute:** ZERO training. Pure re-analysis of stored `test_results.p` (per-timestep obs/sim)
and depth/edge/area CSVs already on disk. Seeds [11, 13, 17], Component 0 (183 basins).

**Motivation.** The core study is complete and publication-valid. This chain hardens its three
most load-bearing / most-attackable joints, each with a saved artifact, before the paper draft.
All steps are gated: a falsification stops the chain (no re-design of a failed test mid-session).

---

## Step A — Statistical significance of the deployable gain (gates B)

**Hypothesis.** The paired per-basin realizable Δ (L+upQ_pred − L), pooled across 3 seeds
(n≈549 basin×seed), is significantly > 0 AND significantly greater than the null (shuffled) Δ,
by paired Wilcoxon signed-rank tests (non-parametric; the Δ distribution is skewed).

**Success criteria (BOTH required to pass):**
- realizable-vs-L: Wilcoxon signed-rank p < 0.01.
- realizable-vs-null: Wilcoxon signed-rank p < 0.05 on the paired per-basin (realizable Δ − null Δ).
- Report effect size: median Δ, fraction of basins positive, and a bootstrap 95% CI on the median.

**Falsification.** If realizable-vs-null is NOT significant (p ≥ 0.05), the "it is the signal,
not added-input capacity" claim is weaker than stated. Report honestly; DO NOT proceed to Step B
(the mechanism is moot if the effect itself is indistinguishable from capacity).

**Robustness check (bundled).** Also run the per-seed Wilcoxon (n=183 each) — the effect should
be significant within individual seeds, not only when pooled.

## Step B — Routing mechanism vs the aggregation-magnitude confound (gated on A passing)

**Hypothesis.** The depth→Δ gradient reflects upstream *routing*, not the *magnitude/variance of
the upstream_q feature* (deep basins aggregate more/larger upstream area → larger feature values;
the gradient could be "bigger feature → bigger effect"). The gradient survives controlling for
feature magnitude.

**Success criteria:**
- depth≥2 median Δ > depth0 median Δ by ≥ +0.01 within ≥ 2/3 feature-magnitude terciles
  (feature magnitude = per-basin mean |upstream_q|, from the lag-1 predicted feature pickle).
- Report Spearman corr(Δ, depth) and a partial correlation of Δ with depth controlling for BOTH
  area and feature-magnitude.

**Falsification.** If the gradient collapses (depth≥2 ≯ depth0 in ≥2 terciles) once feature
magnitude is controlled, the mechanism is feature-scale, not routing. Report; stop the chain.

**Note on the existing CONFOUND.md tension:** T1 flags "monotonic in n_upstream: False" while the
routing story rests on *depth*. This step clarifies which variable (depth vs raw n_upstream vs
feature magnitude) actually carries the gradient.

## Step C — Metric-honesty pass: log-NSE eps-sensitivity + KGE decomposition (gated on B passing)

**Hypothesis.** (1) The log-NSE realizable Δ (+0.027 reported) is stable across a defensible eps
sweep. (2) The KGE seed-sensitivity (seed-13 dips negative) localizes to one KGE component
(correlation r, bias β, or variability γ), sharpening the honest-scope statement.

**Success criteria:**
- log-NSE realizable Δ stays positive and same order of magnitude for eps ∈ {1e-2, 1e-3, 1e-4}
  × (per-basin mean observed flow). Report the Δ at each eps.
- KGE decomposed into (r, β, γ) per condition; identify which component drives the seed-13 dip.

**Falsification.** If log-NSE Δ flips sign under a defensible eps, the 3-metric-robustness claim
needs re-scoping. Report the eps at which it flips.

---

## Discipline
- Pre-registered before any execution. Amend only by dated append.
- A falsification stops the chain at that step. No mid-session redesign of a failed test.
- Every step writes a saved artifact under `analysis/`.
- Reviewer-2 simulation after each non-trivial result (non-optional for /crs-unleashed).
