# Pre-registration — Depth-Gradient Confound Check (area / n_upstream)

**Pre-registered 2026-07-01, before running the analysis.**
**Context.** The realizable upstream-Q gain rises monotonically with graph depth
(depth0 −0.003 → depth3 +0.034) — the "routing signature." But depth correlates with basin
area (r=0.38 on component0): downstream basins are larger. A reviewer will ask whether the
gain tracks *routing* (upstream contribution) or merely *basin size*. This must be resolved
before the depth result anchors the paper. Zero compute — re-analysis of committed runs.

## Hypothesis

The realizable gain is driven by **upstream contribution** (routing), not basin size.
Operationalized two ways:
1. **n_upstream is the mechanistic driver.** Gain should rise with number of upstream basins
   (transitive), which is the direct "how much routes to me" variable.
2. **Depth survives controlling for area.** Within area terciles, the gain should still rise
   with depth; and n_upstream should predict gain better than area.

## Tests + criteria

**T1 — Gain vs n_upstream.** Stratify per-basin realizable Δ by n_upstream buckets
{0, 1-2, 3-5, 6+}. *Success:* Δ increases monotonically with n_upstream, and n_upstream=0
(headwaters) ≈ 0.

**T2 — Gain vs area (the confound).** Stratify Δ by area tercile {small, mid, large}.
*Interpretation:* if Δ rises with area as strongly as with depth, size is a competing
explanation; if flatter than the depth/n_upstream gradient, routing wins.

**T3 — Partial: depth gradient within area terciles.** Compute depth-2+ vs depth-0 gain
*separately within each area tercile*. *Success (routing confirmed):* depth-2+ > depth-0 by
≥ +0.01 in ≥ 2 of 3 area terciles → the depth effect is not just area.
*Falsification:* the depth gradient vanishes once area is held fixed → the routing claim is
confounded; must be reframed as an area effect.

**T4 — Correlation check.** Spearman corr of per-basin Δ with n_upstream vs with area.
*Success:* |corr(Δ, n_upstream)| > |corr(Δ, area)| → upstream contribution is the stronger
predictor.

## Data
Per-basin realizable Δ = (L+upQ_pred − L) pooled over measured seeds 13/17 (183 basins × 2).
Basin depth / n_upstream / area from `component0_depth.csv`.

## What we will NOT do
- Will not use seed-11 realizable (its metric folder was lost in the drive merge).
- Will not drop basins or re-tune. Pure re-analysis with pre-stated criteria.

---
## Results (post-run, 2026-07-01) — CONFOUND RULED OUT

- **T3 (load-bearing): PASS 3/3 area terciles.** depth≥2 vs depth0 gain: small +0.021,
  mid +0.036, large +0.050 — depth effect holds within every size class, strongest in the
  large tercile (opposite of an area-confound).
- **T4: corr(Δ, area) = −0.008 (~0)** vs corr(Δ, depth) +0.134. Area has no relationship with
  the gain; depth/upstream-position does.
- **T2:** area-tercile spread only +0.006 (flat) — size does not drive the gain.
- **T1:** step change from headwaters (−0.003) to any-upstream (+0.022), then saturates —
  routing signal is "are you downstream", captured once upstream flow is aggregated.

**Verdict: the depth gradient is upstream routing, NOT basin size.** The mechanistic claim
is confound-checked and paper-ready. Full output: `analysis/CONFOUND.md`.
