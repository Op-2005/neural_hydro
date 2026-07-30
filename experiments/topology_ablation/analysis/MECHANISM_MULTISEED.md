# Mechanism Experiments — 3-Seed Consolidation

Seeds 11/13/17, observed/predicted-Q as noted, stock cudalstm. Supersedes the single-seed
`DIRECTIONALITY.md` and `K2_GRAPH_CHECK.md` framings. Pre-reg:
`preregistration_multiseed_mechanism.md`. Δ = paired vs L on the relevant connected set.

## 1. Directionality & topology-specificity (forward-connected basins, n=150/seed)

| edge set | per-seed Δ (11/13/17) | pooled median Δ | pooled p vs L |
|---|---|---|---|
| **forward** (true upstream) | +0.046 / +0.056 / +0.027 | **+0.046** | 7.6e-23 |
| **reversed** (downstream) | +0.026 / +0.045 / +0.024 | +0.031 | 3.9e-16 |
| **random** (rewired) | +0.014 / +0.019 / +0.003 | +0.012 | 8.7e-04 |

### Topology-specificity — STRONG, CONFIRMED
Forward − random (paired, pooled): **+0.034, p=2.3e-14.** The gain requires the real river
structure; a degree-preserving random rewire retains only ~26% of it. This holds decisively across
all 3 seeds. **This is the headline mechanism result** and the clean mirror of Kirschstein's
topology-*insensitive* GNNs — our feature is sharply topology-*sensitive*.

### Directionality — MILD, POOLED-DETECTABLE PREFERENCE (not a headline claim)
Forward − reversed (paired): the pre-registered hypothesis is directional (H1: forward improves),
so the one-sided test applies. **Pooled one-sided p ≈ 0.03** (significant) — but the effect is
**small and seed-fragile**, and this must be stated together:
- median only **+0.006**; only **54%** of basins favor forward (barely above chance).
- per-seed one-sided p = 0.19 / 0.07 / **0.23**; at **seed 17 reversed ≈ forward** (Δ −0.001).
- reversed retains 57% / 79% / 90% of forward across seeds.

**Verdict:** directionality is a **mild preference detectable only in aggregate (n=450),
carried by 2 of 3 seeds — NOT a robust or strong effect.** The pre-registered falsification
("reversed ≈ forward → generic correlation") is *not* triggered (forward does exceed reversed in
pooled aggregate), but neither is a strong directional claim supported. The physical reason
(pre-registered) explains the weakness: downstream flow is heavily weather-correlated with the
target and carries the basin's own routed water, so reversing edges cannot destroy the signal.

**Scope decision:** headline the **topology-specificity** (real edges ≫ random, strong, every
seed). Report directionality honestly as *"a mild, aggregate-detectable preference for the
physically-correct upstream direction"* — do **NOT** claim "direction-sensitive" or lean on it.
The 3-seed run's real value was preventing that overclaim: at single seed (11) the direction gap
looked cleaner than it is.

## 2. k=2 graph-robustness — over-connectivity defense holds at the model level (k2-connected, n=150)

| condition | per-seed Δ (11/13/17) | pooled median Δ | pooled p vs L |
|---|---|---|---|
| full-graph realizable | +0.034 / +0.030 / +0.015 | +0.026 | 5.1e-21 |
| **k=2 realizable** | +0.021 / +0.033 / +0.021 | **+0.025** | 1.3e-14 |
| k=2 oracle | +0.049 / +0.074 / +0.048 | +0.059 | 2.8e-43 |

**CONFIRMED across 3 seeds.** The k=2 (in-degree≤2, hydrography-realistic) realizable gain
(+0.025) is statistically indistinguishable from the full-graph realizable (+0.026) — the routing
benefit does **not** depend on the heuristic graph's over-connectivity, at the trained-model level,
multi-seed. The k=2 oracle even strengthens (+0.059), consistent with pruning distant/weak parents
sharpening the observed signal. The over-connectivity threat — the study's biggest prior validity
concern — is now closed multi-seed at both the R1-proxy and the LSTM level.

## Net for the paper

- **Topology-specificity: strong, 3-seed, significant.** Real edges ≫ random rewire (p=2e-14).
  Headline mechanism result.
- **Directionality: mild aggregate-detectable preference, not a headline.** Pooled one-sided
  p≈0.03 but +0.006 median, 54% of basins, null at seed 17. Do NOT overclaim direction-sensitivity.
- **Graph-robustness: strong, 3-seed.** k=2 ≈ full-graph realizable — the result is not a
  heuristic-edge artifact.

Every load-bearing mechanism claim is now multi-seed. Writing is unblocked.
