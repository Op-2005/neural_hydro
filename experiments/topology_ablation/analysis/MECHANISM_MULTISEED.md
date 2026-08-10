# Mechanism Experiments — 3-Seed Consolidation

Seeds 11/13/17, observed/predicted-Q as noted, stock cudalstm. Supersedes the single-seed
`DIRECTIONALITY.md` and `K2_GRAPH_CHECK.md` framings. Pre-reg:
`preregistration_multiseed_mechanism.md`. Δ = paired vs L on the relevant connected set.

## 1. Directionality & topology-specificity (forward-connected basins, n=150/seed)

| edge set | per-seed Δ (11/13/17) | pooled median Δ | pooled p vs L |
|---|---|---|---|
| **forward** (true upstream) | +0.046 / +0.056 / +0.027 | **+0.046** | 7.6e-23 |
| **distance control** (nearby non-parents, distance-matched) | +0.046 / +0.060 / +0.041 | **+0.049** | ~0 (n=450) |
| **reversed** (downstream, true pairs flipped) | +0.026 / +0.045 / +0.024 | +0.031 | 3.9e-16 |
| **random** (rewired, ~511 km edges) | +0.014 / +0.019 / +0.003 | +0.012 | 8.7e-04 |

### Topology-specificity — FALSIFIED (2026-08-10). The driver is PROXIMITY, not the true topology.
The distance-preserving control (`preregistration_distance_control.md`, notebook `colab_distance_control.ipynb`)
substitutes each true edge with the nearest-distance NON-upstream basin: in-degree preserved exactly,
0% edge overlap, edge length 101 km vs true 92 km (vs 511 km for the random rewire). Pre-registered
prediction: if the gain needs the true upstream topology, the distance control falls to the random
level. **It did not — distance control +0.049 ≈ forward +0.046 at every seed** (+0.046/+0.060/+0.041 vs
forward's +0.046/+0.056/+0.027). Destroying the true topology while holding proximity fixed removes
none of the gain.

**What this means.** The earlier "topology-specificity" reading (forward ≫ random, +0.034) was
confounded: the random rewire also destroyed proximity (511 km edges), and that — not the wrong
topology — is why it failed. The gain comes from the dynamic discharge of spatially NEARBY basins
(~90–100 km), whether or not they are truly upstream. This is a spatial / regional-correlation effect,
not physical routing on the true network. The claim "the gain requires the real river network" does
NOT hold; the honest claim is "the gain requires spatially nearby basins' flow."

**What still holds.** (a) true / distance-control (~90–100 km) ≫ random (511 km): proximity matters,
distant basins are uninformative. (b) forward > reversed (+0.046 vs +0.031) survives as a mild
within-neighbour directional preference, but it is secondary and does NOT rescue topology-specificity,
since the distance control (neither forward nor reversed) matches forward.

**Caveat.** The heuristic graph over-connects and may also MISS real upstream basins, so a distance-
control "non-parent" could occasionally be a genuine, uncaptured upstream basin. This does not rescue
the original claim; it further shows the *heuristic* topology is not what drives the gain.

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

## Net for the paper (revised 2026-08-10 after the distance control)

- **Proximity, NOT true topology, drives the gain.** Distance control (nearby non-parents) +0.049 ≈
  forward +0.046 ≫ random +0.012. The gain requires spatially nearby basins' dynamic flow, not the
  specific upstream connectivity. The earlier "topology-specificity" headline is FALSIFIED and must
  be retired; "real river network / physical routing" language must go with it.
- **Proximity matters (this survives):** nearby (~90–100 km) ≫ distant (511 km random). Distant
  basins' flow is uninformative; the effect is spatial/regional correlation.
- **Directionality: mild within-neighbour preference, secondary.** forward +0.046 > reversed +0.031,
  but distance control (neither) matches forward, so direction does not carry the story.
- **Graph-robustness (unaffected): k=2 ≈ full-graph.** Not a heuristic-edge-density artifact — but
  note this is now "nearby-basin flow is robust to pruning," not "true topology is robust."

The static-null (2×2), the gain-is-real (null control), the deployable two-stage, and depth-vanishes-
at-headwaters results are UNAFFECTED. What changes is the mechanism interpretation: spatial proximity,
not the true river topology. Paper reframe required — see JOURNAL 2026-08-10.
