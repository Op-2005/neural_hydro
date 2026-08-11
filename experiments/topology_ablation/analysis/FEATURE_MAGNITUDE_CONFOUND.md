# Step B — Routing vs Feature-Magnitude Confound

Zero training. Per-basin realizable Delta pooled seeds [11, 13, 17] (n=549 basin×seed). feature magnitude = per-basin mean |upstream_q| (lag-1 predicted feature). Pre-reg: `preregistration_hardening_chain.md`.

## T1 — feature magnitude vs depth (structural direction check)

Depth-0 = headwaters have fmag=0 by construction, so they cannot appear in any positive-fmag tercile. The relevant question is whether fmag *rises* with depth (which would make the depth gradient a magnitude artifact). It does NOT:

| depth | n (connected) | median fmag (mm/d) | median Δ |
|---|---|---|---|
| 1 | 126 | 1.847 | +0.0265 |
| 2 | 141 | 1.720 | +0.0194 |
| 3 | 102 | 1.768 | +0.0363 |
| 4 | 63 | 1.500 | +0.0319 |
| 5 | 18 | 1.324 | +0.0121 |

**corr(depth, fmag) among connected basins = -0.296** — feature magnitude DECREASES with depth, so the rising depth→Δ gradient runs AGAINST feature magnitude (the confound is directionally absent).

## T1b — deep vs shallow WITHIN each feature-magnitude tercile (connected only)

Compares depth-1 (shallowest connected) vs depth≥3 (deepest) inside each fmag tercile — holding feature magnitude roughly fixed.

| fmag tercile | depth1 median Δ | depth≥3 median Δ | diff | deeper-wins |
|---|---|---|---|---|
| low | +0.0335 | +0.0336 | +0.0001 | True |
| mid | +0.0185 | +0.0177 | -0.0008 | False |
| high | +0.0407 | +0.0483 | +0.0076 | True |

**deeper (depth≥3) > shallower (depth1) in 2/3 testable terciles.**

## T2 — partial Spearman correlations (the decisive control)

- raw corr(Δ, depth)     = +0.116 (p=6.3e-03)
- raw corr(Δ, fmag)      = +0.115 (p=6.9e-03)
- raw corr(Δ, n_upstream)= +0.084 (p=5.0e-02)
- raw corr(Δ, area)      = +0.015 (p=7.3e-01)

**partial corr(Δ, depth | area, fmag) = +0.121 (p=4.4e-03)**

reverse: partial corr(Δ, fmag | area, depth) = +0.089 (p=3.7e-02)

## Verdict

Depth predicts the realizable gain even after removing area AND feature-magnitude (partial corr +0.121, p=4.4e-03). **ROUTING survives** — the gradient is graph position, not feature scale.
