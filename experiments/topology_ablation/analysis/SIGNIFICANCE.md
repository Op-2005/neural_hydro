# Step A — Statistical Significance of the Deployable Gain

Zero training. Paired per-basin NSE deltas, pooled seeds [11,13,17], Component 0. Wilcoxon signed-rank, one-sided (H1: realizable improves). Pre-reg: `preregistration_hardening_chain.md`.

## Pooled tests (n=549 basin×seed)

| Contrast | median Δ | frac basins + | Wilcoxon p (1-sided) | bootstrap 95% CI (median) |
|---|---|---|---|---|
| realizable − L (upQpred−L) | +0.0225 | 65.94% | 5.99e-19 | [+0.0175, +0.0281] |
| null − L (upQshuf−L) | +0.0040 | 52.64% | — | — |
| **realizable − null (upQpred−upQshuf)** | +0.0167 | 62.48% | **2.30e-12** | [+0.0106, +0.0216] |

## Pre-registered verdict

- realizable-vs-L p < 0.01: **True** (p=5.99e-19)
- realizable-vs-null p < 0.05: **True** (p=2.30e-12)

**PASS — deployable gain is significant AND separable from added-input capacity**

## Per-seed robustness (n=183 each; effect should hold within seeds, not only pooled)

| seed | median (real−L) | p (real−L) | median (real−null) | p (real−null) |
|---|---|---|---|---|
| 11 | +0.0265 | 4.60e-08 | +0.0274 | 1.79e-10 |
| 13 | +0.0258 | 2.79e-12 | +0.0151 | 2.14e-05 |
| 17 | +0.0131 | 1.44e-03 | +0.0078 | 6.13e-02 |
