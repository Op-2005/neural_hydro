# Step B — Per-Depth Significance of the Realizable Gain

Zero training. Realizable per-basin Δ (L+upQ_pred − L) pooled seeds [11, 13, 17] (n=549 basin×seed). Per-depth paired Wilcoxon signed-rank, one-sided (H1: Δ>0). Pre-reg: `preregistration_routing_baseline_chain.md`.

| depth | n | median Δ | Wilcoxon p (1-sided) | significant (p<0.05) |
|---|---|---|---|---|
| 0 | 99 | +0.0018 | 2.40e-01 | False |
| 1 | 243 | +0.0199 | 2.62e-09 | True |
| 2 | 153 | +0.0307 | 4.65e-12 | True |
| 3 | 48 | +0.0443 | 8.42e-04 | True |
| 4 | 6 | +0.0152 | 3.44e-01 | False |

## Pre-registered verdict

- depth 0 (headwaters) NOT significant: **True**
- depths 1 AND 2 significant: **True**

**PASS — the routing gain is statistically significant specifically in the downstream strata (depth≥1) and absent at headwaters. The depth gradient has per-stratum statistical teeth, not just a median trend.**

## CORRECTION 2026-08-10 — use LONGEST-PATH depth (graph_depth), matching Eq.1 and the paper
The table above used shortest-path depth from `component0_depth.csv` (max 4). The paper's Eq.1 and
Table tab:depth use LONGEST-path `graph_depth` (max 5). Recomputed from graph_depth + stored deltas,
pooled 3 seeds (reproduces the paper table exactly):

| depth | n | median Δ | p (1-sided) | sig |
|---|---|---|---|---|
| 0 | 99 | +0.0018 | 2.40e-01 | no |
| 1 | 126 | +0.0265 | 1.94e-06 | yes |
| 2 | 141 | +0.0194 | 3.87e-05 | yes |
| 3 | 102 | +0.0363 | 2.51e-09 | yes |
| 4 | 63 | +0.0319 | 1.73e-06 | yes |
| 5 | 18 | +0.0121 | 2.90e-01 | no (n=18) |
| connected (≥1) | 450 | +0.0264 | 5.13e-21 | yes |

This is the canonical stratification. `plot_depth_figure.py` and paper Fig/Table now use it.
`analyze_depth_significance.py` should be pointed at `graph_depth` to regenerate this directly.
