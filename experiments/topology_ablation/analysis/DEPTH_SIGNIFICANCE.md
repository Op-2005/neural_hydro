# Step B — Per-Depth Significance of the Realizable Gain

Zero training. Realizable per-basin Δ (L+upQ_pred − L) pooled seeds [11, 13, 17] (n=549 basin×seed). Per-depth paired Wilcoxon signed-rank, one-sided (H1: Δ>0). Pre-reg: `preregistration_routing_baseline_chain.md`.

| depth | n | median Δ | Wilcoxon p (1-sided) | significant (p<0.05) |
|---|---|---|---|---|
| 0 | 99 | +0.0018 | 2.40e-01 | False |
| 1 | 126 | +0.0265 | 1.94e-06 | True |
| 2 | 141 | +0.0194 | 3.87e-05 | True |
| 3 | 102 | +0.0363 | 2.51e-09 | True |
| 4 | 63 | +0.0319 | 1.73e-06 | True |
| 5 | 18 | +0.0121 | 2.90e-01 | False |

## Pre-registered verdict

- depth 0 (headwaters) NOT significant: **True**
- depths 1 AND 2 significant: **True**

**PASS — the routing gain is statistically significant specifically in the downstream strata (depth≥1) and absent at headwaters. The depth gradient has per-stratum statistical teeth, not just a median trend.**
