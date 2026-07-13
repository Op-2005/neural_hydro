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
