# Depth-Gradient Confound Check — routing (n_upstream) vs size (area)

Per-basin realizable Δ pooled over seeds [13, 17] (n=366 basin×seed). Pre-reg: `preregistration_confound_check.md`.

## T1 — gain vs n_upstream (the routing variable)

| n_upstream | n | median Δ |
|---|---|---|
| 0 | 66 | -0.0027 |
| 1-2 | 114 | +0.0220 |
| 3-5 | 108 | +0.0252 |
| 6+ | 78 | +0.0249 |

Monotonic increase with n_upstream: False. headwaters(0) median -0.0027.

## T2 — gain vs area (the confound)

| area tercile (cuts 156/362 km²) | n | median Δ |
|---|---|---|
| small | 122 | +0.0207 |
| mid | 122 | +0.0187 |
| large | 122 | +0.0250 |

Area-tercile spread +0.0063.

## T3 — depth gradient within area terciles (the partial control)

| area tercile | depth0 median Δ | depth≥2 median Δ | diff |
|---|---|---|---|
| small | +0.0092 | +0.0300 | +0.0208 |
| mid | -0.0078 | +0.0276 | +0.0355 |
| large | -0.0206 | +0.0294 | +0.0500 |

**depth≥2 > depth0 by ≥+0.01 in 3/3 area terciles.** PASS — routing survives area control

## T4 — Spearman corr of Δ with each variable

- corr(Δ, n_upstream) = **+0.073**
- corr(Δ, depth)      = +0.134
- corr(Δ, area)       = -0.008

**n_upstream is the stronger predictor** (|0.073| vs |-0.008|).