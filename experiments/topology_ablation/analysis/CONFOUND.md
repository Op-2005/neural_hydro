# Depth-Gradient Confound Check — routing (n_upstream) vs size (area)

Per-basin realizable Δ pooled over seeds [11, 13, 17] (n=549 basin×seed). Pre-reg: `preregistration_confound_check.md`.

## T1 — gain vs n_upstream (the routing variable)

| n_upstream | n | median Δ |
|---|---|---|
| 0 | 99 | +0.0018 |
| 1-2 | 171 | +0.0267 |
| 3-5 | 162 | +0.0252 |
| 6+ | 117 | +0.0265 |

Monotonic increase with n_upstream: False. headwaters(0) median +0.0018.

## T2 — gain vs area (the confound)

| area tercile (cuts 156/362 km²) | n | median Δ |
|---|---|---|
| small | 183 | +0.0206 |
| mid | 183 | +0.0212 |
| large | 183 | +0.0250 |

Area-tercile spread +0.0043.

## T3 — depth gradient within area terciles (the partial control)

| area tercile | depth0 median Δ | depth≥2 median Δ | diff |
|---|---|---|---|
| small | +0.0072 | +0.0399 | +0.0328 |
| mid | -0.0068 | +0.0339 | +0.0407 |
| large | -0.0038 | +0.0315 | +0.0353 |

**depth≥2 > depth0 by ≥+0.01 in 3/3 area terciles.** PASS — routing survives area control

## T4 — Spearman corr of Δ with each variable

- corr(Δ, n_upstream) = **+0.084**
- corr(Δ, depth)      = +0.158
- corr(Δ, area)       = +0.015

**n_upstream is the stronger predictor** (|0.084| vs |0.015|).