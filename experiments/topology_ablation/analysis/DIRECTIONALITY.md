# Directionality & Topology-Specificity Controls (the Kirschstein mirror test)

Seed 11, observed discharge, lag 1, stock cudalstm; only the edge set defining `upstream_q` changes. Δ paired vs L on the forward-connected basins (n=150). Pre-reg: `preregistration_directionality_controls.md`.

## Δ vs L by edge set

| edge set | aggregates | median Δ NSE | frac>0 | Wilcoxon p vs L |
|---|---|---|---|---|
| **forward** | true upstream parents | +0.0462 | 71% | 7.6e-06 |
| **reversed** | downstream children | +0.0263 | 64% | 1.3e-06 |
| **random** | random basins (in-degree preserved) | +0.0139 | 56% | 1.0e-01 |

## Pre-registered verdict

- **directional gap** (forward − reversed median Δ) = **+0.0199**  (≥ +0.015: True)
- **topology gap** (forward − random median Δ) = **+0.0323**  (≥ +0.015: True)

**PASS on both pre-registered criteria.**

## The ordering (the real finding)

forward `+0.046` > reversed `+0.026` > random `+0.014` > 0 — a monotone gradient of directional/topological correctness.

| paired contrast | median Δ | Wilcoxon p | reading |
|---|---|---|---|
| forward > reversed | +0.0079 | 1.9e-01 | **directionality: median favors forward but NOT significant per-basin** |
| forward > random | +0.0406 | 3.2e-04 | topology-specificity: strong, significant |
| reversed > random | +0.0262 | 1.6e-04 | even wrong-direction real edges beat random: significant |

## Honest interpretation

- **Topology specificity is strong and significant.** Random rewiring (same in-degree, wrong neighbors) nearly kills the gain (random Δ +0.014, not significant vs L at p=0.10; forward−random +0.041, p=3e-4). The signal lives in the **real river structure**, not any regional flow aggregate. This is the clean win.
- **Directionality is present but partial.** Reversed edges (downstream flow as fake upstream) retain ~57% of the forward gain (+0.026). Forward beats reversed on the median (+0.020, passing the pre-reg) but the *paired per-basin* difference (+0.008) is NOT significant (p=0.19). As the pre-reg anticipated, downstream flow is weather-correlated with the target, so reversal does not zero the signal — but here the residual is larger than a fully-directional mechanism would predict.
- **What this means for the routing claim (honest scope):** the gain is unambiguously **topology-specific** (real edges >> random). It is **directionally-preferential** (forward > reversed in the median, and both >> random) but **not strictly directional** at the per-basin significance level. The right framing is *the model exploits the real hydrological network, with a preference for the physically-correct upstream direction* — not *the gain requires correct direction*. Overclaiming strict directionality would be unsupported by the paired test.

## Positioning vs Kirschstein (mirror, appropriately scoped)

Kirschstein's GNNs were topology-insensitive (any/no adjacency ≈ same). Our feature is sharply **topology-sensitive** (real edges >> random rewire, p=3e-4) — the property their GNNs lacked. On direction specifically, our advantage is a median preference rather than a significant per-basin effect; we report that honestly rather than claiming the stronger result. **Single seed (11); a 3-seed replication would tighten the directional test.**
