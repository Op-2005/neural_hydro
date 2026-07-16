# Graph-Robustness Chain — is the routing signal an over-connectivity artifact?

ZERO training. upstream_q rebuilt on alternative graphs from observed discharge; signal strength scored via no-ML lstsq routing baseline (R1: Qhat=a·upstream_q+b, fit TRAIN 1990-99, scored TEST 2005-08, median over connected basins). Pre-reg: `preregistration_graph_robustness_chain.md`.

**Full graph** (624 edges, in-degree mean 4.16 / max 15): R1 median test-NSE = **+0.3250** (n=150 connected basins).

## Step A — pruned-graph robustness (over-connectivity test)

| pruning | rule | edges kept | R1 median NSE | % of full |
|---|---|---|---|---|
| in-degree ≤ 1 | nearest | 150 | +0.3193 | 98% |
| in-degree ≤ 1 | smallest-ratio | 150 | +0.2630 | 81% |
| in-degree ≤ 2 | nearest | 266 | +0.3263 | 100% |
| in-degree ≤ 2 | smallest-ratio | 266 | +0.2935 | 90% |
| in-degree ≤ 3 | nearest | 359 | +0.3263 | 100% |
| in-degree ≤ 3 | smallest-ratio | 359 | +0.3145 | 97% |

**k=2 (nearest) retains 100% of full-graph R1 NSE.** PASS (≥70%) — routing signal is NOT an over-connectivity artifact.

## Step B — depth-structure stability under k=2 pruning

- basins retaining depth within ±1: **95%** (173/183)
- max depth: full 5 → pruned 4 (Δ=-1)
- DAG preserved: True

**PASS** — depth hierarchy survives pruning.

## Step C — random edge-dropout sensitivity

Random dropout of a fraction of edges, 5 fixed-seed draws each; R1 median NSE.

| dropout | R1 NSE mean ± std (5 draws) | % of full | max-min spread |
|---|---|---|---|
| 20% | +0.3244 ± 0.0023 | 100% | 0.0061 |
| 40% | +0.3213 ± 0.0061 | 99% | 0.0143 |

**20% dropout: PASS** — signal degrades gracefully; not dependent on specific edges.
