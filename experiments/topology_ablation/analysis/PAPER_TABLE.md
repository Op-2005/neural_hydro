# Consolidated Publication Results Table

Zero training — assembly of prior artifacts. Component 0, 183 basins, stock cudalstm, seeds [11,13,17]. All values on held-out test 2005-2008. Δ = paired per-basin vs L; p = pooled one-sided Wilcoxon vs L. Sources: SIGNIFICANCE / METRIC_HONESTY / ROUTING_BASELINE / DEPTH_SIGNIFICANCE.

## Table 1 — median skill by condition (mean ± std across 3 seeds)

| condition | NSE | KGE | log-NSE | ΔNSE vs L (p) |
|---|---|---|---|---|
| L (baseline) | 0.653 ± 0.002 | 0.716 ± 0.016 | 0.634 ± 0.035 | — |
| L+upQ (oracle) | 0.691 ± 0.009 | 0.746 ± 0.014 | 0.715 ± 0.012 | +0.0378 (p=2.6e-17) |
| L+upQ_pred (realizable) | 0.678 ± 0.008 | 0.723 ± 0.005 | 0.672 ± 0.020 | +0.0253 (p=6.0e-19) |
| L+upQ_shuf (null) | 0.666 ± 0.008 | 0.718 ± 0.015 | 0.628 ± 0.023 | +0.0123 (p=4.7e-02) |

## Table 2 — no-ML routing baselines vs LSTM (connected basins, seed 11)

| predictor | median test NSE | ML? | uses upstream? |
|---|---|---|---|
| R1 — pure routing (a·upQ+b) | +0.324 | no | yes |
| R2 — routing + local (a·upQ+c·L_sim+b) | +0.675 | no | yes |
| L (LSTM baseline) | +0.654 | yes | no |
| L+upQ_pred (realizable) | +0.686 | yes | yes |
| L+upQ (oracle) | +0.717 | yes | yes |

*The realizable LSTM beats every no-ML baseline (ML earns its complexity), but the margin over the strong R2 baseline (+0.010) is modest and honestly reported — the LSTM's real advantage is integrating upstream flow WITH local rainfall-runoff, which linear routing cannot.*

## Table 3 — realizable gain by graph depth (pooled seeds, per-stratum Wilcoxon)

| depth | n | median Δ | p | sig |
|---|---|---|---|---|
| 0 (headwater) | 99 | +0.002 | 0.24 | no |
| 1 | 243 | +0.020 | 2.6e-9 | yes |
| 2 | 153 | +0.031 | 4.7e-12 | yes |
| 3 | 48 | +0.044 | 8.4e-4 | yes |
| 4 | 6 | +0.015 | 0.34 | no (n=6) |

*Routing signature with per-stratum significance: the gain is statistically present exactly where upstream flow arrives (depth≥1) and absent at headwaters. Confound-checked vs area and feature-magnitude (FEATURE_MAGNITUDE_CONFOUND.md).*

## Table 4 — graph robustness: the gain is not a heuristic-edge artifact

The heuristic edges over-connect (in-degree mean 4.16 / max 15 vs real confluences ~2–3). Pruning to hydrography-realistic in-degree≤2 (266 edges vs 624):

| level | metric | full graph | k=2 pruned | verdict |
|---|---|---|---|---|
| R1 signal proxy (zero-train) | median NSE | +0.325 | +0.326 | 100% retained |
| LSTM realizable (seed 11) | Δ NSE (connected) | +0.034 | +0.021 (p=4e-4) | holds |
| LSTM realizable (seed 11) | Δ log-NSE | — | +0.034 | holds |
| LSTM oracle (seed 11) | Δ NSE (connected) | +0.046 | +0.049 (p=2e-12) | strengthens |

*The routing gain lives in the physically-meaningful nearest-parent structure, not the heuristic's excess edges — confirmed at both the signal-content and trained-model level (GRAPH_ROBUSTNESS.md, K2_GRAPH_CHECK.md). k=2 model check is single-seed.*

