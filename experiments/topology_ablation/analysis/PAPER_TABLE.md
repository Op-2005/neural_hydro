# Consolidated Publication Results Table

Zero training — assembly of prior artifacts. Component 0, 183 basins, stock cudalstm, seeds [11,13,17]. All values on held-out test 2005-2008. ΔNSE = paired per-basin median (cross-seed mean of the per-seed medians); p = pooled Wilcoxon, one-sided for the directional oracle/realizable and two-sided for the null. Sources: SIGNIFICANCE / METRIC_HONESTY / ROUTING_BASELINE_3SEED / DEPTH_SIGNIFICANCE / MECHANISM_MULTISEED.

## Table 1 — median skill by condition (mean ± std across 3 seeds)

| condition | NSE | KGE | log-NSE | ΔNSE vs L (p) |
|---|---|---|---|---|
| L (baseline) | 0.653 ± 0.002 | 0.716 ± 0.016 | 0.634 ± 0.035 | — |
| L+upQ (oracle) | 0.691 ± 0.009 | 0.746 ± 0.014 | 0.715 ± 0.012 | +0.0352 (p=2.6e-17) |
| L+upQ_pred (realizable) | 0.678 ± 0.008 | 0.723 ± 0.005 | 0.672 ± 0.020 | +0.0218 (p=6.0e-19) |
| L+upQ_shuf (null) | 0.666 ± 0.008 | 0.718 ± 0.015 | 0.628 ± 0.023 | +0.0026 (p=9.4e-02) |

*Oracle log-NSE reflects only seed(s) [13, 17] locally: the other oracle `test_results.p` files are missing or truncated on this machine but exist on Drive (seed 11 lost in a drive merge; seed 13 truncated). The paper reports the 2-seed value (0.715 ± 0.012) computed when those files were intact; re-sync them to reproduce it. Realizable log-NSE (the load-bearing metric) is fully 3-seed and intact. ΔNSE is the paired per-basin median (cross-seed mean), not the difference of the median NSE columns, so it need not equal the column subtraction.*

## Table 2 — no-ML routing baselines vs LSTM (connected basins, mean ± std 3 seeds)

| predictor | median test NSE | ML? | uses upstream? |
|---|---|---|---|
| R1 — pure routing (a·upQ+b) | +0.324 ± 0.000 | no | yes |
| R2 — routing + local (a·upQ+c·L_sim+b) | +0.664 ± 0.008 | no | yes |
| L (LSTM baseline) | +0.655 ± 0.006 | yes | no |
| L+upQ_pred (realizable) | +0.683 ± 0.008 | yes | yes |
| L+upQ (oracle) | +0.706 ± 0.009 | yes | yes |

*The realizable LSTM beats every no-ML baseline at all 3 seeds (ML earns its complexity). Its margin over the strong R2 baseline is +0.019 (3-seed): the LSTM integrates upstream flow WITH local rainfall-runoff, which linear routing cannot. Source: ROUTING_BASELINE_3SEED.md.*

## Table 3 — realizable gain by graph depth (LONGEST-PATH graph_depth, pooled seeds, per-stratum Wilcoxon)

| depth | n | median Δ | p | sig |
|---|---|---|---|---|
| 0 (headwater) | 99 | +0.002 | 0.24 | no |
| 1 | 126 | +0.027 | 1.9e-6 | yes |
| 2 | 141 | +0.019 | 3.9e-5 | yes |
| 3 | 102 | +0.036 | 2.5e-9 | yes |
| 4 | 63 | +0.032 | 1.7e-6 | yes |
| 5 | 18 | +0.012 | 0.29 | no (n=18) |

*Longest-path graph_depth (matches paper Eq.1 and Table tab:depth; the earlier shortest-path stratification is superseded, see DEPTH_SIGNIFICANCE.md). The gain is significant exactly where upstream flow arrives (depth 1-4) and not at headwaters or the sparse depth-5 stratum. Confound-checked vs area and feature-magnitude (FEATURE_MAGNITUDE_CONFOUND.md).*

## Table 4 — graph robustness: the gain is not a heuristic-edge artifact

The heuristic edges over-connect (in-degree mean 4.16 / max 15 vs real confluences ~2–3). Pruning to hydrography-realistic in-degree≤2 (266 edges vs 624):

| level | metric | full graph | k=2 pruned | verdict |
|---|---|---|---|---|
| R1 signal proxy (zero-train) | median NSE | +0.325 | +0.326 | 100% retained |
| LSTM realizable (3-seed) | Δ NSE (connected) | +0.026 | +0.025 (p=1.3e-14) | holds |
| LSTM oracle (3-seed) | Δ NSE (connected) | +0.046 | +0.059 (p=2.8e-43) | strengthens |

*The routing gain lives in the physically-meaningful nearest-parent structure, not the heuristic's excess edges, confirmed at both the signal-content and trained-model level across 3 seeds (GRAPH_ROBUSTNESS.md, K2_GRAPH_CHECK.md, MECHANISM_MULTISEED.md).*

