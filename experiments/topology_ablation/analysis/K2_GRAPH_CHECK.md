# k=2 Pruned-Graph LSTM Check — the definitive (model-level) graph result

Colab-trained runs on the **in-degree≤2 nearest-parent pruned graph** (266 edges vs 624; hydrography-realistic — real confluences join 2-3 tributaries), seeds [11, 13, 17]. This confirms at the LSTM level what the 2026-07-14 GRAPH_ROBUSTNESS chain showed for the R1 lstsq proxy. Pre-reg: `preregistration_baseline_completion_and_k2.md` (Part 3). *3-seed re-analysis (2026-08): the seeds 13/17 runs were already on disk; this is a zero-training re-run, no GPU.*

## Paired Δ vs L, connected basins (n=150), 3 seeds

| condition | graph | per-seed median Δ (11/13/17) | pooled median Δ | Wilcoxon p (1-sided) |
|---|---|---|---|---|
| oracle | full | +0.0462 / +0.0564 / +0.0266 | +0.0462 | 7.6e-23 |
| oracle | k=2 | +0.0494 / +0.0743 / +0.0480 | +0.0594 | 2.8e-43 |
| realizable | full | +0.0340 / +0.0298 / +0.0149 | +0.0264 | 5.1e-21 |
| realizable | k=2 | +0.0214 / +0.0332 / +0.0210 | +0.0252 | 1.3e-14 |

## Pre-registered verdict

- k=2 realizable pooled Δ (connected) = **+0.0252** vs full-graph realizable +0.0264 on the same basins.
- k=2 realizable log-NSE Δ = **+0.0281** (mean over 2 seeds with intact results.p)
- k=2 realizable positive at all three seeds and within ±0.010 of the full-graph realizable: **True**.

**PASS — the routing gain survives at the LSTM level on a hydrography-realistic graph, across three seeds.** The over-connectivity threat is closed at BOTH the signal-content (R1 proxy) AND the trained-model level. The heuristic's excess edges are not doing the work.

## Interpretation

- **Realizable holds:** pooled +0.025 NSE on the pruned graph, positive at all three seeds (p=1.3e-14), essentially equal to the full-graph realizable gain on the same connected basins. Predicted upstream Q remains deployable when the graph is pruned to real-confluence connectivity.
- **Oracle strengthens under pruning:** pooled k=2 oracle Δ (+0.059) exceeds the full-graph oracle on the same basins. Removing the excess (distant, weakly-connected) parents *sharpens* the observed upstream signal — consistent with the routing physics (nearest parents = shortest travel time = most-aligned flow), and with the 2026-07-14 finding that the R1 signal lives in the nearest parents.
- **Scope:** three seeds (11/13/17), single pruning rule (nearest, k=2). One seed-17 `results.p` is truncated, so the log-NSE mean is over the intact seeds; the NSE result is fully 3-seed from `test_metrics.csv`.

