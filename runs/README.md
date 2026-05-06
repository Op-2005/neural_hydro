# Experiment Runs — Master Index

Numbered chronologically by training order. For the research context that links
these runs together, see `../CURRENT_STATE.md`. For the current active
direction see `../idea1.md`.

## Runs

| # | Folder | Setting | What it showed |
|---|---|---|---|
| 01 | `01_lstm_10basin_huc01/` | CudaLSTM, 10 HUC-01 basins, 5 epochs | Historical: first successful LSTM run; median NSE 0.730. Demonstrated pipeline works. |
| 02 | `02_lstm_10basin_huc01_best/` | Same as 01, repeated | Historical: reproduced the 01 result. Set aside once we moved off HUC-01. |
| 03 | `03_lstm_23basin_baseline/` | CudaLSTM, 23-basin Texas network, 30 epochs, **no** basin encoding | Weak baseline, median NSE **0.407**. Used as control for "graph substitutes for basin encoding" finding. |
| 04 | `04_graph_lstm_23basin/` | Graph-LSTM v1 (early architecture), 10 epochs | Historical undertrained run (NSE 0.329). Superseded by run 06. |
| 05 | `05_lstm_23basin_strong_baseline/` | **Strong** CudaLSTM + basin ID encoding, 30 epochs | **Strong baseline**, median NSE **0.423**. Reference for every graph-variant comparison. Kratzert-style. |
| 06 | `06_graph_edge_warm_full/` | Graph-LSTM + edges, warm-started from run 05, full finetune | **Headline result**: median NSE **0.501** (+0.078). Most-cited number in the pilot. |
| 07 | `07_graph_edge_frozen/` | Graph-LSTM + edges, LSTM+head **frozen**; only message passing trains | Isolates pure graph contribution: median NSE 0.436 (+**0.013**). Showed most of +0.078 is LSTM weight drift, not message passing. |
| 08 | `08_graph_edge_diff_jiang/` | Graph-LSTM with Jiang 2025 direction term `h_u − h_v` | NSE 0.492; the ICML 2025 directional fix does not help here — our aggregation already encodes direction via predecessors. |
| 09 | `09_graph_edge_attention/` | Softmax attention over parents | NSE 0.495. Attention ties with mean; softmax cannot down-weight single-parent edges. |
| 10 | `10_graph_edge_sigmoid_gate/` | Independent per-edge sigmoid gate | NSE 0.496. Gates empirically did not learn to differentiate good vs bad parents (all ~0.70). |
| 11 | `11_graph_edge_pruned_edges/` | Graph-LSTM + mean aggregation, 26 of 34 edges (bad-parent edges dropped) | NSE **0.504**. The "bad-parent poisoning" hypothesis was WRONG: pruning did not rescue the affected basins — LSTM drift is the real mechanism. |
| 12 | `12_lstm_ungauged_baseline/` | CudaLSTM, 20 train basins, 3 held-out, **no** basin encoding | PUB baseline; held-out median NSE ≈ 0.23. |
| 13 | `13_graph_ungauged/` | Graph-LSTM warm-started from run 12; 3 held-out basins receive messages from trained parents at inference | **Nuanced**: 08158700 +0.043, 08189500 +0.107, 08164300 −0.575 (middle-node chain contamination). 2 of 3 improved; one catastrophic failure mode identified. |
| **14** | `14_lstm_component0_baseline_seed42/` | NH cudalstm + basin encoding, **183 basins (Component 0)**, 30 epochs, seed 42 | **Condition A scaled.** Median test NSE **0.648** (range [-6.5, 0.85]). First Component-0 baseline; bar that B and C must beat. |
| **15** | `15_graph_c0_topology_features_seed42/` | DirectedGraph-LSTM, no edges, +5 topology static features, 183 basins, seed 42, no warm-start | **Condition B scaled.** Median test NSE **0.591**. Per-basin Δ vs A: median **−0.050**. Topology-as-features alone does not help at scale. |
| **16** | `16_graph_c0_warm_seed42/` | DirectedGraph-LSTM, full edges (624) + message passing, 183 basins, seed 42, no warm-start | **Condition C scaled.** Median test NSE **0.578**. Per-basin Δ vs A: median **−0.077** (113/183 worse, 15 better). vs B: median Δ **−0.021** (much tighter distribution; std 0.082). **Pilot's +0.078 NSE does NOT replicate at scale, single seed.** |

## What the set of runs establishes

### 183-basin Component-0 setting (single-seed, seed=42)

| Setup | Median NSE | vs Condition A | per-basin Δ stats |
|---|---|---|---|
| **A** Baseline LSTM (run 14) | **0.648** | — | — |
| **B** Topology-as-features (run 15) | 0.591 | **−0.050** | 92/183 basins worse by ≥0.05; 20 better; std 0.487 |
| **C** Full graph-LSTM (run 16) | 0.578 | **−0.077** | 113/183 basins worse by ≥0.05; 15 better; std 0.526 |

**C − B median Δ = −0.021** (std 0.082 — much tighter than C−A and B−A,
suggesting most of the gap to A is shared between B and C, not unique to
message passing).

Depth-stratified plot: A wins at every depth except depth 4 (n=2, noise).
Gap (A − C) ~0.05 across depths 0–3 — global deficit, not depth-dependent.

Single seed; multi-seed verification is the load-bearing follow-up (5 seeds
via `'full'` mode of `notebooks/colab_publication_run.ipynb`). 23-basin
pilot results below remain a separate body of evidence at the smaller scale.

### Full 23-basin setting (pilot, single-seed)

| Setup | Median NSE | vs strong baseline |
|---|---|---|
| Strong baseline (run 05) | 0.423 | — |
| Graph + edges, full finetune (06) | 0.501 | **+0.078** |
| Graph + edges, pruned 26 edges (11) | 0.504 | +0.081 |
| Graph + edges, frozen LSTM (07) | 0.436 | +0.013 (pure graph) |
| Attention (09) | 0.495 | +0.072 |
| Sigmoid gate (10) | 0.496 | +0.073 |
| Jiang diff (08) | 0.492 | +0.069 |
| Weak baseline (03) | 0.407 | − |

The +0.078 full-finetune gain decomposes into ~+0.013 "pure graph message" and
~+0.065 "LSTM weight drift during joint training." Aggregation variants all
converge (error correlation 0.994–0.999).

## Run contents

```
config.yml / run_config.json   Experiment config
model_best.pt                    Best-by-test-NSE checkpoint (graph runs)
model_epoch*.pt                  Final checkpoint (intermediate deleted)
test_metrics.csv                 Per-basin test NSE
output.log                       Training log (NH runs)
train_data/                      Scaler + basin-encoding dicts
```

## How to evaluate

```bash
# NH runs (01–03, 05, 12)
/Applications/anaconda3/envs/nh/bin/python neuralhydrology/nh_run.py evaluate \
    --run-dir runs/05_lstm_23basin_strong_baseline --epoch 30

# Compare any baseline vs graph variants
/Applications/anaconda3/envs/nh/bin/python experiments/analysis/compare_results.py \
    --baseline runs/05_lstm_23basin_strong_baseline/test/model_epoch030/test_metrics.csv \
    --baseline-label "Strong LSTM" \
    --graph runs/06_graph_edge_warm_full/test_metrics.csv:Edge+Warm \
            runs/11_graph_edge_pruned_edges/test_metrics.csv:Pruned
```

## Archive

`_archive/` contains early runs trained against the wrong baseline (glob-pattern
bug in `find_strong_baseline()`, fixed since). Kept for historical completeness
only — do not cite these numbers.
