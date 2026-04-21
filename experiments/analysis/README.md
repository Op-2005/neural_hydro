# Analysis Scripts

Post-hoc scripts that run against *already-trained* runs and produce the
CSVs/figures in `../analysis_outputs/`. None of these train models.

| File | What it does | Produces (in `../analysis_outputs/`) | Which `INSIGHTS.md` finding it supports |
|---|---|---|---|
| `compare_results.py` | Per-basin and depth-stratified NSE tables across one baseline and one-or-more graph-variant runs. Primary tool for the "graph vs baseline" comparisons. | Console output (copy into docs / NOTES.md) | Finding #1 (headline +0.078), Finding #4 (aggregation variants tie) |
| `analyze_results.py` | Hydrograph plots (observed / baseline / graph) for selected basins, learned-weight inspection (`W_msg_edge`, `W_out` statistics), per-basin ΔNSE vs static attributes. | `hydrograph_*.png`, `learned_weights_*.{png,json}`, `delta_vs_properties.png`, `per_basin_analysis.csv`, `per_basin_edge_warm.csv` | Finding #3 (graph helps well-predicted basins), general visualizations |
| `ensemble_analysis.py` | Error correlation matrix across graph-variant predictions; tests whether ensembling helps. | `ensemble_analysis.csv`, `attention_weights.csv`, `sigmoid_gates.csv` | Finding #4 (error correlations 0.994–0.999; ensemble gains only +0.0005 NSE) |
| `investigate_correlation.py` | Correlations between per-basin ΔNSE and baseline NSE / upstream parent NSE / static attributes. Feeds the r=+0.82 claim. | Console output (copy into docs) | Finding #3 (r = +0.82 with baseline NSE; parents' NSE adds +0.11 R²) |

## Running these

All are run from the repo root.

```bash
# Compare a strong baseline against multiple graph variants
python experiments/analysis/compare_results.py \
    --baseline runs/05_lstm_23basin_strong_baseline/test/model_epoch030/test_metrics.csv \
    --baseline-label "Strong LSTM" \
    --graph runs/06_graph_edge_warm_full/test_metrics.csv:Headline \
            runs/07_graph_edge_frozen/test_metrics.csv:Frozen \
            runs/11_graph_edge_pruned_edges/test_metrics.csv:Pruned

# Hydrographs + per-basin analysis figures
python experiments/analysis/analyze_results.py

# Ensemble correlations
python experiments/analysis/ensemble_analysis.py

# r=+0.82 correlation finding
python experiments/analysis/investigate_correlation.py
```

## How to extend these for the scale-up

For the Idea-1 Component-0 experiment, `compare_results.py` already works —
point it at the new baseline's `test_metrics.csv` and the graph-variant
`test_metrics.csv`. `investigate_correlation.py` uses
`../analysis_outputs/per_basin_analysis.csv` as its input, so we'll need to
produce a Component-0 version of that file (straightforward — just point
`analyze_results.py` at the new runs).
