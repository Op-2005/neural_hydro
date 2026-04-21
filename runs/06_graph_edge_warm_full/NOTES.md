# Run 06 — Headline Graph-LSTM with edges (+0.078 NSE)

**Model.** DirectedGraph-LSTM. 23 basins, 34 edges. Mean aggregation over
parents. Edge features `[log_dist, log_area_ratio, elev_drop]`. Warm-started
from run 05 with `W_out` zero-initialized (so pre-training NSE exactly
matches baseline). 15 epochs of joint finetuning; all parameters trainable.

**Script.** `experiments/training/train_graph_lstm.py` with default flags (edge
features + basin encoding + warm-start all on, frozen off).

**Result.** Median test NSE **0.501** — **+0.078 vs strong baseline (run 05)**.
The headline number of the pilot.

**Why it matters.** Reproducible, consistent across aggregation variants
(attention, sigmoid gate, Jiang diff all within ±0.01), and large enough to
notice on a small network. **But most of the gain is LSTM weight drift
during joint training, not message passing** — see run 07 for the isolated
contribution.

**Associated outputs.**
- Per-basin NSE: `test_metrics.csv` in this folder
- Hydrographs: `experiments/analysis_outputs/hydrograph_*.png`
- Weight inspection: `experiments/analysis_outputs/learned_weights_edge_warm.*`
- Depth-stratified comparison: via `compare_results.py`
