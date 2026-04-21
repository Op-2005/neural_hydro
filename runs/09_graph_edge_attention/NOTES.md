# Run 09 — Softmax attention over parents

**Model.** DirectedGraph-LSTM with per-child softmax attention over parent
messages (replaces mean aggregation). Attention scorer is a small 2-layer
MLP over `[h_u, h_v, e_uv]`. Warm-started from run 05.

**Script.** `experiments/training/train_graph_lstm.py` with `USE_ATTENTION = True`.

**Result.** Median test NSE 0.495 — within 0.01 of mean-aggregation run 06.

**Why it matters.** Attention is the most natural "learn which parent
matters" mechanism. On this network, it does not outperform mean aggregation.
**Structural reason**: softmax over a single parent forces weight = 1 by
construction, so single-parent edges cannot be down-weighted. This is a
known limitation for tree-structured graphs with many headwaters.

**Associated outputs.** `experiments/analysis_outputs/attention_weights.csv`
contains the learned per-edge attention weights.

**Where it fits.** Part of the "aggregation variants saturate" finding
(runs 08–10 all within ±0.01 of run 06, error correlations 0.994–0.999).
