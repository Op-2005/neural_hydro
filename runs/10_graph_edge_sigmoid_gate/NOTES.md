# Run 10 — Independent sigmoid gate per edge

**Model.** DirectedGraph-LSTM with an independent sigmoid gate per edge.
Unlike softmax-attention, a single-parent edge can have gate < 1, so the
model can (in principle) down-weight an unreliable parent even if it's the
only one. Initialized with a +2 bias so gates start ≈ 0.88 (near-mean-agg at
initialization). Warm-started from run 05.

**Script.** `experiments/training/train_graph_lstm.py` with `USE_SIGMOID_GATE = True`.

**Result.** Median test NSE 0.496. Within ±0.005 of run 06.

**Why it matters.** The design was specifically meant to address the
softmax-attention limitation (run 09). Empirically, **the model did not
learn to differentiate good vs. bad parents** — all gates converged to
≈ 0.70. The gates are used for temporal modulation within a window, not for
static parent selection.

**Root cause.** The model has no direct supervision signal for "was this
message useful?" — it only sees output loss, and credit assignment through
any aggregator doesn't uniquely identify bad messages on a deep tree.

**Associated outputs.** `experiments/analysis_outputs/sigmoid_gates.csv`.

**Where it fits.** Part of the aggregation-saturation finding. Supports the
conclusion that mean aggregation + `W_out` residual is already flexible
enough to represent any selective weighting the more complex variants offer.
