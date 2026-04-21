# Run 08 — Graph-LSTM with Jiang et al. direction-gradient term

**Model.** DirectedGraph-LSTM with the Jiang et al. (ICML 2025) direction-
gradient term `h_u − h_v` concatenated into the message input. Mean
aggregation. Edge features on. Warm-started from run 05.

**Script.** `experiments/training/train_graph_lstm.py` with `USE_DIFF_TERM = True`.

**Result.** Median test NSE 0.492. **−0.009 vs run 06's headline.**

**Why it matters.** Jiang's ICML 2025 paper proposes a directionality fix for
GNNs operating on river networks by including `h_u − h_v` in the message
(acting as a discrete gradient operator). **Their fix did not help us.** Our
aggregation already encodes direction via parent predecessors; adding the
explicit gradient is redundant and slightly harmful (extra parameters, same
information).

**Where it fits.** Ablation row in `runs/README.md`. Demonstrates that a
recently-published "fix" to the directional-insensitivity problem is not
universally applicable — the specific mechanism depends on the aggregation
structure.
