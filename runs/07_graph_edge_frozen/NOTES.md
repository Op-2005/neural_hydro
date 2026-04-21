# Run 07 — Frozen-LSTM isolation (pure graph contribution)

**Model.** Same DirectedGraph-LSTM as run 06, but LSTM cell and head are
**frozen** at the run-05 baseline weights. Only the 8,448 message-passing
parameters (`W_msg_edge` + `W_out`) train.

**Script.** `experiments/training/train_graph_lstm.py` with `FREEZE_LSTM = True`.

**Result.** Median test NSE 0.436 — **+0.013 vs baseline**. Pure graph
contribution.

**Why it matters.** **The most important ablation in the pilot.** Run 06's
+0.078 headline is *not* all coming from the graph — only +0.013 is. The
other +0.065 is LSTM weight drift during joint training, acting as a
gradient-signal regularizer rather than as physical information transfer.

Sanity verification: the 7 headwater basins (no parents → message = 0 →
W_out·0 = 0 → no residual added to h) have |ΔNSE| < 1e-4 from baseline,
confirming the architecture is behaving as designed.

**Where it fits.** Central piece of the honest narrative in `INSIGHTS.md`.
Also the basis for Idea 2's hypothesis (temporal-lag preserves high-freq
content) — see `idea2/` for that direction.
