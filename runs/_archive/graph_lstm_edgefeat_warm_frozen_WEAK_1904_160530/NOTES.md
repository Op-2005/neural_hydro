# (archived) Graph-LSTM warm-started from WEAK baseline, edge features ON, LSTM frozen

**Status.** Archived (do not cite as primary). See `../README.md`.

**Model.** DirectedGraph-LSTM, mean aggregation, edge features ON, basin
encoding ON. Warm-started from `runs/03_lstm_23basin_baseline` (the
**weak** 23-basin baseline). LSTM cell + head **frozen**; only message-
passing parameters trainable. 15 epochs.

**Result.** Final median NSE 0.493; mean 0.348. Best epoch 9.

**Why archived.** Same root cause as the sibling archived run: warm-start
resolved the wrong baseline due to a `find_strong_baseline()` glob bug,
since fixed in `experiments/training/train_graph_lstm.py:75-91`.

**Where its evidence ended up.** This is the WEAK-baseline + frozen +
edges variant. Compared against the canonical STRONG-baseline + frozen +
edges (run 07, +0.013), it shows **+0.086 vs the WEAK baseline (0.407)**
— Finding #2's clearest evidence in `INSIGHTS.md` that the graph and
basin ID encoding carry overlapping information. When encoding is absent,
the frozen-graph contribution is large (+0.086); when encoding is
present, it shrinks to +0.013.
