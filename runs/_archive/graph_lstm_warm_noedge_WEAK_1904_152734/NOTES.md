# (archived) Graph-LSTM warm-started from WEAK baseline, no edge features

**Status.** Archived (do not cite as primary). See `../README.md`.

**Model.** DirectedGraph-LSTM, mean aggregation, edge features OFF, basin
encoding ON (matched to weak-baseline config). Warm-started from
`runs/03_lstm_23basin_baseline` (the **weak** 23-basin baseline, no basin
encoding embeddings — the warm-start only carried LSTM core weights).
15 epochs, full finetune.

**Result.** Final median NSE 0.479; mean 0.344.

**Why archived.** Used the wrong baseline for warm-start due to a
`find_strong_baseline()` glob-pattern bug since fixed in
`experiments/training/train_graph_lstm.py:75-91`. The intended-canonical
runs warm-start from `runs/05_lstm_23basin_strong_baseline`.

**Where its evidence ended up.** Part of the `INSIGHTS.md` Finding #2
"graph substitutes for basin ID encoding" comparison — the WEAK-baseline
side of the contrast that established graph adds more (~+0.07) when basin
encoding is absent versus +0.013 when it is present.
