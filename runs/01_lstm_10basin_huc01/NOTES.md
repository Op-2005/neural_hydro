# Run 01 — LSTM on 10 HUC-01 basins (historical)

**Model.** CudaLSTM. 10 HUC-01 headwater basins. 5 epochs.

**Result.** Median test NSE ≈ 0.73.

**Why it matters.** First successful end-to-end LSTM training in this fork.
Confirmed the NH pipeline and data loaders worked on our installation.

**Status.** Historical. HUC-01 basins have no topology (Phase 0 verified the
heuristic inference returns zero edges), so we moved off this basin set.
Superseded by runs 03+ on the 23-basin Texas network.
