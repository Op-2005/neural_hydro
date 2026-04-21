# Run 12 — Ungauged-basin LSTM baseline (PUB setup)

**Model.** CudaLSTM. 20 training basins, 3 held-out basins (08158700,
08164300, 08189500). **No basin ID encoding** — held-out basins wouldn't
have a learnable embedding anyway.

**Config.** `experiments/configs/lstm_ungauged_train.yaml`.

**Result.** Held-out per-basin NSE:
- 08158700 (leaf with training parent): 0.059
- 08164300 (middle node, held-out parent): 0.360
- 08189500 (held-out parent + training parent): 0.233

Median 0.233. Training basins' median NSE ≈ 0.42.

**Why it matters.** PUB (Prediction in Ungauged Basins) baseline — the
entity-aware LSTM has to rely on static attributes alone to predict unseen
basins. Provides the reference for the graph-assisted PUB experiment
(run 13).

**Where it fits.** Baseline for run 13's ungauged graph evaluation. Feeds
into the ongoing question of whether the graph can substitute for basin
identity on truly unseen basins.
