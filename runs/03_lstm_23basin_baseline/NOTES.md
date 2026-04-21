# Run 03 — Weak LSTM baseline on 23-basin Texas network

**Model.** CudaLSTM. 23 HUC-12 Texas basins. 30 epochs. **No basin ID
encoding** (just 5 static attributes: elev, area, slope, p_mean, pet_mean).

**Config.** `experiments/configs/lstm_study_network.yaml`.

**Result.** Median test NSE **0.407**.

**Why it matters.** Weak reference baseline. Used as the "no-entity-aware"
control for the "graph substitutes for basin encoding" finding: compared to
run 05 (strong baseline with encoding, 0.423), the encoding-less baseline is
worse by the same margin the graph adds — which is the empirical basis for
the substitution claim.

**Where it fits.** The weaker of two baselines. See `runs/README.md` for the
full comparison table.
