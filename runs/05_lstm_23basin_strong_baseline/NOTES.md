# Run 05 — Strong LSTM baseline (Kratzert-style)

**Model.** CudaLSTM + basin ID one-hot encoding. 23 HUC-12 Texas basins.
30 epochs.

**Config.** `experiments/configs/lstm_study_network_strong.yaml`.

**Result.** Median test NSE **0.423**. Mean NSE ≈ 0.43. See
`test/model_epoch030/test_metrics.csv` for per-basin numbers.

**Why it matters.** **The strong baseline every graph-variant comparison is
measured against.** Matches Kratzert 2019's multi-basin LSTM design; the
basin-encoding layer gives the LSTM explicit per-basin identity so it can
learn arbitrary per-basin behavior beyond what 5 static attributes encode.

**Where it fits.** Reference line for the +0.078 headline (vs run 06), the
+0.013 pure-graph isolation (vs run 07), and every ablation in runs 08–11.
