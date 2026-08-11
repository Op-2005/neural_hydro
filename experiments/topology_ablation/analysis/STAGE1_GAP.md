# Stage-1 Train/Test Asymmetry in the Realizable Model

Zero training. Re-analysis of the stored full-span (1990-2008) stage-1 baseline
predictions used to build the realizable upstream-flow input.

**Why this matters.** The realizable feature is built from stage-1 predictions over the
full record, but stage 1 was trained on 1990-1999. Its predictions are therefore
in-sample on the stage-2 training window and out-of-sample on the test window, so the
input degrades between training and deployment. This table measures that degradation.

## Median per-basin NSE of the stage-1 predictor

| seed | n basins | train window 1990-1999 (in-sample) | test window 2005-2008 (OOS) | gap |
|---|---|---|---|---|
| 11 | 183 | +0.898 | +0.653 | +0.231 |
| 13 | 183 | +0.906 | +0.651 | +0.244 |
| 17 | 183 | +0.904 | +0.656 | +0.223 |

**Cross-seed median: +0.904 in-sample vs +0.653 out-of-sample, a gap of +0.251 NSE.**
Pooled per-basin gap: median +0.236, IQR [+0.159, +0.357].

## Reading

The gap is the amount by which the upstream-flow input is better during stage-2
training than at test. Stage 2 therefore learns to weight a cleaner signal than it
receives at evaluation, which biases the realizable gain **downward**: the reported
+0.022 is a conservative estimate of what a matched-quality feature would deliver.
It does not inflate the result. Removing the asymmetry entirely would require a
held-out-fold stage 1 (train stage 1 on a subset, predict the rest), which is a
retraining experiment and is left as future work.
