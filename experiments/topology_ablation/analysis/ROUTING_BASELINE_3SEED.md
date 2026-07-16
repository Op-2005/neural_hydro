# Part 1 — 3-Seed No-ML Routing Baseline

ZERO training. R1 (a·upQ+b) and R2 (a·upQ+c·L_sim+b) fit on TRAIN 1990-99, scored TEST 2005-08, per seed. Uses the fullspan L-sim available for all 3 seeds. Pre-reg: `preregistration_baseline_completion_and_k2.md` (Part 1).

## Median test NSE, mean ± std across seeds [11,13,17] (connected basins)

| predictor | mean ± std | per-seed (11/13/17) | note |
|---|---|---|---|
| R1 — pure routing | +0.3241 ± 0.0000 | +0.324/+0.324/+0.324 | no ML, upstream only |
| R2 — routing + local | +0.6640 ± 0.0080 | +0.675/+0.660/+0.657 | no ML, + L_sim |
| L (LSTM baseline) | +0.6548 ± 0.0059 | +0.654/+0.648/+0.662 | ML, no upstream |
| L+upQ_pred (realizable) | +0.6833 ± 0.0078 | +0.686/+0.691/+0.673 | ML + predicted upstream |
| L+upQ (oracle) | +0.7061 ± 0.0086 | +0.717/+0.705/+0.696 | ML + observed upstream |

## Pre-registered verdict

- realizable & oracle LSTM beat R1 (pure routing) at ALL seeds: **True** (per-seed: [True, True, True])

- realizable (+0.6833) vs strong R2 baseline (+0.6640): margin **+0.0192** (multi-seed; prior single-seed was +0.010)

**PASS — the LSTM-beats-naive-routing conclusion holds across all 3 seeds; not seed-fragile.**
