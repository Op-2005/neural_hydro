# Pre-registration — Multi-Seed Confirmation of the Upstream-Signal Headline

**Pre-registered 2026-06-27, before observing seeds 13 & 17.**
**Context.** Single-seed (11) results are strong but directional: oracle +0.037/+0.050,
realizable +0.027 (72% of ceiling), null-control −0.002. Before any paper claim, the
headline contrasts need 3-seed confirmation with the variance reported.

## Conditions (3 seeds: 11, 13, 17; stock cudalstm, component0, 30 epochs)

| Cond | What | seed-11 median NSE |
|---|---|---|
| L | baseline | 0.653 |
| L+upQ | upstream OBSERVED Q (oracle, lag1) | 0.703 |
| L+upQ_pred | upstream PREDICTED Q (realizable) | 0.683 |
| L+upQshuf | shuffled-Q null control | 0.658 |

(Seed 11 already done; this adds seeds 13 and 17.)

## Success / falsification (on the realizable headline, L+upQ_pred − L)

- **Success (publishable):** cross-seed mean Δ ≥ **+0.015** with all 3 seeds positive
  (or bootstrap 95% CI excluding 0).
- **Falsification:** mean Δ ≤ +0.005 or seeds disagree in sign → the seed-11 result was
  seed-fragile; reframe before claiming.

## Pre-committed reporting
- Mean ± std across seeds for each condition (the tracked invariant).
- Realizable gain as % of oracle ceiling, per seed and pooled.
- Null control must stay ≈ 0 across seeds (else the gain is suspect).

## Compute
4 conditions × 2 new seeds = 8 cudalstm runs + 2 full-span evals (for predicted-Q per seed).
~30–40 min on Colab T4. (Needs full CAMELS dataset → Colab.)

## What we will NOT do
- Will not drop a seed that comes out unfavorable.
- Will not change the metric or the success bar after seeing seed 13/17.
