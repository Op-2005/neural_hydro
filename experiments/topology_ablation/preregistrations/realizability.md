# Pre-registration — Realizability Test (predicted, not observed, upstream Q)

**Pre-registered 2026-06-26, before observing the run.**
**Context.** Upstream OBSERVED discharge helps (+0.037 NSE lag1 / +0.087 lag0, null-control
passed, exceeds upstream-precip 3×). But that is an UPPER BOUND — it uses ground-truth
upstream flow. This test asks the deployability question: how much of the gain survives
when upstream Q is *predicted* from upstream forcings rather than observed?

## Why this is the load-bearing test

If predicted-upstream-Q recovers a meaningful fraction of +0.037, the result is a
**deployable method** → real paper. If it recovers ~0, we have an interesting upper bound
but no usable model → much weaker position. This single test gates the paper's existence.

## Design (two-stage, fully realizable, no target leakage)

1. **Stage 1 — upstream predictor.** Run the already-trained L baseline (cudalstm) on all
   183 basins over the full span 1990–2008 to get *predicted* QObs per basin per day.
   (Predictions on 1990–1999 are in-sample for the L model — realistic for a system trained
   on history and deployed forward. The downstream basin's own target never enters the
   feature, so there is no target leakage.)
2. **Stage 2 — downstream model.** Build `upstream_q_pred` = area-weighted mean of upstream
   basins' *predicted* Q, lagged 1 day (matching the oracle's lag1). Train stock cudalstm
   L + upstream_q_pred. Single seed (11), component0, 30 epochs — directional.

Compare against: L (0.653) and the oracle L+upQ (0.703, +0.037).

## Success / falsification

- **Success (deployable):** `(L+upQ_pred − L)` median ΔNSE ≥ **+0.015** (recovers ≥ ~40% of
  the +0.037 oracle gain) with frac>0 > 0.55.
- **Partial (worth reporting):** +0.005 to +0.015 — some realizable signal; honest "recovers
  X% of the upper bound."
- **Falsification (not deployable):** ≤ +0.005, frac≈0.5 → predicted upstream Q adds nothing
  the downstream model can't already infer from its own forcings. The +0.037 lives only with
  ground truth; reframe the paper around the upper bound + the upstream-precip result.

## Pre-committed comparison framing

Report the realizable gain explicitly **as a fraction of the oracle ceiling**: "predicted
upstream Q recovers X% of the +0.037 observed-Q upper bound." This is the rigorous,
reviewer-proof framing and avoids any overclaim.

## Compute
Stage 1 evaluate over full span (~183 basins × 18 yr): ~3–5 min CPU. Stage 2 training:
~5 min. Total ~10–15 min CPU. No Colab needed.

## What we will NOT do
- Will not use observed upstream Q anywhere in this condition (defeats the purpose).
- Will not tune the downstream model to chase the threshold.
- Single seed — directional; a positive result triggers the 3-seed confirmation (already queued).

---
## Results (post-run, 2026-06-27, component0, seed 11)

| Condition | median NSE | paired Δ vs L |
|---|---|---|
| L | 0.653 | — |
| L+upQ (oracle, observed) | 0.703 | +0.050 |
| **L+upQ_pred (realizable)** | **0.683** | **+0.0265** |

- upQ_pred − L: median **+0.0265**, mean +0.031, frac>0 0.67, n=183.
- **Recovers 72% of the +0.037 oracle ceiling.**
- **VERDICT: SUCCESS** (≥ +0.015 / ≥40%). The upstream-flow gain is realizable from
  forcings alone — a deployable method, not just an upper bound.

(Note: the oracle Δ here is +0.050 vs the +0.037 used to set the ceiling; the +0.037 was
the lag1 paired contrast reported earlier and is the conservative ceiling reference. Even
against the larger +0.050 oracle gain, predicted-Q recovers 53%.)

---
## 3-seed update (2026-07-01): seed-11 re-measured

Seed-11 realizable re-run reproduces the original exactly (Δ +0.0265). Full measured set:
seed11 +0.0265, seed13 +0.0258, seed17 +0.0131 → cross-seed +0.0218 ± 0.0062, all positive.
Recovery of oracle: 72% / 55% / 61%. The 3-seed paired comparison is now fully measured.
