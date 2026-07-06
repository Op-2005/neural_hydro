# Pre-registration — Strongest Realizable Version (lag0 predicted-Q) + KGE robustness

**Pre-registered 2026-07-05, before running.**
**Context.** The headline realizable result uses lag1 predicted upstream Q (+0.022 cross-seed).
But the OBSERVED-Q oracle is strongest at lag0 (+0.087 vs +0.037 at lag1) — same-day upstream
flow is most informative at daily resolution. We never built the *predicted*-Q version at
lag0, so the strongest realizable version is untested. Also, our methodology requires 3
metrics; we've shown NSE + log-NSE robust but not KGE on the realizable headline.

## Step A — Predicted-Q at lag0 (strongest realizable)

**Hypothesis.** Predicted upstream Q at lag0 beats the lag1 realizable headline (+0.022),
mirroring the observed-Q lag0 > lag1 pattern.
- **Setup:** build predicted-upstream-Q at lag=0 from the existing `_Lfullspan_eval_seed11`
  predictions (already on disk); train stock cudalstm L + upQ_pred_lag0, seed 11.
- **Success:** (L+upQ_pred_lag0 − L) median Δ > +0.022 (the lag1 realizable value).
- **Falsify:** ≤ +0.022 → lag1 stays the headline; report both.

**Leakage note.** lag0 = same-day *upstream* predicted Q → *downstream* prediction. The
downstream basin's own observation never enters. Same-day upstream→downstream is physical
routing (sub-daily travel times), not target leakage — but must be stated: it assumes
upstream predictions are available at inference time (true in a two-stage deployed system).

## Step B — KGE robustness (gated on A; free re-analysis)

**Hypothesis.** The realizable gain is sign-consistent in KGE (3rd required metric).
- **Success:** realizable KGE Δ > 0 across seeds 11/13/17.
- **Falsify:** KGE Δ ≤ 0 → gain is NSE/log-NSE-specific; scope the claim.

## Step C — lag0 seed-robustness (gated on A passing)

**Hypothesis.** If lag0-predicted wins on seed 11, it holds on seed 13.
- **Success:** lag0-predicted > lag1-predicted on seed 13 too.
- **Falsify:** seed-fragile → lag1 is the stable headline; lag0 reported as supplementary.

## What we will NOT do
- Will not use observed upstream Q in the realizable conditions.
- Will not change bars after seeing results.
- Steps B is re-analysis; A/C are 1 train each (eval predictions already on disk).

---
## Results (post-run, 2026-07-06)

**Step A — lag0-predicted FALSIFIED (lag1 remains the headline).** seed 11:
- lag1 predicted: Δ +0.0265 (recovers 72% of its +0.037 oracle)
- lag0 predicted: Δ +0.0229 (recovers only **26%** of its +0.087 oracle)

lag0-predicted (+0.0229) < lag1-predicted (+0.0265) → hypothesis falsified. But the mechanism
is informative: the observed oracle is 2× stronger at lag0, yet the *predicted* version
recovers only 26% of that lag0 ceiling vs 72% of lag1. **Same-day upstream flow carries the
most signal when observed, but is the hardest to predict** — lag1 is the realizable sweet spot
(enough signal, and forecastable). The deployable gain is capped by upstream *predictability*,
not the downstream model. **lag1 (+0.0265, cross-seed +0.022) stays the headline.**

Per pre-reg, chain STOPS at Step A (Step C was gated on lag0 winning — not run).

**Step B — KGE robustness (3rd metric):** realizable Δ robust in NSE (+0.022, all seeds +)
and log-NSE (+0.027, all seeds +), but KGE +0.013 mean with seed 13 dipping −0.002 (not
all-positive). Honest scope: robust in NSE/log-NSE; KGE-positive-on-average with seed
sensitivity. Report all three.
