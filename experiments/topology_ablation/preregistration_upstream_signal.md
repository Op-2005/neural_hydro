# Pre-registration — Upstream-Signal Chain (Branch A1, post-oracle)

**Pre-registered 2026-06-21, before observing any of the runs below.**
**Context:** oracle passed — L 0.653 → L+upstream_q 0.703, paired median +0.037 NSE
(component0, seed 11, single seed, observed-discharge UPPER BOUND). Static topology
features were null. This chain stress-tests the +0.037 before investing in a realizable
learned model.

All runs: stock NH cudalstm, component0 (183 basins), seed 11, 30 epochs, one-hot ON,
upstream feature injected via `additional_feature_files`. Baseline L already trained
(median 0.653).

---

## Step A — Null control: shuffled-upstream-Q (run FIRST; gates the chain)

**Hypothesis.** The +0.037 oracle gain comes from real upstream-flow content, not from
adding any same-distribution extra input. A shuffled-in-time upstream_q column should give
≈0 gain.

**Design.** Build `upstream_q` as in the oracle, then permute its values across time within
each basin (preserves marginal distribution, destroys temporal alignment with the target).
Train L + upstream_q_shuffled.

**Success (signal is real):** `(L+upQ_shuffled − L)` median ΔNSE ≤ +0.01.
**Falsification (artifact):** median ≥ +0.02 → the oracle gain is a capacity/regularization
artifact, NOT upstream content. **STOP the chain; the positive direction is in question.**

## Step B — Content test: upstream precipitation (gated on A passing)

**Hypothesis.** Upstream discharge carries content beyond upstream precipitation.

**Design.** Build `upstream_precip` = area-weighted lagged (1d) upstream PRCP(mm/day). Train
L + upstream_precip.

**Read (disambiguation, no falsification):**
- `(L+upPrecip − L)` ≈ +0.037 → the signal is upstream RAIN; discharge/graph machinery
  unnecessary. Cleaner, more deployable finding.
- `(L+upPrecip − L)` ≪ +0.037 → discharge/state carries extra content beyond rain →
  justifies the learned message-passing direction.

## Step C — Lag sensitivity (gated on A passing)

**Hypothesis.** The gain is robust to the upstream→downstream travel-time lag, not a lag-1
coincidence (and not leakage).

**Design.** Re-run oracle upstream_q at lag=0 and lag=2 days (lag=1 already done at +0.037).

**Success:** positive gain same order (+0.02 to +0.05) across lags 0/1/2.
**Falsification/flag:** gain only at one lag, ~0 elsewhere → investigate for leakage before
trusting the headline.

---

## Compute
All stock cudalstm, CPU, ~4 min/run. Step A: 1 new run. Step B: 1. Step C: 2. ≈ 20 min total.

## What we will NOT do
- Will not interpret B as success/failure — it's a disambiguation, reported either way.
- Will not proceed to the realizable learned-upstream model until A passes (signal is real)
  and B clarifies whether discharge-vs-precip even matters.
- Single seed throughout — directional only; any headline needs 3-seed confirmation later.
