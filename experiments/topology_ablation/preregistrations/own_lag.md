# Pre-registration — own lagged discharge (camera-ready, reviewer request)

**Written 2026-10-21, before the runs.** Requested by the Area Chair and Reviewer BaCR:
*"a comparison with the target basin's own lagged discharge would help assess the benefit of the
neighboring inputs"* and *"does adding the basin's own lagged flow absorb the gain? That would be a
direct test of the storage idea."*

## Why this is the right test

The paper's surviving mechanism candidate is that neighbour discharge reports **catchment state**
(storage) that the meteorological forcings do not observe. A basin's *own* discharge yesterday is
the most direct available proxy for its own state. So the reviewers' question is the sharpest
remaining test of our own candidate: if local state is what the neighbour signal carries, a basin
should be able to read most of it off itself.

This also answers the framing objection (BaCR) that calling observed neighbour flow an "oracle"
overstates things, since yesterday's gauge readings are in fact available for a next-day forecast.
Autoregression on observed discharge is standard in operational rainfall–runoff ML
(Feng et al. 2020 "data integration"; Nearing et al. 2022 compares it against data assimilation and
finds autoregression both more accurate and cheaper). The paper currently cites neither, and a
reviewer who knows that literature will ask why.

## Conditions

All use stock `cudalstm` with the byte-identical config of every published condition; only the
appended input changes. Three seeds (11, 13, 17).

| Condition | Appended input |
|---|---|
| `L_selfQ` | the basin's **own** observed discharge, lagged 1 day (`self_q`) |
| `L_selfQ_upQ` | own lagged discharge **and** network neighbour discharge (`self_q`, `upstream_q`) |
| `L_selfQ_knn2` | own lagged discharge **and** nearest-2 neighbour discharge |

Baselines already on disk: `L`, `L_upQ`, `L_upQknn2` (3 seeds each).

**Input construction.** `build_own_lag_inputs.py`. Own discharge uses the same loader and
area source as every published condition (forcing-header area in m², via
`load_camels_us_discharge`), shifted one day. Missing days are forward-filled with the last
available reading — what an operational forecaster would hold — and any value still missing at the
start of record is 0. Filling is required rather than cosmetic: NH drops training samples
containing a NaN input but retains them at evaluation, so an unfilled NaN would both poison that
day's prediction and change the evaluated day set relative to `L`, breaking the pairing. CAMELS
discharge is near-complete over this window (train 1990–99: mean 0.05 % of days missing, one basin
at 10 %; test 2005–08: 0 % across all 183), so filling touches almost nothing.

**Verified before running:** the same script, driven from the committed edge lists, reproduces the
published `upstream_q` network and kNN-2 feature files **bit-for-bit** (max |diff| = 0.000e+00 over
all 183 basins, index identical). The new conditions are therefore directly comparable to the
existing runs rather than merely similar.

## Hypotheses and pre-committed readings

Primary contrasts, paired per basin, Wilcoxon signed-rank, **weakest of three seeds**, on the 150
connected basins (so the comparison with the network conditions is on a common set; the all-183
column is also reported since `self_q` is defined everywhere).

**H1 — own lagged flow helps on its own.** `L_selfQ` − `L` > 0.
Expected: yes, and plausibly by more than the neighbour gain. This is not a threat to the paper;
it is the expected behaviour of autoregression and the reason the literature uses it.

**H2 (the decisive one) — does own flow *absorb* the neighbour gain?**
`L_selfQ_upQ` − `L_selfQ`, and `L_selfQ_knn2` − `L_selfQ`.

| Outcome | Pre-committed reading |
|---|---|
| **Increment survives** (> 0 at every seed, weakest-seed $p<0.05$) | Neighbour discharge carries spatial information **beyond** the target's own state. This *strengthens* the paper: the gain is not merely a proxy for local autocorrelation, and the proximity finding stands as a statement about spatial information. |
| **Increment collapses** (not significant, or < +0.005) | Most of what the neighbour signal provides is recoverable from the basin's own history. The paper's **core claims are unchanged** — C1 (static inert), C5 (proximity not connectivity), C6 (kNN beats network) are all contrasts *among neighbour selectors* and do not depend on the increment over an autoregressive baseline. What must change is scope: the gain is then an upper bound on what spatial input adds once local state is available, and §3.4's storage reading gains direct support rather than losing it. Report it plainly in Results and Limitations. |

**H3 — ordering.** If proximity (not connectivity) is what matters, then conditional on own flow
the nearest-2 increment should be ≥ the network increment. This is a weaker, directional check.

## What this cannot do

It cannot distinguish "storage" from any other persistent local state. A basin's own discharge is a
proxy for everything slowly varying about that basin, so a surviving increment bounds the spatial
contribution but does not identify a mechanism. The paper's standing refusal to claim storage as
identified is unaffected either way.

## Falsification of the paper's thesis — explicitly not at stake

No outcome here can overturn C5 or C6, because both are comparisons between neighbour sets at fixed
encoding. A reader should be able to see that from this file, which is why it is written before the
runs.
