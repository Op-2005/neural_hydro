# Own lagged discharge — does the basin's own flow absorb the neighbour gain?

**Ran 2026-10-24.** Camera-ready reviewer request (Area Chair + Reviewer BaCR).
Pre-registration: `../preregistrations/own_lag.md`, written before these runs existed.
Conditions: `L_selfQ`, `L_selfQ_upQ`, `L_selfQ_knn2`, seeds 11/13/17, stock `cudalstm`,
configs identical to every published condition apart from the appended input.

Paired per-basin Wilcoxon signed-rank, weakest of three seeds, 150 connected basins.
Values here are cross-seed means of per-seed medians; see `VERIFIED_NUMBERS.md` once
`verify_all_numbers.py` carries these rows.

## Headline

| Contrast | ΔNSE | per-seed | weakest p |
|---|---|---|---|
| `L_selfQ` − `L` | **+0.1356** | +0.1295 / +0.1504 / +0.1269 | 3.9e−23 |
| network increment (`L_selfQ_upQ` − `L_selfQ`) | **+0.0052** | +0.0042 / +0.0027 / +0.0087 | **0.294** |
| nearest-2 increment (`L_selfQ_knn2` − `L_selfQ`) | **+0.0024** | +0.0007 / +0.0001 / +0.0065 | **0.749** |

For reference, unconditional on own flow: network +0.0431, nearest-2 +0.0806.

**Pre-registered verdict: the increment collapses.** The nearest-2 increment is below the
+0.005 threshold *and* not significant; the network increment is not significant at any
seed. Both land in the pre-registration's "collapse" branch.

## Magnitudes in context (vs `L`, connected basins)

| Input | ΔNSE |
|---|---|
| own lagged discharge alone | +0.1356 |
| nearest-2 neighbour discharge alone | +0.0806 |
| network neighbour discharge alone | +0.0431 |
| own + network | +0.1488 |
| own + nearest-2 | +0.1433 |

A basin's own discharge yesterday is worth roughly **three times** the river network and
**1.7×** the best neighbour selector. Adding either neighbour set on top of it changes
little.

## Robustness

**KGE.** The increment is *negative* under KGE — network −0.0040 (p=0.977), nearest-2
−0.0079 (p=1.0) — while `self_q` − `L` is strongly positive (+0.1197, p=9.4e−26). So the
collapse is not an NSE artifact; if anything NSE is the metric most favourable to the
increment.

**All 183 basins** (`self_q` is defined everywhere, unlike the network input): network
increment +0.0041 (p=0.675), nearest-2 +0.0022 (p=0.906). Same conclusion off the
connected subset.

**Selector contrast, conditional on own flow.** `L_selfQ_knn2` − `L_selfQ_upQ` = −0.0011,
two-sided p=0.951 — indistinguishable. Unconditionally the same contrast is +0.0334
(p=1.8e−7), the paper's C6. **So the nearest-gauge advantage is specific to the
no-own-flow setting**, which parallels the existing finding that it vanishes under
predicted neighbour flow (C8).

**Flow bands** (RMSE reduction, mm d⁻¹, connected basins). Official values are in
`VERIFIED_NUMBERS.md` → *Own lagged discharge* → low-flow block, computed by the same
`by_band()` as the paper's C9 (below each basin's **strict** median), which it reproduces
exactly (+0.015 / +0.043 / +0.059 per seed).

| Band | network-pred over `L` (C9) | `self_q` over `L` | network increment over `self_q` |
|---|---|---|---|
| below median | +0.039 | **+0.195** (p=1.7e−25) | +0.006, sign-flips at seed 13 (p=0.43) |
| highest decile *(exploratory, ad-hoc banding, not in VERIFIED_NUMBERS)* | +0.087 | +0.519 | +0.023 |

The low-flow gain the paper attributes to neighbour discharge is **largely recoverable
from the basin's own history**, and the residual increment is unstable across seeds.

## Reading

Own discharge is a far stronger proxy for local catchment state than any neighbour's
discharge, and it absorbs most of what the neighbour signal provides. Two consequences:

1. **The storage reading gains direct support.** §3.4 proposed that neighbour discharge
   reports how much water the surrounding landscape is holding — a state the
   meteorological forcings do not observe. If that is what the signal carries, the
   basin's own discharge should carry it better, and it does. This is the paper's own
   candidate mechanism confirmed by the test the reviewers asked for.
2. **The spatial claims need a scope sentence, not a retraction.** C1 (static inert), C5
   (proximity not connectivity) and C6 (nearest-2 beats network) are contrasts *among
   neighbour selectors at fixed encoding*. None is computed against an autoregressive
   baseline, so none is affected. What changes is the framing of *why* spatial input
   helps: it is a substitute for unavailable local state, not an independent source of
   predictive information.

**H3 (ordering) is not supported.** Conditional on own flow the nearest-2 increment
(+0.0024) is *below* the network increment (+0.0052), the reverse of the pre-registered
expectation. Both are null, so the ordering of two nulls carries no weight — report it as
uninformative rather than as evidence against proximity.

## Scope

Own discharge proxies everything slowly varying about a basin, so a collapsed increment
bounds the spatial contribution without identifying a mechanism. The paper's standing
refusal to claim storage as *identified* is unchanged.

Operationally this is the standard autoregression / data-integration setup
(Feng et al. 2020; Nearing et al. 2022, which finds autoregression both more accurate and
cheaper than data assimilation). CAMELS discharge is near-complete over this window —
0.05 % of training days missing, 0 % of test days — so the forward-fill in
`build_own_lag_inputs.py` touches almost nothing. Nearing et al. identify missing data as
autoregression's main weakness, which does not bite here but would in an operational
deployment.
