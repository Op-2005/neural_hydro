# Step A — No-ML Routing Baseline (the reviewer baseline)

Zero training. Least-squares routing coefficients fit on TRAIN 1990-1999, applied to TEST 2005-2008 (no test-period fitting). Seed 11 (fullspan eval available). Connected basins only (those with upstream). Pre-reg: `preregistration_routing_baseline_chain.md`.

## Median test NSE on connected basins (n=150)

| predictor | median NSE | note |
|---|---|---|
| **R1 — pure routing** (a·upstream_q + b) | +0.3241 | no ML, upstream flow only |
| **R2 — routing + local** (a·upstream_q + c·L_sim + b) | +0.6752 | no ML, + LSTM's local pred |
| L (LSTM baseline) | +0.6538 | ML, no upstream |
| L+upQ (oracle) | +0.7171 | ML + observed upstream |
| L+upQ_pred (realizable) | +0.6857 | ML + predicted upstream |

## Pre-registered verdict

- L+upQ (oracle) and L+upQ_pred (realizable) both exceed R1 (pure routing): **True** (oracle +0.7171, realizable +0.6857 vs R1 +0.3241)
- realizable exceeds R2 (routing+local): **True** (realizable +0.6857 vs R2 +0.6752)

**PASS — the LSTM's learned use of upstream flow beats naive physical routing; the ML earns its complexity.**

*Interpretation.* Pure routing (R1) alone reaches median NSE +0.324 — upstream flow genuinely carries predictive content even without ML (this is WHY the LSTM+upQ gain is real, not spurious). But the full LSTM baseline already reaches +0.654 using local forcings the routing rule ignores, and L+upQ_pred (+0.686) combines both — so the ML is not redundant with routing; it integrates upstream flow WITH local rainfall-runoff, which naive routing cannot.

## Robustness — headwaters (R1 undefined: no upstream)

33 headwater basins have no upstream_q, so R1/R2 routing is undefined there; only the LSTM predicts them. Confirms the comparison is scoped to connected basins, where routing is even defined.

