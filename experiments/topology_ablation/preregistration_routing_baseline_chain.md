# Pre-registration — Reviewer-Baseline + Significance-by-Depth Chain

**Date:** 2026-07-12 (later). **Author:** /crs-unleashed session (queue-execution, re-scoped).
**Compute:** ZERO training. Uses the observed `upstream_q` feature (oracle input) and stored
per-timestep obs/sim from `test_results.p`, both already on disk.

**Why this chain (re-scoping the stale queue).** The queued "oracle seed-11 re-eval" needs
RETRAINING (checkpoint lost in the drive merge, only config+test survive — not a 10-min eval),
and "scale curve" needs GPU (no subgraph runs on disk). Neither is CPU-cheap. Meanwhile
`FORWARD_PLAN.md` names a reviewer baseline the paper does NOT yet have and calls it
"near-free once EXP-0 infra exists" — which it is. This chain fills that genuine gap and
strengthens the results table. All steps contribute directly to the paper.

---

## Step A — No-ML routing baseline (the missing reviewer baseline; gates B)

**The reviewer question.** "+0.022 NSE from upstream flow — but does a trivial physical
routing rule get the same? Is the LSTM earning its complexity?"

**Design (no training).** For each downstream basin, build naive predictors of test-period Q
from data already on disk and score them against observed Q:
- **R1 — pure routing:** predicted Q = a·(observed upstream_q, lag1) + b, coefficients fit by
  least squares on the TRAIN period (1990–1999) per basin, applied to TEST (2005–2008). Uses
  only upstream flow — the strongest "structure without ML" predictor.
- **R2 — routing + local:** predicted Q = a·upstream_q + c·(L baseline's own sim) + b, fit on
  train, applied to test. Adds the LSTM's local prediction as the "local runoff" term.
- Compare median test NSE of R1, R2 against the LSTM conditions (L, L+upQ oracle, L+upQ_pred).

**Success (BOTH):** median NSE of L+upQ (oracle) AND L+upQ_pred (realizable) each exceed R1's
median NSE on the connected basins (those with upstream). I.e. the LSTM's learned use of the
same upstream signal beats using it naively.

**Falsification.** If R1 or R2 matches/beats the realizable LSTM, the paper's "ML earns its
complexity" claim is undermined — report honestly; this is the single most important thing to
learn, and it stops the chain (the results table would need reframing first).

**Robustness (bundled).** Report on headwaters (fmag=0) too — there R1 is degenerate (no
upstream), so the LSTM should trivially win; confirms the comparison is set up correctly.

## Step B — Significance-by-depth of the realizable gain (gated on A passing)

**Hypothesis.** The routing gain has statistical teeth per stratum, not just a median gradient:
realizable Δ is significantly > 0 at depth ≥ 1 and NOT at depth 0 (headwaters).

**Design.** Per-depth paired Wilcoxon signed-rank (one-sided) on the realizable per-basin Δ,
pooled seeds. Report per-depth n, median Δ, p.

**Success.** depth 0: not significant (p ≥ 0.05, ideally ~0 median). depth ≥ 1: significant
(p < 0.05) in at least depths 1 and 2 (the well-populated strata).

**Falsification.** If no depth stratum is individually significant, the depth gradient is
descriptive only — scope the routing claim to "median gradient" not "per-stratum significant."

## Step C — Consolidated publication results table (gated on B passing)

**Deliverable (enhancement).** One `analysis/PAPER_TABLE.md` — the Results table a reviewer
reads first: every condition (L, L+upQ, L+upQ_pred, L+upQshuf) × {NSE, KGE, log-NSE} as
mean±std across seeds, with paired-Δ significance vs L, PLUS the R1 routing-baseline row. This
consolidates the six analysis files into the single table the paper needs. No new stats beyond
A/B and prior artifacts — pure assembly for auditability.

---

## Discipline
- Pre-registered before execution. Amend only by dated append.
- Least-squares coefficients for R1/R2 fit on TRAIN, applied to TEST — no test-period fitting.
- A falsification stops the chain at that step; no mid-session redesign.
- Reviewer-2 after each non-trivial result.
