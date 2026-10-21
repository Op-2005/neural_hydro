# Pre-registration — Graph-Robustness Chain (over-connectivity → depth-stability → edge-dropout)

**Date:** 2026-07-14. **Author:** /crs-unleashed session.
**Compute:** ZERO training. Rebuilds the `upstream_q` feature on ALTERNATIVE graphs and scores
each via the no-ML lstsq routing baseline (fit on TRAIN 1990-99, scored on TEST 2005-08) — the
same machinery as `analyze_routing_baseline.py`. No model training anywhere.

**Motivation.** The graph-similarity analysis (prior session) quantified the study's single
biggest unaddressed validity threat: the heuristic edges OVER-CONNECT vs real hydrography.
Child in-degree is mean 4.16 / max 15 among connected basins; 66 of 150 children have >3
parents. Real river confluences almost always join 2-3 tributaries. A hostile hydrology
reviewer's first attack: "your routing gain is an artifact of an unrealistically dense graph."
This chain tests that head-on, cheaply, before any paper write-up. If the signal is
pruning-robust, the graph caveat is contained; if not, it is a material limitation to report.

**Method note (why lstsq, not training).** Training a cudalstm per graph variant is infeasible
this session (no GPU; ~hours CPU each). But the routing signal's *strength* is measurable
without a model: fit Qhat = a·upstream_q + b on train, score test-NSE. This R1 statistic is a
faithful, monotone proxy for "how much predictive content the upstream-flow feature carries"
under a given graph. Relative comparisons across graph variants are valid; absolute NSE is a
lower bound on what the LSTM would extract.

---

## Step A — Pruned-graph robustness (over-connectivity test; gates B)

**Hypothesis.** The routing signal survives capping in-degree to a hydrography-realistic level.
Rebuild `upstream_q` keeping only the k NEAREST parents per child (by distance_km), for
k ∈ {1,2,3}, and score each via R1 (pure-routing lstsq test-NSE, median over connected basins).

**Success.** R1 median test-NSE at cap k=2 retains ≥ 70% of the full-graph R1 median NSE.
(Signal is not an artifact of over-connection.)

**Falsification.** If cap k=2 retains < 50% of full-graph R1 NSE, the routing result depends
materially on the heuristic's excess edges. Report as a limitation; STOP the chain (do not
proceed to depth/dropout — the premise is undermined).

**Robustness (bundled).** Report k=1 and k=3 as a sensitivity sweep over the cap level. Also
report "nearest" vs "largest-area" parent-selection rule as a second pruning criterion, since
which parents to keep is itself a modeling choice.

## Step B — Depth-structure stability under pruning (gated on A passing)

**Hypothesis.** Pruning to k=2 preserves the graph's depth hierarchy.

**Success.** ≥ 80% of basins retain depth within ±1 of their full-graph depth after k=2
pruning, AND max depth is preserved within 1. (The depth-gradient story is not an artifact of
edge density.)

**Falsification.** If pruning collapses/fragments the depth structure (many basins change depth
by >1, or max depth drops sharply), the depth-gradient result is heuristic-density-dependent →
scope the routing-signature claim accordingly.

## Step C — Edge-dropout sensitivity (robustness sweep; gated on B passing)

**Hypothesis.** The routing signal is stable to random edge-choice noise.

**Success.** Under random 20% edge dropout, R1 median test-NSE stays within 15% of the
full-graph value, averaged over 5 random draws (report mean ± std). Also report 40% dropout.

**Falsification.** High variance across draws (std > 20% of mean) → the graph is too noisy to
anchor the claim; the specific edge set is doing load-bearing work.

---

## Discipline
- Pre-registered before any execution. Amend only by dated append.
- lstsq coefficients fit on TRAIN, scored on TEST — no test-period fitting.
- Random dropout uses fixed per-draw seeds (Date.now/random-safe) for reproducibility.
- A falsification stops the chain at that step; no mid-session redesign.
- Reviewer-2 after each non-trivial result (non-optional for /crs-unleashed).
- HONEST SCOPE: R1 lstsq NSE is a proxy for signal strength, not the LSTM's actual gain.
  Conclusions are about the signal's ROBUSTNESS across graphs, not absolute skill.
