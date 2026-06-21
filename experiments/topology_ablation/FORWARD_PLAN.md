# Forward Experiment Tree — Built With Intent, Gated on Phase 1

**Created 2026-06-21.** Designed while Phase 1 (the encoding × topology 2×2) runs.
Every experiment below answers a *specific question raised by the one before it*. Nothing
is built speculatively; each is gated on a concrete Phase-1 outcome. Single seed until a
signal appears; multi-seed only at publication.

The research question is unchanged: **does river-network structure improve LSTM
streamflow prediction, and under what conditions?** The encoding axis is a *control*,
not the thesis.

---

## The bounding experiment (run regardless of Phase 1 outcome)

### EXP-0: Upstream-discharge ORACLE upper bound
**Intent.** The PI's framing: the LSTM self-stabilizes; the question is what *external*
information usefully destabilizes it. Static topology features are constant; learned
message passing is a weak proxy. The *strongest possible* structural signal is the actual
lagged **observed** discharge of upstream basins — the literal water arriving downstream.
If this oracle doesn't help, **no** learned message-passing scheme can — it bounds the
entire graph enterprise.

**Design.** Stock NH cudalstm + one extra dynamic input `upstream_q` (area-weighted mean
of upstream basins' lagged observed discharge, mm/d). Two conditions: L (no upstream_q) vs
L+upQ. Built + verified: `build_upstream_discharge_feature.py` (oracle, lag=1d).

**Reads as.**
- L+upQ ≫ L → structure carries strong signal; pursue a realizable message-passing model
  (Phase 2). This is the *ceiling*.
- L+upQ ≈ L → upstream flow is uninformative for next-day downstream flow at this scale →
  graph methods cannot help; pivot to the honest negative paper with this as the killer
  control.

**Status:** infrastructure built + smoke-verified (CPU). Wire into a config + run next.
Cheap (stock cudalstm). **This is the single most decisive experiment in the program** and
should run right after Phase 1 regardless of how Phase 1 lands.

---

## Gated branches from Phase 1

Phase 1 measures: `topo_benefit_without_ID = (L_noID+T) − L_noID` and
`topo_benefit_with_ID = (L+T) − L`.

### Branch A — topology helps WITHOUT one-hot, null WITH (the predicted outcome)
Redundancy hypothesis confirmed: structure carries signal that basin-ID memorization masks.

- **EXP-A1 — message passing in the no-ID regime (the original thesis, asked fairly).**
  Does *dynamic* message passing add anything beyond *static* topology features, with the
  one-hot off? Reuse the existing DirectedGraphLSTM machinery — but only now is it
  justified, and it must be trained to convergence (the prior undertraining bug). Gated +
  pre-register.
- **EXP-A2 — per-feature importance.** Which of the 5 topology features carries the signal?
  Drop-one-out on the L_noID+T condition. Hypothesis: `total_upstream_area` dominates
  (discharge scales with contributing area). If it's just upstream area, the story
  simplifies to "contributing-area awareness," physically clean. Stock cudalstm, cheap.
- **EXP-A3 — entity-aware encoding (deployable version).** Does an EA-LSTM /
  `embcudalstm` (which *embeds* statics instead of concatenating) let topology features
  help even WITH the one-hot on? If yes, that's a practitioner-relevant finding (people use
  the one-hot). Stock NH model, config-only.

### Branch B — topology does NOT help even without one-hot
Static features can't capture network position, OR the signal isn't there.

- **EXP-B1 — EXP-0 becomes load-bearing.** If even the oracle upstream-discharge helps,
  the failure is the *static-feature representation*, not the idea → go to message passing.
  If the oracle also fails → strong, clean negative result (structure uninformative);
  publish the controlled decomposition (corroborates Kirschstein 2024 but far more
  rigorous — we ruled out the encoding confound, the training confound, and the oracle).
- **EXP-B2 — richer dynamic structure.** Upstream lagged *precipitation* (not discharge) as
  a dynamic input — tests whether any upstream signal helps before giving up. Stock cudalstm.

### Branch C — scale-dependent (helps on subgraphs, not component0, or vice versa)
The professor's hypothesis directly.

- **EXP-C1 — scale sweep.** Topology benefit vs network size: walker subgraphs at
  n ≈ {10, 20, 40, 80, 183}, measure `topo_benefit_without_ID` at each. A clean
  "structure helps at small scale, washes out at large scale" curve is itself a publishable
  figure and the professor's hypothesis made quantitative. Stock cudalstm; cheap per point.

---

## Strong baselines a reviewer will demand (build before publication)

- **Routing baseline (no ML).** Downstream Q ≈ area-weighted lagged sum of upstream Q +
  local runoff. If our graph-LSTM can't beat simple physical routing, the ML machinery
  isn't earning its complexity. This is the `upstream_q` feature used as a *direct
  predictor*, not an input — a near-free baseline once EXP-0 infra exists.
- **mclstm (mass-conserving LSTM).** Stock NH. The physically-grounded model the
  hydrology audience expects as a comparison.

---

## Priority order (intent-ranked)

1. **Phase 1** (running) — does the encoding confound explain the negatives? Gates everything.
2. **EXP-0 oracle** — bounds the whole enterprise. Run regardless. Decisive + cheap.
3. Branch experiment matching Phase-1's outcome (A1/A2/A3, or B1/B2, or C1).
4. Reviewer baselines (routing, mclstm) before any publication run.
5. Multi-seed scale-up of whatever survived.

## What we will NOT do
- Build Phase-2 message passing before EXP-0/Phase-1 justify it (the prior mistake).
- Pre-commit to the result's sign.
- Chase a positive number across configs/seeds and report only the winner.
- Turn the encoding axis into the paper's thesis — it's a control.
