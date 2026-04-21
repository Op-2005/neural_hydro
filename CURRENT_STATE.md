# Current State — Chronological Log

A brief, dated walk through every experiment, what it showed, and how it
changed the plan. For the current active direction see `idea1.md`. For the
set-aside direction see `idea2/`.

---

## Phase 0 — Scaffold (Jan–Feb 2026)

**What we ran.** Built `topology_analysis/phase0_scaffold/`: BasinGraph +
NodeStateMatrix data structures, heuristic edge inference, a minimal MPNN,
synthetic chain/tree signal-decay experiments.

**What it showed.** On 10 HUC-01 headwater basins, the heuristic edge inference
(50 km, area ratio ≥ 1.1) returns an **empty edge set** — those basins have no
upstream-downstream relationships. Signal decay on synthetic chains confirmed
the standard over-squashing finding: multi-hop message passing on deep trees
loses information exponentially with depth.

**Impact on thinking.** Confirmed we needed a different basin set — one with
actual topology — before a graph experiment made sense.

---

## Phase 1 — Network discovery (April 2026)

**What we ran.** `topology_analysis/phase1_network_discovery/discover_network.py`
on all 671 CAMELS-US basins with wider parameters (150 km, area ratio ≥ 1.5,
elevation-decreasing required).

**What it showed.** 1298 directed edges across 584 basins, forming 46 connected
components. Top 5: Component 0 (183 nodes, depth 4, eastern US), Component 1
(72, depth 3, Pacific NW), Component 2 (33, depth 2), **Component 3 (23,
depth 3, HUC-12 Texas)**, Component 4 (19, depth 3).

**Impact on thinking.** Selected Component 3 (23-basin Texas) as the pilot study
network — small enough for fast iteration, deep enough for depth effects. Left
Component 0 (183 basins) as the scale-up target.

---

## Phase 2 pilot — 23-basin experiments (April 19–20, 2026)

### Baseline and headline (runs 03, 05, 06)

**What we ran.** Weak baseline (no basin encoding, run 03) → strong Kratzert-
style baseline (basin encoding, run 05) → warm-started Graph-LSTM with edge
features (run 06).

**What it showed.** Baseline median NSE 0.423 → Graph +0.078 (0.501). Headline
result of the pilot.

**Impact on thinking.** Proof-of-concept that graph message passing could
improve NSE on this network. Triggered the ablation program.

### Ablations (runs 07–11)

**What we ran.** Frozen-LSTM isolation (07), Jiang direction term (08), softmax
attention (09), sigmoid gate (10), pruned edges (11).

**What it showed.**
- Frozen LSTM: only +0.013 pure-graph NSE. Most of the +0.078 headline is
  LSTM weight drift during joint training, not message passing.
- Attention / sigmoid gate / Jiang diff all within ±0.005 of mean aggregation
  (0.492–0.496). Error correlations between variants: 0.994–0.999.
- Pruning "bad-parent" edges did NOT rescue the basins the theory predicted.
  The "bad-parent poisoning" hypothesis was wrong — LSTM drift is
  optimization-path-dependent, not graph-content-dependent.

**Impact on thinking.** The headline gain is real but mechanistically modest.
Aggregation-family architecture search has saturated. Pivoted focus to
either (a) the ungauged setting or (b) a larger-scale test.

### Ungauged experiment (runs 12, 13)

**What we ran.** Ungauged baseline (run 12, 20 train / 3 held-out, no basin
encoding) then graph-LSTM warm-started on it (run 13).

**What it showed.** 2 of 3 held-out basins improved (+0.043 on 08158700,
+0.107 on 08189500); the middle-node 08164300 collapsed −0.58 because its
held-out parent's LSTM was inaccurate AND it was itself parent to another
held-out basin. Chain-contamination failure mode identified.

**Impact on thinking.** Graph helps ungauged leaves but breaks on ungauged
middle nodes. The failure mode is specific and actionable — confidence-gated
messages, leaf-only held-out sets, or parent-variance edge features are
concrete fixes.

---

## Idea 2 excursion (April 20, 2026, set aside same day)

**What we ran.** Rewrote `HYPOTHESIS.md` as a pre-registered mechanistic claim:
*DirectedGraph-LSTM's temporal lag is a time-domain analog of Jiang's spatial
gradient operator → preserves high-frequency hidden content → benefit scales
with basin depth.* Wrote `idea2/spectral_analysis.py` and ran it on runs 05 /
06 / 07.

**What it showed.** Graph+warm's high-freq residual power vs. baseline: depth
0 +104% (LSTM drift destroys high-freq fidelity where graph adds nothing),
depth 1 −1.5%, depth 2 −9.7%, depth 3 +2.4%. The FROZEN variant showed
directionally-correct monotone behavior (depth 0 → 0%, depth 1 → −10.6%,
depth 2 → −17.6%, depth 3 → +3.6%) but below the pre-registered 15% bar, with
n=2 statistical noise at deep strata.

**Impact on thinking.** Framing was elegant but hard to defend in a 15-minute
conversation. User preferred the original, simpler research question.
Artifacts moved to `idea2/`. Spectral outputs preserved there.

---

## Idea 1 resurrection (April 20, 2026, current)

**What we committed to.** Return to the original pilot framing, scale it to
Component 0 (183 basins), add an explicit topology-as-features condition, and
run it as a clean three-way ablation. See `idea1.md`.

**Current status.** Component 0 extracted (183 basins, 624 edges, proper depth
distribution). NH baseline config written (`experiments/configs/lstm_component0_baseline.yaml`).
Graph runner parameterized (`experiments/training/train_graph_component0.py`). Nothing
launched — waiting on compute-resource decision from the professor meeting.

---

## What the week produced

- A strong pilot-scale empirical result (+0.078 NSE) with a mechanistic
  decomposition (+0.013 pure graph + ~+0.065 LSTM drift).
- A documented failure mode for graph methods under PUB (chain contamination).
- A characterization of how far architectural aggregation variants can take us
  (essentially nowhere beyond mean aggregation; they all converge).
- A negative result on the "bad-parent poisoning" hypothesis.
- Infrastructure ready for the scaled experiment (183-basin Component 0).

## What it didn't produce — what we still need

- Multi-seed results (every number is single-seed).
- Ground-truth edges (all heuristic).
- Statistical power at depth (n=2 at depths 2 and 3).
- Compute for the scale-up (CPU is 8× slower at 183 vs 23 basins).
- A clear publication venue and deadline.
- A PI sign-off on the scaled basin set and forcing product.
