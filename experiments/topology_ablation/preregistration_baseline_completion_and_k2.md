# Pre-registration — 3-Seed Routing Baseline + Oracle Completion + k=2 LSTM Graph Check

**Date:** 2026-07-16. **Author:** /crs session.
**Compute reality (verified this session):** No CUDA; **MPS available** (Apple Silicon). All
three `_Lfullspan_eval` runs (seeds 11/13/17) present on disk with `results.p`. L checkpoints
present all seeds; **L_upQ seed11 checkpoint + results.p LOST** (drive merge); L_upQ seeds 13/17
present. **NHDPlus flowline data is NOT on disk** — NHDPlus validation is out of scope this
session (data-acquisition task; surfaced to user).

---

## Part 1 — 3-seed routing baseline (ZERO training; do first)

**Motivation.** The no-ML routing baseline (`analyze_routing_baseline.py`) was single-seed
(seed 11) because only seed-11 fullspan eval was thought available. All 3 fullspan evals are in
fact present → extend to 3 seeds for a multi-seed baseline, matching the rest of the study.

**Method.** Per seed s∈{11,13,17}: fit R1 (a·upstream_q+b) and R2 (a·upstream_q+c·L_sim+b) on
TRAIN 1990-99 using that seed's fullspan L-sim, score TEST 2005-08. Report median NSE per
predictor, mean±std across seeds, vs the LSTM conditions.

**Success:** the prior single-seed conclusion holds across seeds — realizable and oracle LSTM
beat R1 at all 3 seeds; R2 remains the strong-but-beaten baseline. **Falsification:** if the
LSTM-beats-R1 ordering flips at any seed, the "ML earns its complexity" claim is seed-fragile →
report honestly.

## Part 2 — Oracle seed-11 completion (MPS re-train; gated on smoke test)

**Motivation.** `PAPER_TABLE.md` / `METRIC_HONESTY.md` oracle log-NSE & KGE columns are blank at
seed 11 (results.p lost). Re-train L_upQ seed11 to restore the complete 3-seed oracle metrics.

**Method.** Re-train stock cudalstm with the existing `L_upQ_component0_seed11.yaml` config
(device→mps), evaluate, regenerate `test_results.p`. **Determinism check:** the restored median
NSE must reproduce the recorded seed-11 oracle (0.703 ± tolerance) — same discipline as the
2026-07-01 realizable seed-11 restore, which reproduced exactly.

**Success:** median test NSE within ±0.005 of the recorded 0.703 → deterministic restore, oracle
columns completed. **Falsification:** if it does not reproduce, the recorded seed-11 oracle
number is suspect → flag, do not silently overwrite the record.

## Part 3 — k=2 pruned-graph LSTM re-train (the definitive graph check; gated on Part 2 pipeline working)

**Motivation.** The graph-robustness chain (2026-07-14) showed the *R1 signal-content proxy* is
invariant to pruning the over-connected graph to hydrography-realistic in-degree≤2 (100%
retained). The named follow-up: confirm the **actual LSTM** — not just the proxy — retains its
gain on the k=2 graph. This is the executable form of "definitive graph check" (NHDPlus being
data-blocked).

**Method (two-stage, all on k=2 nearest-parent pruned edges):**
1. Build `upstream_q` (observed, oracle) on the k=2 graph → train `L_upQ_k2_seed11`.
2. Build `upstream_q_pred` (predicted, realizable) on the k=2 graph using the existing
   seed-11 fullspan predictions → train `L_upQpred_k2_seed11`.
3. Compare paired Δ vs L (seed 11) on k=2 against the full-graph seed-11 Δ
   (oracle +0.037, realizable +0.027).

**Success (pre-registered):** the k=2 realizable Δ vs L is within ±0.010 of the full-graph
realizable Δ (+0.027), i.e. retains ≥ ~65% of the gain and stays clearly positive. This confirms
the LSTM-level gain is not an over-connectivity artifact.

**Falsification:** if the k=2 realizable Δ collapses (<+0.010 or negative), the LSTM-level gain
DID depend on the excess edges even though the R1 proxy did not → material finding, report the
proxy/LSTM divergence honestly; do NOT re-scope to hide it.

**Robustness (bundled, cheap):** also report k=2 oracle Δ (upper bound under pruning).

---

## Discipline
- Pre-registered before execution. Amend only by dated append.
- Smoke-test the training pipeline (1 epoch) before any 30-epoch run — no multi-run job on an
  unvalidated pipeline.
- Determinism: seed fixed in config; MPS non-determinism tolerance ±0.005 on medians.
- lstsq baselines fit on TRAIN, scored on TEST — no test-period fitting.
- A falsification is reported, not redesigned around. NHDPlus is explicitly out of scope (data
  not on disk) — surfaced, not faked.

---

## AMENDMENT 2026-07-16 — compute block discovered; Parts 2 & 3 deferred to GPU

**Executed:** Part 1 (3-seed routing baseline) — DONE, zero-training. Result: realizable-vs-R2
margin widens to +0.019 (multi-seed) from the single-seed +0.010; all 3 seeds beat R1. PASS.

**Blocked:** Parts 2 & 3 require model training, which **cannot run on this machine**. Discovered
via 1-epoch smoke test: `nh_run.py train` aborts with SIGABRT (exit 134) at startup on both
`device: mps` and `device: cpu` — an AVX illegal-instruction crash from an AVX-compiled
dependency on this CPU. Confirmed the prior runs were all trained on Colab (`device: cuda:0` in
their configs on disk); this Mac has only ever done analysis. Not faked with a degraded run
(none is even possible).

**Staged for turnkey GPU execution (all zero-training prep done this session):**
- Part 2: `configs/L_upQ_component0_seed11.yaml` — set `device: cuda:0`, train, evaluate.
- Part 3: `configs/L_upQ_k2_component0_seed11.yaml` + `configs/L_upQpred_k2_component0_seed11.yaml`
  (generated this session), pointing at `features/upstream_q_{obs,pred}_component0_k2_lag1.p`
  (built this session from the k=2 nearest-parent pruned edge set `component0_edges_k2.csv`,
  266 edges). On GPU: train both, compare Δ vs L against full-graph +0.037/+0.027.

**Success/falsification criteria unchanged** from the original pre-registration above.
