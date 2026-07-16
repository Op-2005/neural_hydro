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

## Idea 1 resurrection (April 20, 2026)

**What we committed to.** Return to the original pilot framing, scale it to
Component 0 (183 basins), add an explicit topology-as-features condition, and
run it as a clean three-way ablation. See `idea1.md`.

**Current status.** Component 0 extracted (183 basins, 624 edges, proper depth
distribution). NH baseline config written (`experiments/configs/lstm_component0_baseline.yaml`).
Graph runner parameterized (`experiments/training/train_graph_component0.py`). Nothing
launched — waited on PI meeting before scaling.

---

## Idea 1 reframing — dynamical-systems lens (April 21, 2026, current)

**What changed.** Post-PI meeting (full notes in `JOURNAL.md` 2026-04-21
entry). The plan is reframed around dynamical-systems-on-networks
language, while the A/B/C ablation remains intact under the new framing.

**Reframing in one paragraph.** The trained LSTM exhibits self-stabilizing
dynamics — its rolled-out predictions are dominated by the model's own
hidden-state evolution, not by external forcings. The pilot's +0.013
frozen-graph NSE is now read as the **small real destabilizing-forcing
effect**, and the +0.065 of LSTM weight drift during joint training as
the **LSTM finding a new self-consistent attractor that incorporates the
forcing** — not a confound, but a different mode of the same mechanism.
The research question reframes from "does topology help?" to "what graph
topologies admit external forcings strong enough to destabilize the
LSTM's self-consistent regime in a useful direction?"

**New experiments added to the plan.**
- **E0** (gate): verify LSTM self-stabilization empirically via
  perturbation-recovery and forcing-replacement probes on run-05.
- **E0.5** (gate): 60-epoch loss-saturation curve on the strong baseline.
- **Forcing comparison** (post-gates): compare graph hidden-state
  messages (C) against random noise (C-rand), upstream raw precipitation
  (C-precip), and self-lagged forcing (C-lag).
- **Reproducibility packaging**: `setup.py` + `run.py` for Colab.

**Impact on thinking.** The dynamical-systems framing explains the pilot
results coherently (instead of "+0.078 with caveats"), connects the work
to a richer mathematical literature (network dynamics, graph signal
processing, physics-of-flow-on-graphs), and produces verifiable physical
claims anchored in Saint-Venant / Manning / linear-reservoir-routing
models. The A/B/C ablation is preserved as the empirical core, but each
condition is now interpreted as a different "channel of external
forcing" rather than as a generic "feature ablation."

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

---

## CRS Session Audits

Brief, high-level audits of `/crs` and `/crs-unleashed` invocations.
Newest at the bottom (chronological with the rest of the file). Detailed
reasoning and pre-registrations live in `JOURNAL.md`.

### CRS Session — 2026-04-24 15:08

- **Reviewed:** README.md, idea1.md, JOURNAL.md, CURRENT_STATE.md, INSIGHTS.md, runs/README.md, experiments/README.md.
- **Ran:** E0 self-stabilization probes on run-05 (Probe A perturbation recovery + Probe B forcing replacement), at σ=0.5 canonical and σ=2.0 sensitivity.
- **Result:** PASS — 100% of basins on both probes. Probe A median recovery 1 step at σ=0.5 / 2 steps at σ=2.0. Probe B median deviation 0.007 of natural prediction std.
- **Decision:** E0 marked PASS in `idea1.md`. Dynamical-systems framing empirically grounded. E0.5 (loss saturation) is the next gate.
- **Files:** new `experiments/probes/e0_self_stabilization.py`; new `experiments/analysis_outputs/e0/` outputs (decision_record.json + sigma_2_0 sensitivity, two PNGs, two CSVs); JOURNAL.md entry for 2026-04-24; idea1.md status updated.
- **Caveats:** Probe B has a known weakness — "replace forcing with t-1's" is near-null on dry days. Probe A is the diagnostic test; cite it primarily. A stronger forcing-test (zero-out, random historical day) is queued as supplementary.
- **Next:** E0.5 — 60-epoch loss-saturation curve on the strong baseline (~20 min CPU).

### CRS Session — 2026-04-24 15:14

- **Reviewed:** existing `~/.claude/skills/crs/SKILL.md`; the just-completed E0 session for shape of an audit entry.
- **Ran:** answer-only — no experiments. Skill modifications.
- **Result:** `/crs` Step 5 expanded to "Update memory layer" with mandatory audit-to-CURRENT_STATE + conditional JOURNAL entry. New `/crs-unleashed` skill created with chain-of-2–3, robustness-bundling, hostile-reviewer simulation, and queued-next-sessions plan.
- **Decision:** Both skills now installed and invokable. `/crs` for narrow asks; `/crs-unleashed` for "drive this forward" sessions.
- **Files:** modified `~/.claude/skills/crs/SKILL.md`; new `~/.claude/skills/crs-unleashed/SKILL.md`; this audit entry.
- **Caveats:** none.
- **Next:** unchanged — E0.5 loss-saturation curve still queued for the next invocation.

### CRS Session — 2026-05-06 (Condition C lands; pilot's +0.078 does NOT replicate at scale)

- **Reviewed:** user's overnight Colab session completed Cell 11. Uploaded `experiments/graph_c0_warm_seed42_0605_063025` (real, 30 epochs, 183 basins, 624 edges) plus a duplicate of yesterday's A+B upload.
- **Ran:** `/organize` (cleanup + move) — deleted the duplicate `experiments/drive-download-20260506T052646Z-3-001 2/`; moved Condition C → `runs/16_graph_c0_warm_seed42/` per the established numbered convention; wrote NOTES.md following the run-14/15 template. Re-ran `experiments/analysis/compare_abc_component0.py` with all three conditions; produced final `summary.json`, `per_basin_long.csv`, `per_basin_deltas.csv`, `delta_distributions.png`, `nse_by_depth.png`, `depth_stratified.csv`, `summary_table.txt`.
- **Result (single-seed, seed=42):** A median NSE 0.648, B median 0.591, **C median 0.578**. C − A = **−0.077** at the median; C − B = **−0.021** with much tighter distribution (std 0.082 vs 0.487 for B−A). **Pilot's +0.078 does NOT replicate at scale.** Depth-stratified: A wins at every depth except depth 4 (n=2 noise); gap A−C constant ~0.05 across depths 0–3.
- **Decision:** Honest read — the pilot's positive result was likely a small-N artifact + warm-start optimization-trajectory effect (we already had partial evidence of this from the +0.013 frozen-isolation in run 07). At Component-0 scale with from-scratch protocol, both ablations underperform baseline. **Multi-seed verification is now load-bearing** before any framing-level claim. If multi-seed confirms, this becomes a strong negative result at scale, publishable as a workshop paper aligned with Kirschstein 2024.
- **Files:** new `runs/16_graph_c0_warm_seed42/` + NOTES.md; updated `runs/README.md`, `experiments/analysis_outputs/abc_component0/` (all three conditions now), `JOURNAL.md`, `idea1.md`. Removed duplicate `experiments/drive-download-20260506T052646Z-3-001 2/`.
- **Caveats:** SINGLE SEED. Yesterday's E0.5 multi-seed result showed cross-seed variance of ±0.111 NSE on the 23-basin baseline; with similar variance at scale, the C−A delta could plausibly shift on other seeds. The C − B = −0.021 with tight std is the diagnostic finding: the deficit is not unique to message passing, it's shared between B and C — likely an architecture/optimization-trajectory effect (DirectedGraphLSTM's Python LSTMCell loop vs NH's batched cudalstm).
- **Smoke test:** N/A — analysis only.
- **Deferred:** Multi-seed `MODE='full'` run (4 more seeds = 11, 13, 17, 19, 23). On T4 ~70 hr / ~105 units (in budget for one Pro month, multi-session); on L4 ~25 hr / ~125 units (over budget).
- **Next:** *User decides whether to launch multi-seed.* CRS will interpret once it lands.

### CRS Session — 2026-05-05 (organize Component-0 runs from Drive + first scaled A vs B finding)

- **Reviewed:** user pulled Drive runs into local `drive-download-20260506T052646Z-3-001/`. Conditions A and B from Colab Pro (single-seed, seed=42, Component-0 / 183 basins). Condition C still running on Colab Cell 11.
- **Ran:** organize-style work — moved `A_baseline_seed42_0605_003517` → `runs/14_lstm_component0_baseline_seed42`; moved `graph_c0_topology_features_seed42_0605_005241` → `runs/15_graph_c0_topology_features_seed42`. Archived the incomplete second A run and the Colab-retrained pilot baseline. Patched run-14's `config.yml` to `device: cpu` and ran `nh_run.py evaluate` locally to produce `test/model_epoch030/test_metrics.csv` (which the Colab `train` step alone does not produce). Wrote NOTES.md for both new runs. Built `experiments/analysis/compare_abc_component0.py` (handles partial states; works now with A+B and will work with C when it lands). Updated `runs/README.md` with runs 14/15 and the new Component-0 results table.
- **Result:** First scaled-experiment numbers in hand. **A median NSE 0.648, B median 0.591, per-basin ΔNSE(B−A) median −0.050.** Topology-as-static-features alone *hurts* relative to baseline on Component 0 at seed 42. Depth-stratified plot shows A wins at depths 0–3 (bulk of the network), B and A converge at depth 4 (small n).
- **Decision:** B vs A is the first real Component-0 finding. Pilot's +0.078 doesn't transfer to "topology features for free" at scale. Whether C recovers any gap is the open question. Multi-seed verification is the next no-compute-needed priority once C completes.
- **Files:** see JOURNAL.md 2026-05-05 entry for full file list.
- **Caveats:** All Component-0 numbers are single-seed. Yesterday's E0.5 multi-seed result (cross-seed band 0.111 NSE) suggests these numbers could shift by similar magnitudes at other seeds. A's mean (0.586) is outlier-pulled (one basin at -6.495); median is the robust summary.
- **Smoke test:** N/A — analysis only on top of completed training runs.
- **Deferred:** wait for Condition C from Colab Cell 11 (~5-7 hr). Then run analysis script again to get the full A/B/C comparison.
- **Next:** Once C lands, rerun `experiments/analysis/compare_abc_component0.py` and write the interpretation. Multi-seed run (full MODE in notebook) is the publication-grade follow-up.

### CRS Session — 2026-04-27 (Colab notebook for the publication run)

- **Reviewed:** user's compute pivot — Google Colab Pro / Colab for Research available; supersedes the cloud-GPU recommendation.
- **Ran:** answer-only — wrote `notebooks/colab_publication_run.ipynb` (25 cells, valid nbformat 4.5) implementing the locked A/B/C protocol as a single Run-All notebook. Idempotent (skip-if-done per-seed), writes results to Google Drive symlinked to `runs/`, includes pre-flight smoke test + GPU check + numpy<2 pin. Wrote `notebooks/README.md` covering one-time setup (zip+upload data and code to Drive), running instructions, two result-pull options (Drive desktop sync vs git-push), and compute-estimate table per GPU type.
- **Result:** Compute path is unblocked. User can: zip data + code locally → upload to Drive → open notebook in Colab → Runtime → Run All → ~3-4 hr A100 / ~7-8 hr T4 → pull `summary.json` + `per_basin_per_seed.csv` back to local repo → ask CRS to interpret.
- **Decision:** Colab Pro+ on A100 (Research-tier) is the canonical compute target. Cloud GPU rental ($10) and academic-grant paths remain as fallbacks but are no longer needed.
- **Files:** `notebooks/colab_publication_run.ipynb` (new); `notebooks/README.md` (new); this audit.
- **Caveats:** Notebook assumes either zip-upload-to-Drive flow or git-clone-from-private-GitHub (both supported via Cell 2 options). Tested only for JSON validity locally — first end-to-end run is the smoke test that happens on Colab.
- **Smoke test (locally):** N/A (notebook ships unrun; smoke test happens as Cell 7 inside Colab).
- **Deferred:** Multi-seed pilot 23-basin re-run on local CPU (now superseded by going straight to Component-0 A/B/C on Colab GPU — same data, more decisive result, similar wall-clock).
- **Next:** *User executes the notebook on Colab.* Expected 3-8 hours. Then `crs interpret abc results` to read summary.json + per_basin_per_seed.csv and propose the next step.

### CRS Session — 2026-04-26 (later, `/crs` continuation — multi-seed E0.5 analysis)

- **Reviewed:** background sweep completion notification fired; 5 seeds × 60 epochs all finished.
- **Ran:** wrote `experiments/analysis/plot_e0_5_multiseed.py`; parsed all 5 seeds' output.log files; computed per-seed plateau medians, MADs, and ep10→60 linear slopes.
- **Result:** Within-seed saturation confirmed (all slopes within ±0.0019/epoch, MADs ≤ 0.032). **But cross-seed plateau medians span 0.366→0.478 = 0.111 NSE**, more than 2× the pre-registered ≤0.05 bar. Pilot run-05 (NSE 0.423 at seed 42) sits in the middle of the band.
- **Decision:** **The pilot's +0.078 absolute headline is now contingent on multi-seed verification.** Cross-seed variance in *just baseline* is larger than the headline gap. Framing-side probes (E0, E1, state-space, E0-B') are unaffected. Paper claim shifts from absolute NSE gain to multi-seed-CI graph-vs-baseline comparison. Pivots queued: cheap no-compute path = multi-seed re-run of pilot Conditions A and C on 23 basins.
- **Files:** `experiments/analysis/plot_e0_5_multiseed.py` (new); `experiments/analysis_outputs/e0_5/loss_saturation_multiseed.png` (new); `experiments/analysis_outputs/e0_5/decision_record_multiseed.json` (new); `JOURNAL.md` (new entry); `idea1.md` (status update); this audit.
- **Caveats:** None on methodology — this is the result the multi-seed sweep was designed to produce. The finding is honest and substantive.
- **Smoke test:** N/A (analysis only).
- **Deferred:** Component-0 A/B/C still compute-blocked. Multi-seed pilot 23-basin re-run is the next no-compute step (~5 hrs CPU overnight, would be the natural next session).
- **Next:** Multi-seed pilot 23-basin re-run (Conditions A + C × 5 seeds) — answers whether +0.078 survives at the original pilot scale, before any Component-0 compute spend.

### CRS Session — 2026-04-26 (`/crs` — A/B/C methodology lock, no-compute work only)

- **Reviewed:** state of overnight multi-seed E0.5 (seed 11 done, seed 13 at ep 7/60, seeds 17/19/23 not started — sweep is slower than projected). Yesterday's queued plan items.
- **Ran:** answer-only — no experiments. Single doc-only step: wrote the **A/B/C Publication-Run Protocol** as a locked section in `idea1.md`. Specifies model spec per condition, matched hyperparameters, basin set + edges, reporting metrics, execution checklist, falsification conditions. Critically: pivots to **uniform from-scratch training** across A/B/C (no warm-start), per yesterday's reviewer Q5.
- **Result:** Methodology is locked. Once compute arrives, the publication run is one command per condition × seed (15 runs total for headline; 20 runs all-in including NHDPlus follow-up). No methodology ambiguity remaining.
- **Decision:** All future A/B/C ablation runs use the locked protocol. Pilot warm-start pattern (run 06 etc.) is retired for the publication run.
- **Files:** `idea1.md` (added "A/B/C Publication-Run Protocol" section + Compute spec table); this audit entry.
- **Caveats:** Multi-seed E0.5 is running unusually slowly (1 of 5 done after 24 hrs). Should investigate at next session start — possible CPU contention from the smoke tests yesterday, or a stalled process. Not blocking for now.
- **Smoke test:** n/a — doc only.
- **Deferred:** multi-seed E0.5 analysis (waiting on sweep), Component-0 baseline launch (waiting on compute).
- **Next:** *Get compute.* Once landed, execute the locked A/B/C protocol per `idea1.md` execution checklist.

### CRS Session — 2026-04-25 15:35 (`/crs-unleashed` — multi-seed E0, t=29, state-space, Condition B)

- **Reviewed:** `idea1.md`, `JOURNAL.md`, `CURRENT_STATE.md`, queued next-2-3-sessions plan from yesterday's JOURNAL entry. No new external feedback.
- **Pre-registered:** chain of 4 (multi-seed E0, Probe A at t=29, state-space recovery, Condition B implementation) + 1 background (multi-seed E0.5). All CPU-feasible in this session except the background sweep.
- **Ran:**
   1. **Multi-seed E0** — 6 seeds (11/13/17/19/23/42) × Probe A + Probe B on run-05. Refactored e0 probe with `--seed`. **All 6 seeds: 100%/100% pass, Probe A 1-step recovery on every seed, Probe B max-dev range 0.006-0.010.** Reviewer Q5 (single seed) closed.
   2. **Probe A at t=29** — perturbation immediately before prediction (no recovery time). Verdict was "FAIL" by the recovery-within-5-steps criterion (artifactual: no time available). Actual deviation: median 0.098, range [0.056, 0.266], 23/23 basins below 30% threshold. Reviewer Q2 closed.
   3. **State-space recovery** — new probe `e0_state_space_recovery.py` measuring `‖Δh‖_norm` alongside `|Δy|_norm`. **‖Δh‖_norm: 0.478 (t=T) → 0.06 (t=T+1) → 0.012 (t=T+5).** Both spaces recover within 5 steps to <10% of natural variance. Strongest result of the session — true contracting dynamics, not head-orthogonality. Reviewer Q1 closed.
   4. **Condition B implementation** — added `topology_features` variant to `train_graph_component0.py` with `compute_topology_features()` (5 z-normalized scalars: depth, in-deg, out-deg, transitive upstream count, log upstream-area ratio) + `warm_start_with_extra_input_dims()` for partial warm-start. **Pre-training NSE = 0.423 = exact baseline match — pipeline verified.** 2-epoch smoke test ran clean.
   5. **Multi-seed E0.5 in background** — 5 seeds (11/13/17/19/23) × 60-epoch retrain. seed 11 at ep 18/60 as of session close. Analysis next session.
- **Result:** Framing now reviewer-defensible across all reviewer-2 questions raised yesterday. Multi-seed E0 + state-space recovery give genuinely robust evidence. Condition B infrastructure ready; A/B/C ablation can run when compute lands. Multi-seed E0.5 analysis is the only outstanding gate experiment.
- **Decision:** Mark all 4 foreground items as PASS in idea1.md. Defer A/B/C uniform-warm-start methodology lock to next session (cheap, code-only, no compute). Multi-seed E0.5 verdict at start of next session.
- **Files:** `experiments/probes/e0_self_stabilization.py` (new CLI args); `experiments/probes/e0_state_space_recovery.py` (new); `experiments/probes/run_e0_multiseed.sh`, `run_e0_5_multiseed.sh` (new); `experiments/configs/lstm_study_network_strong_60ep_template.yaml` (new); `experiments/training/train_graph_component0.py` (topology_features variant + helpers); `experiments/analysis_outputs/e0/` (12 new files: 6 multi-seed pairs + state_space + t29); JOURNAL.md entry; idea1.md status; this audit.
- **Caveats:** Multi-seed E0 zero-variance is partly a saturation artifact of the binary recovery criterion (1-step ceiling). Smoke-test +0.038 on Condition B is overfit-inflated; not a real performance claim. Condition B partial-warm-start is smoke-test-only; publication A/B/C should use uniform no-warm-start across conditions.
- **Smoke test:** all probes' decision_record.json files written; figures saved; smoke test pre-training NSE matches baseline exactly (0.423).
- **Deferred to next session:** multi-seed E0.5 analysis, A/B/C pre-registration document, Component-0 baseline launch (when compute lands).
- **Next:** Multi-seed E0.5 analysis (cheap, ~15 min); A/B/C methodology lock (~30 min code-only); then await compute for Component-0 launch.

### CRS Session — 2026-04-24 17:35 (`/crs-unleashed` — chain of E1 + E0.5 + E0-B')

- **Reviewed:** `idea1.md`, `JOURNAL.md`, `CURRENT_STATE.md`, `runs/README.md`, `experiments/README.md`, recent git log, the queued TODOs in JOURNAL 2026-04-24 entry.
- **Pre-registered:** chain of three independent gate experiments (E1, E0.5, E0-B'), all CPU-feasible in ~30 min total. All pre-registration written before any executed.
- **Ran:**
   1. **E1** — E0 probes on weak baseline (run 03, no basin encoding). Refactored `e0_self_stabilization.py` with argparse (`--baseline-dir`, `--sigma`, `--probe-b-mode`, `--out-suffix`) to support variants. Probe A 100% / 1-step recovery; Probe B 100% / 0.008 max-dev. **PASS — identical signature to strong baseline.**
   2. **E0.5** — 60-epoch retrain of strong baseline via new config `lstm_study_network_strong_60ep.yaml` + new analysis script `plot_e0_5_saturation.py`. Strict pre-registered criterion FAILED on val NSE (too tight for natural noise); pragmatic reading from figure + linear regression: **val NSE saturated at 0.355 ± 0.022 MAD from epoch 5 onward, slope −0.0004/epoch (slightly declining). Train loss keeps descending (0.090 → 0.065 ep30→60).** Pilot's epoch-30 stop near-optimal on val. Classic overfitting past epoch 5.
   3. **E0-B'** — stronger Probe B variants (zero-out forcing; random historical day). Both 100% pass; max-dev 0.035 (zero) and 0.033 (random) — 5× stronger replacement than t-1 (0.007) but still well below 30% threshold. Probe B caveat from morning JOURNAL entry resolved.
- **Result:** Framing significantly hardened. E1 rules out "encoding creates the attractor." E0.5 confirms pilot wasn't under-trained (was slightly overfit, both baseline and graph in same regime, comparison fair). E0-B' resolves the methodological objection from earlier in the day.
- **Decision:** All three pass. Dynamical-systems framing now stands on substantially harder evidence than this morning. Next session = multi-seed replication (gates publication credibility); after that, t=29 perturbation + state-space recovery probes.
- **Files:** `experiments/probes/e0_self_stabilization.py` (refactored with CLI); `experiments/configs/lstm_study_network_strong_60ep.yaml` (new); `experiments/analysis/plot_e0_5_saturation.py` (new); `experiments/analysis_outputs/e0/{probe_a_recovery,probe_b_forcing,decision_record}_{weak_baseline,probeB_zero,probeB_randomday}.{csv,png,json}` (9 new files); `experiments/analysis_outputs/e0_5/{loss_saturation.png,decision_record.json}` (new); `runs/lstm_study_network_strong_60ep_2404_173615/` (new training run, 60 ckpts); `idea1.md` (status updated); `JOURNAL.md` (full entry with hostile-reviewer Q&A and queued next sessions).
- **Caveats:** E0.5 strict criterion fail → pragmatic reading documented in JOURNAL with full slope/MAD numbers (not goalpost-moving). All single-seed; multi-seed verification queued.
- **Smoke test:** all probes' decision_record.json files written; figures saved; analysis script verdicts consistent.
- **Deferred to next session:** multi-seed E0+E0.5 replication; Probe A perturb-at-t=29 condition; state-space recovery measurement; Condition B implementation.
- **Next:** Multi-seed E0 + E0.5 replication (5 seeds, ~3 hrs CPU, can run overnight).

### CRS Session — 2026-04-24 16:10 (`/crs` discovery fix)

- **Reviewed:** `~/.claude/` directory contents (no `commands/` folder existed).
- **Ran:** answer-only — created the missing `~/.claude/commands/` directory and three thin slash-command files (`crs.md`, `crs-unleashed.md`, `organize.md`) that delegate to the existing `~/.claude/skills/<name>/SKILL.md` source-of-truth files via Read instructions. `$ARGUMENTS` is passed through so `/crs do X` works.
- **Result:** All three slash commands now invokable. Architecture: skills (`~/.claude/skills/`) hold the canonical instructions; commands (`~/.claude/commands/`) are user-typeable thin wrappers that load the skill content. Single source of truth, two invocation paths.
- **Decision:** Slash commands → `~/.claude/commands/<name>.md` (user-typeable). Skills → `~/.claude/skills/<name>/SKILL.md` (assistant-internal, invoked by the Skill tool). Future skills must populate both directories.
- **Files:** new `~/.claude/commands/{crs,crs-unleashed,organize}.md`; this audit entry.
- **Caveats:** none.
- **Next:** unchanged — E0.5 loss-saturation curve is the queued research step. The user can now invoke `/crs-unleashed` to drive it.

### CRS Session — 2026-04-24 15:32 (`/crs-unleashed organize`)

- **Reviewed:** repo top-level layout, `experiments/{probes,analysis_outputs}/`, `runs/_archive/` and its 2 subfolders, `topology_analysis/phase0_scaffold/`, git status, `~/.claude/skills/`.
- **Ran:** (1) renamed `~/.claude/skills/crs_unleashed/` → `crs-unleashed/` and updated all 17 internal refs via `sed`; updated 4 cross-refs in `CURRENT_STATE.md` from `/crs_unleashed` → `/crs-unleashed`. (2) Created new `/organize` skill at `~/.claude/skills/organize/SKILL.md` codifying the 8 reorganization patterns from this session. (3) Audited the project for org gaps and fixed the top 4: wrote `experiments/probes/README.md`, `experiments/analysis_outputs/e0/NOTES.md`, `experiments/analysis_outputs/README.md`, `runs/_archive/README.md`, plus a `NOTES.md` in each of the 2 archived run folders.
- **Result:** `/crs-unleashed` is now invokable as a slash command (was failing before due to underscore). `/organize` is a new standalone skill. Six new docs land all the loose ends from the E0 session and earlier reorgs.
- **Decision:** Skill-name convention is hyphenated; future skills will follow it. `/organize` is now the canonical structural-cleanup skill, separate from `/crs` (strategy) and `/crs-unleashed` (execution).
- **Files:** renamed `~/.claude/skills/crs_unleashed/` → `crs-unleashed/`; new `~/.claude/skills/organize/SKILL.md`; new `experiments/probes/README.md`, `experiments/analysis_outputs/README.md`, `experiments/analysis_outputs/e0/NOTES.md`, `runs/_archive/README.md`, `runs/_archive/graph_lstm_warm_noedge_WEAK_1904_152734/NOTES.md`, `runs/_archive/graph_lstm_edgefeat_warm_frozen_WEAK_1904_160530/NOTES.md`; modified `CURRENT_STATE.md`.
- **Caveats:** none structural. The `paper_research.md` → `research_papers.md` rename and the disappearance of `SESSION_HANDOFF.md` are user-side cleanups noted in passing; no action needed.
- **Smoke test:** zero stale `crs_unleashed` references repo-wide; all training-script imports + hardcoded paths resolve cleanly post-session.
- **Deferred:** none high-severity. Low-severity polish items (e.g., unifying `phase0_scaffold/outputs/summary.md` into a `README.md`) deliberately not done — already serving the same purpose.
- **Next:** unchanged from prior — E0.5 loss-saturation curve.

### CRS-Unleashed Session — 2026-05-12 (5cond factorial post-mortem)

- **Reviewed:** all 15 runs in `experiments/5cond_factorial/multi_condition_ablation/`; loss histories from each `run_config.json`; per-basin paired ΔNSE; DirectedGraphLSTM source (`experiments/training/train_graph_lstm.py`) `__init__`, `forward`, `train_epoch`.
- **Headline:** L (cudalstm) median NSE 0.653; G/G+T/G+M/G+T+M medians 0.609 / 0.605 / 0.583 / 0.586. Six pairwise contrasts produced; bootstrap 95% CIs across seeds; depth/area stratification + outlier-trimmed all in `experiments/analysis_outputs/5cond_component0/RESULTS.md`.
- **Diagnosis (load-bearing claim found):** L − G = +0.050 NSE is **not an architecture confound but a training-budget confound.** NH cudalstm trainer samples random (basin, window) pairs → ~2,610 gradient steps/epoch. DirectedGraphLSTM trainer samples whole-windows-of-all-183-basins → ~14 gradient steps/epoch. 186× ratio. Loss is still decreasing at epoch 30 (0.37 → 0.36 in last 4 epochs) — graph trainer is severely undertrained.
- **Decision:** the L − G contrast is invalid as currently designed. Before claiming "graph signal doesn't help at scale" we must train the graph variants under matched gradient budget. The G+T/G+M/G+T+M − G contrasts (within-graph-trainer) ARE valid and show topology/messages are net null-to-slightly-negative, but they sit on an undertrained baseline.
- **Files:** `runs/5cond_factorial/` symlinks added so `compare_5conditions.py` discovers the runs; full RESULTS.md + per-basin CSVs + figures regenerated in `experiments/analysis_outputs/5cond_component0/`. No code changes yet.
- **Caveats:** the within-graph-trainer contrasts assume the architecture is "settled enough" to compare variants — possibly true, possibly not.
- **Next:** the training-budget rematch experiment is queued — see JOURNAL.md.

### CRS Session — 2026-05-12 (architecture deep-audit + framework redesign)

- **Reviewed:** DirectedGraphLSTM class (`train_graph_lstm.py:98-294`), `compute_topology_features` (`train_graph_component0.py:98-146`), message passing forward, edge feature normalization, training pipeline (per-window batching, best-checkpoint selection, validation handling).
- **Produced (3 root-dir files):** `experiments/5cond_factorial/analysis/5cond_run_analysis.md` (results-only digest of 5cond_factorial), `experiments/5cond_factorial/analysis/architecture_analysis.md` (deep technical critique of model + topology features + message passing + training pipeline, with prioritized improvement tiers and 3 paper-narrative paths), `experiments/5cond_factorial/analysis/testing_framework_proposal.md` (6-step diagnostic ladder with pre-registration discipline, compute estimates, and a decision diagram).
- **Key findings beyond training-budget confound:**
  1. **Basin one-hot encoding subsumes the 5 topology features** (5/681 ≈ 0.7% of static input is drowned by 671-dim one-hot). Explains G+T − G ≈ 0.
  2. **Mean aggregation gives equal weight to a 1km² parent and a 100km² parent** — physically wrong. Area-weighted aggregation is the obvious fix.
  3. **Single linear `W_msg_edge` + saturating `tanh(W_out(m))` residual** — message function is too shallow, and zero-init W_out forces the graph path to grow slowly from nothing.
  4. **Test-set leakage in `train_graph_lstm.py:635-643`** (best-checkpoint selection on test data). Production trainer is unaffected but the helper code is buggy.
  5. **`compute_topology_features` uses `shortest_path_length` despite docstring saying "longest path"** — minor bug affecting depth values on multi-parent basins (~10% of basins).
- **Recommended path:** Start with Step 1 (matched-budget L control on Colab, 15 min compute). Then Step 2 (one-hot ablation, 6 hr). Both pre-registered before launching.
- **Files:** new `experiments/5cond_factorial/analysis/5cond_run_analysis.md`, `experiments/5cond_factorial/analysis/architecture_analysis.md`, `experiments/5cond_factorial/analysis/testing_framework_proposal.md` at repo root.
- **Caveats:** no code changes yet — pure analysis pass. Tier-1 architectural fixes (area-weighted aggregation, 2-layer message MLP) are recommended but require pre-registration before implementation.
- **Next:** write `preregistration_step1.md`, launch matched-budget L on Colab.

### CRS-Unleashed Session — 2026-05-12 (Step 1 executed: matched-budget L on CPU)

- **Reviewed:** `experiments/5cond_factorial/analysis/5cond_run_analysis.md`, `experiments/5cond_factorial/analysis/architecture_analysis.md`, `experiments/5cond_factorial/analysis/testing_framework_proposal.md` (just-written), NH `BaseTrainer` (has `max_updates_per_epoch` — directly usable).
- **Produced (3 files):** `experiments/5cond_factorial/preregistration_step1.md` (pre-registered before any data), `experiments/5cond_factorial/preregistration_step2.md` (queued, not run), `experiments/training/train_matched_budget_lstm.py` (standalone matched-budget cudalstm trainer; ~120 LOC).
- **Executed:** Step 1 on CPU. 3 seeds × 420 gradient steps cudalstm on Component 0. Total time ~25 min.
- **Headline result (third-category outcome, neither pre-registered prediction):**
  - L_420 cross-seed median NSE = **0.502**
  - L_420 − G paired median Δ = **−0.100** (CI [−0.105, −0.092])
  - 431 of 549 paired comparisons strongly favor G; only 9 favor L_420. **94.5% basin × seed pairs prefer G over L_420.**
- **What it tells us:** "Matched gradient steps" is the wrong matching variable. Graph trainer's effective per-step batch is 47k examples vs cudalstm's 256, so at matched steps the graph trainer has seen 200× more data. The original L − G gap of +0.050 NSE in 5cond is **NOT** explained by step count (both L and G saw ~20M examples, but L still beats G).
- **What's still open:** the L − G gap at matched-examples (the original 5cond comparison) is real but the mechanism is now unclear. Candidate mechanisms: (1) Adam gradient-noise scale effect (small-batch L vs large-batch G); (2) data-exposure pattern (random per-sample vs whole-window); (3) actual architectural difference (cuDNN vs Python loop). Next experiment differentiates these.
- **Caveats:** Step 1 result is robust (3 seeds, n=549 paired comparisons, tight bootstrap CIs). Robustness check with alternative batch sizes for L_420 was pre-committed but deferred to a follow-up — the 200× per-step example gap makes the matching question moot regardless.
- **Files:** new `train_matched_budget_lstm.py`, `preregistration_step1.md` (with results section), `preregistration_step2.md`. `L420_seed{11,13,17}/` runs in `runs/5cond_factorial/`.
- **Next:** train G with smaller batches (batch=32 → 4,576 steps per epoch × 30 = 137k gradient steps) — tests whether more gradient updates at the same example exposure closes the L − G gap.

### Organize Session — 2026-05-12 (post-Step-1 cleanup)

- **Audited:** `experiments/5cond_factorial/{configs,multi_condition_ablation,notebooks}/`, `runs/5cond_factorial/L420_seed*/`.
- **Fixes applied:** 6 new docs total — 3 NOTES.md (Pattern 4, results-bearing folders) + 3 READMEs (Pattern 2, subfolder grouping).
- **Files created:**
  - `runs/5cond_factorial/L420_seed{11,13,17}/NOTES.md` — per-seed result + reference to pre-registration.
  - `experiments/5cond_factorial/configs/README.md` — L vs L420 config table.
  - `experiments/5cond_factorial/notebooks/README.md` — single-notebook description.
  - `experiments/5cond_factorial/multi_condition_ablation/README.md` — 15-folder run-output index.
- **Smoke test:** path references in new docs verified to point to existing files.
- **Deferred:** none — root README and existing subfolder docs are already in good shape; no medium/low items worth this session.

### Organize Session — 2026-05-12 (cleanup pass)

- **Audited:** root *.md files, `notebooks/` folder, README content.
- **Fixes applied:** (1) Grouped the 4 5cond analysis docs into a new `experiments/5cond_factorial/analysis/` folder with its own README. (2) Archived the superseded root-level `notebooks/colab_publication_run.ipynb` (old A/B/C) to `notebooks/_archive/`. (3) Rewrote root README as a clean signpost — removed runs-table content that drifts; kept premise + pointers + credits.
- **Files moved:** 4 root .md → `experiments/5cond_factorial/analysis/`; `notebooks/{colab_publication_run.ipynb,README.md}` → `notebooks/_archive/` (preserving git history via `git mv`).
- **References updated:** CURRENT_STATE.md, JOURNAL.md, preregistration_step1.md, preregistration_step2.md, multi_condition_ablation/README.md — all bare-name refs to the moved files now use full paths.
- **Smoke test:** no broken refs; all moved files still resolvable from their referrers.
- **Deferred:** none — clean state.

### CRS Session — 2026-06-20 15:10

- **Reviewed:** post_meeting_plan.md, 5cond invariant (recomputed mean±std), Drive deliverable decks (wk1/4/10), local_subgraphs batch I just built.
- **Ran:** launched the full local-subgraph batch (6 subgraphs × 3 conditions × 3 seeds = 54 CPU trainings, background task bw81vl3qn). First subgraph L training confirmed in progress.
- **Result:** pending — multi-hour CPU run. Tracked metric: per-seed median NSE → mean±std; key contrast G+T+M − L per subgraph.
- **Decision:** pre-registered the local-scale test. Success = G+T+M − L > 0 on ≥1 subgraph (graph beats plain LSTM at small scale). Falsification = ≤0 on all 6 → Phase-4 architectural redesign, NOT further shrinking.
- **Files:** new experiments/local_subgraphs/preregistration_local_scale.md (committed 0678273 batch + prereg).
- **Caveats:** subgraphs span 2 HUCs each (graph-coherent, not climate-coherent); only sg_texas_pilot is single-region. 30-epoch runs on small data may under/over-fit differently than the 183-basin run — watch the std.
- **Next:** on batch completion, run analyze_subgraphs.py, judge vs pre-registration, append JOURNAL entry with the verdict.

### CRS-Unleashed Session — 2026-07-12 (analysis-only hardening chain)

- **Reviewed (deep orient):** git log (last 30), current_implementation.md, README, all 6 analysis/*.md, MULTISEED/CONFOUND/COMPLIANCE, FORWARD_PLAN, JOURNAL tail (open TODOs), analysis-script signatures. Key reuse discovery: every headline run stores `test_results.p` (per-timestep obs/sim) → any metric/stratification is zero-compute.
- **Diagnosed (top-3 load-bearing claims):** (1) routing-not-artifact [medium-conf, high-imp → tested]; (2) deployable gain significant vs null [high-conf, high-imp → tested]; (3) log-NSE/KGE robustness [medium → tested].
- **Pre-registered:** `preregistration_hardening_chain.md` — 3 gated analysis-only steps, written before execution.
- **Ran (zero training compute), all 3 steps PASSED:**
  - **Step A — significance** (`analyze_significance.py` → `analysis/SIGNIFICANCE.md`): realizable-vs-L Wilcoxon p=6e-19; **realizable-vs-null p=2.3e-12**, bootstrap 95% CI on median Δ [+0.011,+0.022] excludes 0. Per-seed: sig in 3/3 (vs-L) and 2/3 (vs-null; seed17 p=0.061, disclosed). Upgrades "all-seeds-positive" → "statistically significant, capacity-controlled, with CI."
  - **Step B — routing vs feature-magnitude confound** (`analyze_feature_magnitude_confound.py` → `analysis/FEATURE_MAGNITUDE_CONFOUND.md`): feature magnitude *decreases* with depth (corr −0.369) while gain *increases* → confound runs OPPOSITE to effect. Within-tercile deep>shallow 3/3. Partial corr(Δ, depth | area, fmag)=+0.149 (p=4e-4); reverse fmag|depth,area drops to +0.080 (p=0.061, n.s.). Depth is load-bearing; feature scale is not. Resolves the CONFOUND.md depth-vs-n_upstream ambiguity: the routing variable is **depth**.
  - **Step C — metric honesty** (`analyze_metric_honesty.py` → `analysis/METRIC_HONESTY.md`): log-NSE realizable Δ stable +0.027→+0.030 across 100× eps sweep (null stays negative). KGE seed-13 dip **localized**: realizable improves timing (KGE-r positive in ALL 3 seeds, Δr +0.018/+0.021/+0.005); the dip is a variability-ratio (γ) overshoot, not a timing loss.
- **Net effect:** three of the paper's most-attackable joints hardened — significance (with CI), mechanism (confound runs backwards), and metric honesty (KGE weakness now understood as γ-overshoot with r always improving). No result changed sign; the record is materially stronger and better-scoped. No new runs; all on stored predictions.
- **Reviewer-2:** addressed pooling non-independence (per-seed corroborates), collinearity (direction argument dominates partial corr), eps non-standardness (swept 100×, sign stable), γ-overshoot (honest limitation, r-always-positive shows timing intact).
- **Caveats/TODO:** oracle seed-11 `test_results.p` lost in drive merge (metrics.csv survives) → Step C oracle column blank at seed11 only, non-load-bearing. Follow-up: report at NH's default fixed eps for direct comparability.
- **Next:** paper skeleton (science + hardening complete); the 3 new artifacts feed the results/discussion directly.

### CRS-Unleashed Session — 2026-07-12 (later): routing-baseline chain (queue re-scoped)

- **Re-scoped the stale queue (validity-first).** Queued items 2–3 were mis-costed: oracle seed-11 re-eval needs RETRAINING (checkpoint lost in drive merge; only config+test survive), and the scale curve needs GPU (no subgraph runs on disk). Neither is CPU-cheap. Instead executed the reviewer baseline `FORWARD_PLAN.md` names as "near-free once EXP-0 infra exists" and which the paper genuinely lacks.
- **Pre-registered:** `preregistration_routing_baseline_chain.md` — 3 gated analysis-only steps.
- **Ran (zero training), all 3 PASSED:**
  - **Step A — no-ML routing baseline** (`analyze_routing_baseline.py` → `analysis/ROUTING_BASELINE.md`): least-squares routing fit on TRAIN, scored on TEST. R1 pure routing +0.324; R2 routing+L_sim +0.675; L +0.654; realizable +0.686; oracle +0.717. **Realizable beats every no-ML baseline (ML earns its complexity)** — but the margin over the strong R2 (+0.010) is modest and honestly reported. This is the reviewer's first question, now answered in-paper.
  - **Step B — per-depth significance** (`analyze_depth_significance.py` → `analysis/DEPTH_SIGNIFICANCE.md`): per-stratum Wilcoxon. depth0 n.s. (p=0.24, expected); depth1 p=2.6e-9; depth2 p=4.7e-12; depth3 p=8.4e-4; depth4 n.s. (n=6). **The routing gain is statistically significant exactly where water arrives and absent at headwaters** — the gradient now has per-stratum teeth, not just a median trend.
  - **Step C — consolidated paper table** (`build_paper_table.py` → `analysis/PAPER_TABLE.md`): Table 1 (conditions × NSE/KGE/log-NSE × Δ-vs-L p), Table 2 (routing baselines), Table 3 (depth significance). The Results section assembled in one auditable place. Surfaced honestly: null-vs-L is weakly sig (p=0.047), which is why realizable-vs-null (p=2.3e-12, prior artifact) is the load-bearing contrast.
- **Net effect:** the paper gains its missing reviewer baseline, upgrades the depth story to per-stratum significance, and now has a single consolidated Results table. All from stored data — zero training.
- **Reviewer-2:** addressed R2-near-miss (R2 uses L's own sim; standalone no-ML is R1 at +0.324; honest margin), train/test split (no leakage), depth-4 n.s. (n=6, no power), seed-11-only baseline (fullspan eval availability; 3-seed extension is a cheap follow-up).
- **Corrected queue:** oracle 3-seed completion and scale curve both require compute not available this session; logged accurately for next time.
- **Next:** paper skeleton — every Results artifact now exists; the write-up is the natural next move.

### CRS-Unleashed Session — 2026-07-14 (graph-robustness: over-connectivity threat CLOSED)

- **Reviewed:** git log (post-90400e0 clean tree), last queued plan, and the prior session's graph-similarity finding (the one unaddressed validity threat). Reuse insight: the R1/R2 lstsq routing baseline scores any upstream-flow feature WITHOUT training, so alternative graphs are testable at zero compute.
- **Diagnosed (top-3):** (1) routing signal survives on a hydrography-realistic pruned graph [low-conf, high-imp → test first]; (2) depth hierarchy stable under pruning [med, high]; (3) edge-choice-noise robustness [med, med].
- **Pre-registered:** `preregistration_graph_robustness_chain.md` — 3 gated zero-training steps.
- **Ran (`analyze_graph_robustness.py` → `analysis/GRAPH_ROBUSTNESS.md`), all 3 PASSED:**
  - **Step A — over-connectivity is NOT the source.** Full graph (in-degree mean 4.16/max 15) R1 NSE +0.325. Prune to hydrography-realistic in-degree≤2 (nearest) → +0.326, **100% retained**; even in-degree≤1 (76% of edges deleted) retains 98%. The signal lives in the nearest parents; the heuristic's excess edges contribute ~nothing. **The study's single biggest reviewer attack is closed.**
  - **Step B — depth hierarchy stable:** 95% of basins keep depth ±1 under k=2 pruning; DAG preserved; max depth 5→4. The routing-signature (depth gradient) is not an edge-density artifact.
  - **Step C — edge-noise robust:** random 20% edge dropout → +0.324 ± 0.002 (100%, spread 0.006); 40% → 99%. Anchored in aggregate structure, not specific edges.
- **Key nuance:** `nearest`-parent pruning is invariant (physically meaningful — nearest = shortest travel time); `smallest-ratio` pruning drops to 81%, proving the metric responds to graph changes (not saturated). Signal is invariant across a 4× density range (in-degree 1→4.16).
- **Scope (honest):** tested via the R1 lstsq signal-content proxy, not a full LSTM re-train per graph (GPU follow-up). Signal-content invariance is the load-bearing fact and is established; NHDPlus ground-truth edges remain the definitive future check.
- **Reviewer-2:** addressed proxy-vs-LSTM (content invariance → same info for the LSTM), k=2-not-real-NHDPlus (invariance across the density range is the point), metric-saturation (smallest-ratio moves it → responsive), seed cherry-picking (deterministic seeds, 5 draws, mean±std).
- **Next:** paper skeleton — the graph caveat is now contained with quantified evidence; write-up is unblocked.

### CRS Session — 2026-07-16

- **Reviewed:** last queued plan (JOURNAL), all L/L_upQ/fullspan run availability on disk, GPU/MPS status, training entry points, prior-run configs.
- **Ran:** Part 1 — 3-seed no-ML routing baseline (`analyze_routing_baseline_3seed.py` → `analysis/ROUTING_BASELINE_3SEED.md`), zero-training. Attempted Parts 2/3 training; smoke-tested the pipeline.
- **Result:** Part 1 PASS — realizable-vs-R2 margin **widens to +0.019** (multi-seed) from the single-seed +0.010; all 3 seeds beat R1 (+0.324). Corrects an understatement in Table 2. Parts 2/3 **BLOCKED**: `nh_run.py train` SIGABRTs (exit 134) at startup on this Mac (AVX illegal-instruction from an AVX-compiled dep) on both mps AND cpu. Prior runs were all trained on Colab (`cuda:0`); this machine only does analysis.
- **Decision:** Part 1 delivered locally. Parts 2 (oracle seed-11 restore) & 3 (k=2 LSTM graph check) deferred to GPU — staged turnkey. NHDPlus (the other "definitive check" option) is data-blocked (no flowline data on disk); recommend k=2 LSTM re-train over NHDPlus since it needs only Colab, not new data.
- **Files:** `analyze_routing_baseline_3seed.py`, `analysis/ROUTING_BASELINE_3SEED.md`, `preregistration_baseline_completion_and_k2.md` (+ dated amendment), `configs/L_upQ_k2_component0_seed11.yaml`, `configs/L_upQpred_k2_component0_seed11.yaml`, `features/upstream_q_{obs,pred}_component0_k2_lag1.p`, `topology_analysis/.../component0_edges_k2.csv`.
- **Caveats:** Part 1 R1 is seed-independent by construction (uses observed upstream_q); the multi-seed gain is in the R2/L/LSTM rows. Training genuinely cannot run here — not a config fix.
- **Next:** on Colab GPU — run Part 2 (1 config) + Part 3 (2 configs, all staged); ~20-40 min each. Then paper skeleton.

### CRS-Unleashed Session — 2026-07-16 (later): k=2 LSTM graph check LANDED — over-connectivity closed at model level

- **Reviewed:** the two Colab-trained k=2 runs added to root (L_upQ_k2, L_upQpred_k2 seed11), full-graph seed-11 references, k=2 predicted feature; relocated both runs to canonical `runs/topology_ablation/component0/` (gitignored, alongside siblings).
- **Result — the definitive graph check PASSED at the LSTM level (not just the R1 proxy).** Paired Δ vs L on the 150 connected basins, seed 11:
  - k=2 **realizable** Δ = **+0.021 NSE** (p=4e-4), **+0.034 log-NSE**, 65% basins positive — ~78% of full-graph realizable (+0.034 same basins), inside the pre-registered ±0.010 band.
  - k=2 **oracle** Δ = **+0.049** > full-graph +0.046 (p=2e-12) — the oracle *strengthens* under pruning.
- **Interpretation:** pruning the over-connected heuristic graph (624→266 edges, in-degree≤2, real-confluence-like) does NOT kill the gain; it slightly sharpens the observed-Q oracle. The heuristic's excess edges were not doing the work — routing lives in the nearest (shortest-travel-time) parents. Consistent with the 2026-07-14 R1-proxy result; now confirmed on the trained model.
- **Verdict: the over-connectivity threat — the study's biggest remaining validity concern — is closed at BOTH the signal-content and trained-model level.** The paper can present the heuristic-edge caveat AND show the result is robust to it.
- **Decision:** wrote `analyze_k2_graph_check.py` → `analysis/K2_GRAPH_CHECK.md` (reproducible). Study is now empirically complete for a regional workshop paper; write-up is the next move.
- **Caveats:** k=2 is single-seed (11), single pruning rule (nearest). Part 2 (oracle seed-11 full-graph restore) was NOT brought back — its results.p still missing; log-NSE/KGE oracle columns at seed11 remain blank (non-load-bearing; seeds 13/17 complete).
- **Files:** `analyze_k2_graph_check.py`, `analysis/K2_GRAPH_CHECK.md`; k=2 runs relocated to `runs/`.
- **Next:** paper skeleton (empirical case complete + graph threat closed). Optional robustness: 3-seed k=2 replication; oracle seed-11 restore for metric-column completeness.
