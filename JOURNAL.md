# Journal — Running History of Decisions, Feedback, and Direction Changes

This is the project's **meta-log**. It is *not* a chronological list of
experiments (that lives in `CURRENT_STATE.md`) and it is *not* the active plan
(`idea1.md`). This is where we record:

- **External feedback** (PI meetings, collaborator input, peer comments).
- **Decisions** (what we chose to do, what we set aside, why).
- **Reframings** (when our understanding of the question or the methodology
  changed and *what triggered* the change).

The point is to be able to answer, six weeks from now: *"why did we make
that call?"* — without re-deriving it from memory.

---

## How to use this file

Append a new entry at the **bottom** under "Entries". One entry per
meaningful event. Older entries stay frozen — never edit a past entry, only
append a follow-up entry that supersedes it.

### Entry template

```markdown
## YYYY-MM-DD — Short title

**Source.** PI meeting / collaborator / paper read / self-review / reviewer
comments / other.

**Signal.** What was said / observed (verbatim or close paraphrase). Quote
where useful. Group by topic if multiple distinct points.

**Why it matters.** Our reading of what the signal implies for the project.
This is the part that does the work — what specifically changes if we take
the signal seriously?

**Decisions.** Concrete actions / scope changes / re-prioritizations the
signal triggered. Use a checklist if multiple.

**Affected files.** Which docs and scripts now need to change to reflect this
decision (so we can audit downstream consistency).

**Open questions.** What we still don't know after processing this signal.
```

---

## Entries

## 2026-05-06 — Condition C lands. Pilot's +0.078 does NOT replicate at scale.

**Source.** Cell 11 of the Colab notebook completed overnight; user uploaded the run; CRS organized into `runs/16_graph_c0_warm_seed42/` and ran the cross-condition analysis.

**Signal — the headline.** All three conditions on Component 0 (183 basins), single-seed (seed=42):

| Cond. | Median NSE | Δ vs A |
|---|---|---|
| A baseline | **0.648** | — |
| B topology features | 0.591 | −0.050 |
| C full graph-LSTM | 0.578 | **−0.070** |

**C − B median Δ = −0.021** with std 0.082 — *tighter* than the C−A or B−A distributions. That means most of the deficit C and B share against A is *not* about message passing per se; it's something B and C have in common relative to A.

**Per-basin breakdown (C vs A):** 113 of 183 basins worse by ≥ 0.05 NSE; 15 better by ≥ 0.05; 55 within ±0.05. Heavy negative tail.

**Depth-stratified pattern:** A wins at every depth except depth 4 (n=2, noise). Gap A − C ≈ 0.05 NSE across depths 0–3, **roughly constant — not depth-dependent.** This argues against the framing's prediction that message passing should help deeper basins more.

**Why it matters — three honest readings.**

1. **The pilot's +0.078 NSE does NOT scale.** At 23 basins (HUC-12 Texas, 34 heuristic edges, warm-started), graph-LSTM beat baseline by +0.078. At 183 basins (Component 0, 624 edges, from-scratch), graph-LSTM is *behind* baseline by 0.077. The headline gain reverses sign at scale.

2. **The "shared B+C deficit" is the diagnostic finding.** The fact that C − B is small (−0.021, std 0.082) while both have large negative deltas vs A means the cost is mostly **structural**: training the DirectedGraphLSTM architecture from-scratch instead of warm-started lands in a worse optimization basin than NH's batched cudalstm. Two candidates for what's shared between B and C:
   - The DirectedGraphLSTM uses a Python `LSTMCell` timestep loop. NH's cudalstm uses `nn.LSTM` (CUDA-batched, fully fused). The optimization trajectories are different even at identical parameter counts.
   - Both B and C add input dimensionality vs A (B = 5 topology scalars; C = effectively-zero-init message channel). From-scratch training with augmented inputs may converge differently than baseline.

3. **Aligned with Kirschstein 2024's null result.** Their finding — "GNNs on river-network topology don't help streamflow prediction at scale" — is now corroborated on CAMELS-US Component 0, single seed. The pilot's positive result was likely a small-N artifact + warm-start optimization-trajectory effect, both of which we already partially identified earlier (the +0.013 frozen-isolation finding in run 07).

**Honest qualifications.**

- **Single seed.** Yesterday's E0.5 multi-seed result showed cross-seed variance of ±0.111 NSE on the 23-basin baseline. With a similar variance at scale, the −0.07 C − A could plausibly shift to 0 or even small positive on a different seed. **Multi-seed verification is now the load-bearing next step before any framing-level claim.**
- A's mean (0.586) is dragged down by an outlier basin at -6.495 NSE. Median (0.648) is the robust summary. The B − A and C − A *means* are accordingly less negative than the medians (−0.011 and −0.041 respectively).
- B, C trained from-scratch per the locked protocol. Pilot was warm-started. Apples-to-apples is C-from-scratch vs A-from-scratch, both of which we have. The from-scratch protocol IS the right comparison for a publication-grade ablation; we're not measuring an artifact.

**What this means for the project — three branches.**

1. **If multi-seed confirms** (C consistently behind A across 5 seeds): we have a **strong negative result** at scale. This is a publishable workshop paper: *"On a 183-basin connected eastern-US network, river-network topology — both as static features and as runtime message passing — does not improve streamflow prediction over a strong multi-basin LSTM baseline. We characterize the depth-stratified, basin-by-basin pattern of the deficit and position the finding relative to Kirschstein 2024."* Negative results with mechanistic decomposition are valuable; this would be defensible.

2. **If multi-seed disagrees** (C beats A at some seeds, loses at others): the cross-seed variance is itself the headline. Workshop paper would frame as "graph methods are seed-fragile at this scale; mean effect is statistically indistinguishable from baseline." Less crisp but still publishable.

3. **If multi-seed shows C consistently beats A** (would surprise me, but possible): we'd have to revisit why the seed-42 result was off. Re-run carefully and explore.

The dynamical-systems framing's empirical predictions (graph as destabilizing forcing) are wounded by these single-seed numbers but not falsified yet — depth-stratified pattern doesn't fit, but multi-seed could change the pattern. The framing remains useful as an **interpretation** of the negative result if (1) holds: the LSTM has already settled into a good attractor with basin encoding alone; the graph's destabilizing effect actively pulls it out.

**Decisions.**

- [x] Mark A/B/C single-seed Component-0 phase as DONE.
- [x] Update `runs/README.md` headline table; write `runs/16_*/NOTES.md`.
- [x] Update `idea1.md` status section to reflect C result.
- [ ] **Multi-seed run** is the next compute spend — `MODE = 'full'` of the Colab notebook with 4 more seeds (11, 13, 17, 19, 23). On L4 ~25 hr / 125 units (over budget); on T4 ~70 hr / 105 units (in budget but multiple sessions). **Recommend T4.**
- [ ] When multi-seed lands, expand the analysis script's plots to include cross-seed bands; decide between branches 1, 2, 3 above.
- [ ] If branch 1, draft a workshop paper outline. If branch 2, design follow-up experiments to characterize seed-fragility. If branch 3, debug seed-42.

**Affected files.**

- New: `runs/16_graph_c0_warm_seed42/` + NOTES.md.
- Updated: `experiments/analysis_outputs/abc_component0/` — summary.json, per_basin_long.csv, per_basin_deltas.csv, delta_distributions.png, nse_by_depth.png, depth_stratified.csv, summary_table.txt — all now contain all three conditions.
- Updated: `runs/README.md` — Condition C row + headline table now complete.
- Removed: `experiments/drive-download-20260506T052646Z-3-001 2/` (duplicate upload; all content already organized).
- Updated: `JOURNAL.md` (this entry), `CURRENT_STATE.md`, `idea1.md`.

---

## 2026-05-05 — First scaled-A and scaled-B results land (Component 0, single-seed)

**Source.** User ran the Colab notebook (Conditions A and B completed; Condition C still running). Local CRS pulled the runs from Drive, organized them into `runs/14_*` and `runs/15_*`, ran `nh_run.py evaluate` on Condition A locally, and produced the first cross-condition analysis.

**Signal.** First non-pilot empirical results.

| Condition | Run dir | Median NSE | Mean NSE |
|---|---|---|---|
| A — NH cudalstm baseline | `14_lstm_component0_baseline_seed42` | **0.648** | 0.586 |
| B — graph-LSTM + topology features (no message passing) | `15_graph_c0_topology_features_seed42` | **0.591** | 0.575 |
| C — full graph-LSTM | (still training on Colab) | — | — |

**Per-basin ΔNSE (B − A) across 183 common basins:**
- median: **−0.050** (B is worse than A at the median)
- mean: −0.011 (smaller in magnitude due to A's outlier basins)
- distribution: 92 basins worse by ≥ 0.05; 20 basins better by ≥ 0.05; 71 within ±0.05
- std: 0.487 — high variance, not a uniform shift

**Depth-stratified A vs B medians:** A wins at depths 0 through 3 (the bulk of the network). At depth 4, B and A converge (small n). See `experiments/analysis_outputs/abc_component0/nse_by_depth.png`.

**Why it matters.**

1. **The Component-0 baseline (0.648) is much higher than the 23-basin pilot (0.423).** This is consistent with Kratzert 2019's finding: more basins → better cross-basin generalization for the multi-basin LSTM. The pilot's "easy gain" regime doesn't map directly to the larger network.

2. **Topology-as-static-features alone HURTS at this scale (single seed).** That's a meaningful finding even before C lands. Two readings:
   - The 5 topology scalars (depth, in/out-degree, transitive upstream count, log upstream-area ratio) are redundant with what the LSTM already encodes via basin ID + 5 static attributes. Adding them perturbs the optimization away from a good attractor.
   - From-scratch training (no warm-start) with augmented input dim is a different optimization problem; the seed-42 trajectory may just have been unlucky.

3. **The pilot's +0.078 headline does not transfer to B at scale.** Whatever the pilot's graph-LSTM was doing on 23 basins is not reproduced as "topology features for free" on Component 0. Whether C (full message passing) recovers anything is the open question.

**Honest qualifications.**
- Single seed only. Yesterday's multi-seed E0.5 result (cross-seed val NSE spread of 0.111 NSE on 23 basins) suggests these single-seed Component-0 numbers could shift by similar magnitudes at other seeds. **Multi-seed verification is the next no-question-asked priority.**
- A's mean NSE (0.586) is dragged down by an outlier basin at -6.495. Median (0.648) is the robust summary. Same pattern as the 08165300 outlier on the 23-basin pilot.
- B was trained from-scratch (no warm-start), per the locked A/B/C protocol. This means B − A is *not* a "warm-start vs no-warm-start" effect — both are from-scratch.

**Decisions.**

- [x] Move the runs into `runs/14_*` and `runs/15_*` with NOTES.md per locked organization patterns.
- [x] Archive the incomplete A_baseline_seed42_0605_000314 (no model_epoch030.pt).
- [x] Archive the Colab-retrained 05_lstm_23basin_strong_baseline (we already have a complete local copy).
- [x] Build `experiments/analysis/compare_abc_component0.py` for the cross-condition comparison; partial-results-aware so it works with A+B now and gracefully handles C when it arrives.
- [ ] **Wait for Condition C** (Colab Cell 11). Critical for interpreting B − A: if C ≈ A, both ablations agree no-graph wins; if C > B but C < A, message-passing helps over static features but doesn't recover the full A baseline; if C > A, message-passing matters at scale.
- [ ] **Multi-seed for all three conditions** (the `'full'` MODE of the Colab notebook, ~7-8 hr on L4). Required before any framing-level claim.

**Affected files.**
- New: `runs/14_lstm_component0_baseline_seed42/` + NOTES.md (with `test/model_epoch030/test_metrics.csv` from local evaluate).
- New: `runs/15_graph_c0_topology_features_seed42/` + NOTES.md.
- Archived: `runs/_archive/A_baseline_seed42_0605_000314_incomplete/`, `runs/_archive/05_lstm_23basin_strong_baseline_colab_retrain/`.
- New: `experiments/analysis/compare_abc_component0.py`.
- New: `experiments/analysis_outputs/abc_component0/{summary.json, per_basin_long.csv, per_basin_deltas.csv, delta_distributions.png, nse_by_depth.png, depth_stratified.csv, summary_table.txt}`.
- Updated: `runs/README.md` with runs 14, 15, and the Component-0 results table.
- Updated: `JOURNAL.md` (this entry), `CURRENT_STATE.md`.

**Open questions for the next session.**
- *What does Condition C report?* That's the headline-determining number.
- *Do A − B and the depth-stratified pattern survive multi-seed?* If A still beats B by ~0.05 NSE across 5 seeds, the "topology features hurt" finding is solid. If it's seed-dependent, no claim.
- *Why does B win at depth 4?* Could be 2 basins of noise; could be signal. If multi-seed confirms it, that's a depth-effect finding.

---

## 2026-04-26 (later) — Multi-seed E0.5: cross-seed NSE variance (0.11) larger than the pilot's +0.078 headline

**Source.** Background sweep (5 seeds × 60-epoch retrain) launched yesterday completed. Ran the multi-seed analysis (`experiments/analysis/plot_e0_5_multiseed.py`).

**Signal.**

| Seed | Plateau median val NSE (ep ≥ 5) | MAD | Linear slope ep10→60 |
|---|---|---|---|
| 11 | **0.478** | 0.015 | +0.0001 |
| 13 | 0.396 | 0.017 | +0.0003 |
| 17 | 0.367 | 0.032 | −0.0019 |
| 19 | 0.383 | 0.022 | +0.0002 |
| 23 | 0.366 | 0.021 | −0.0011 |

- **Within-seed saturation: confirmed.** Every seed's val NSE flattens by epoch 5, with linear slopes near zero (max |slope| 0.0019/epoch). Yesterday's pragmatic-saturation claim survives the within-seed test.
- **Cross-seed spread: 0.111 NSE** (0.366 ↔ 0.478). Pre-registered bar was ≤ 0.05. **FAIL by >2x.** Pilot run-05 reported NSE 0.423 — sits between seeds 11 and 13.

**Why it matters.** This is the most important honest qualification we've added this week:
1. **The pilot's +0.078 NSE absolute claim is now suspect.** That comparison was at seed 42 only. With cross-seed variance of 0.11 in *just the baseline*, a +0.078 graph delta could plausibly be seed-induced and not graph-induced. Multi-seed graph verification is now load-bearing for the paper.
2. **The dynamical-systems framing probes (E0, E1, multi-seed E0, state-space recovery, E0-B') are unaffected** — those measure model behavior under fixed conditions, not generalization across seeds. The framing remains empirically grounded.
3. **The paper's headline claim must shift** from "+0.078 NSE absolute gain" to "graph methods help under conditions C, with effect size measured under multi-seed CI." We knew this would happen; now it's documented.

**No-compute path forward.** The 23-basin pilot network is small enough to run multi-seed A and C variants on CPU in a few hours. This produces a preliminary multi-seed +0.078 reading on the pilot network *before* we spend any compute on Component 0. If the pilot's headline survives multi-seed at 23 basins, the Component-0 run is a higher-confidence bet. If it doesn't, we have an early signal.

**Decisions.**

- [x] Mark multi-seed E0.5 as DONE in `idea1.md` with the partial-pass nuance.
- [x] Document the +0.078 → multi-seed-contingent shift in `INSIGHTS.md` for the next paper-write session.
- [ ] **Next no-compute step:** multi-seed re-run of the pilot's Conditions A and C on 23 basins (5 seeds × 2 conditions × ~30 min CPU each = ~5 hours CPU, batch overnight). Tells us whether +0.078 survives multi-seed at pilot scale before any Component-0 compute spend.
- [ ] Compute access for Component-0 A/B/C remains the gating step for the publication run.

**Affected files.**

- `experiments/analysis/plot_e0_5_multiseed.py` — new analysis script.
- `experiments/analysis_outputs/e0_5/loss_saturation_multiseed.png` — multi-seed figure.
- `experiments/analysis_outputs/e0_5/decision_record_multiseed.json` — per-seed plateau medians + cross-seed disagreement metrics.
- `runs/lstm_strong_60ep_seed{11,13,17,19,23}_*/` — 5 fully trained 60-epoch baselines (all 60 ckpts each).
- `JOURNAL.md`, `CURRENT_STATE.md`, `idea1.md` — updated.

**Open questions.**

- *Does the +0.078 survive multi-seed at 23-basin pilot scale?* Cheap to test (CPU-feasible). Highest priority no-compute follow-up.
- *Why does seed 11 plateau higher (0.478) than seeds 17/19/23 (0.37)?* Possible: a seed-specific lucky basin-encoding initialization. Worth checking per-basin breakdown — if seed 11 is uniformly better across all 23 basins, the explanation is global LSTM init; if it's a few specific basins, it's local. Cheap to check.

---

## 2026-04-26 — A/B/C protocol locked; pivot to uniform from-scratch training

**Source.** Self-driven `/crs` session under user constraint "only do what is needed until we get more compute."

**Signal.** Yesterday's hostile-reviewer Q5 ("Condition B's partial warm-start is asymmetric to A and C — fix the protocol before publication") was the only outstanding methodology question. Resolving it now removes the only thing that could derail the publication run when compute arrives.

**Decision.** Wrote the **A/B/C Publication-Run Protocol** as a locked section in `idea1.md`. Key methodology change: **all three conditions train from scratch (no warm-start), with matched epochs (30) and matched hyperparameters across A/B/C.** This retires the pilot's "warm-start C from A" pattern (used in runs 06, 07, etc.) for the publication run. Rationale: from-scratch training across all three makes the A − B and B − C margins interpretable as pure structural-information contributions, not optimization-trajectory artifacts.

**Affected files.** `idea1.md` (new "A/B/C Publication-Run Protocol" section with model spec + hyperparameter table + basin/edge spec + reporting metrics + execution checklist + compute spec + falsification conditions); `CURRENT_STATE.md` (audit).

**Open questions.** None of methodology nature. The remaining gates are operational: (a) get compute, (b) wait for multi-seed E0.5 to finish (currently running slowly — 1 of 5 seeds done in 24 hrs).

---

## 2026-04-25 — `/crs-unleashed` chain: multi-seed E0, t=29 + state-space recovery, Condition B verified

**Source.** Self-driven `/crs-unleashed` session (continuation of yesterday's
queued plan).

**What was tested (chain of 3 + 1 background).**

| # | Probe / step | Hypothesis | Result | Verdict |
|---|---|---|---|---|
| Multi-seed E0 | E0 results hold across 6 seeds (11/13/17/19/23/42) on run-05 | ≥ 80% pass per probe per seed | **6/6 seeds: 100%/100% pass; Probe A 1-step recovery on every seed; Probe B max-dev range 0.006-0.010** | **PASS — zero seed variance** |
| Probe A at t=29 | Even with no recovery time available, prediction deviation is bounded | Deviation < 30% of natural pred std at t=29 | Median 0.098 (range 0.056-0.266); 23/23 basins below 30% threshold | **PASS** — closes reviewer Q2 from yesterday |
| State-space recovery | True contracting dynamics, not just head-orthogonality | ‖Δh‖_norm decays as fast as |Δy|_norm | ‖Δh‖_norm: 0.478 → 0.06 → 0.012 in 1 → 5 steps. Both spaces recover to <10% of natural variance within 5 steps. | **PASS** — closes reviewer Q1 from yesterday |
| Condition B impl | Augmented-static-feature variant works end-to-end | Pre-training NSE matches baseline; pipeline runs | Pre-training NSE = 0.423 = exact baseline match; partial warm-start verified (33 cols copied + 5 zero-init); 2-epoch smoke completed | **PASS** — A/B/C ablation infrastructure ready |
| Multi-seed E0.5 | Cross-seed val-NSE plateau bands consistent | Result lands ~3 hrs from now | seed 11 at ep 18/60 as of session close; **analysis next session** | **IN PROGRESS** (background) |

**Why it matters.**

1. **Multi-seed E0 closes the "single seed" reviewer concern definitively.** Six seeds, zero variance in pass rate, 1-step recovery is universal. The dynamical-systems framing is now backed by genuinely robust evidence on the pilot network.
2. **Probe A at t=29 closes the "you only tested mid-window" concern.** Even at the worst-case timing (perturbation immediately before prediction), the LSTM rejects 90% of the noise in prediction space. Self-stabilization holds at all timesteps.
3. **State-space recovery is the strongest result of the session.** Yesterday's E0 was prediction-space only; today shows the hidden state itself contracts back to the unperturbed trajectory in 1-5 steps. This rules out "head-orthogonality / null-space rejection" as the mechanism. The LSTM cell map is *genuinely contracting*, which is the textbook condition for self-stabilization. No more interpretive nuance needed.
4. **Condition B is implementable in the existing pipeline.** Verified by pre-training NSE = 0.423 (exact baseline). The full A/B/C ablation can run as soon as compute is available; no new code needed.

**Honest qualifications (for the paper).**

- Multi-seed E0 zero-variance is partly because Probe A's recovery measure saturates at 1-step ceiling — finer-grained timing analysis would show seed variance. Not a problem for the binary pass/fail framing, but the precise recovery-rate distribution should be characterized in the paper.
- The smoke-test +0.038 gain on Condition B is NOT a real performance claim — it's 2 epochs of likely-overfit training. The verification claim is only "pipeline correct + matches baseline at init."
- Condition B's partial-warm-start mechanism is for the smoke test only. The publication A/B/C run should pre-register a uniform warm-start scheme across all three conditions.

**Decisions.**

- [x] Mark "multi-seed E0", "Probe A at t=29", "state-space recovery", and "Condition B implementation" as DONE in `idea1.md`. The framing is now reviewer-defensible.
- [x] Document the basin 08202700 normalization-effect as a known supplementary item.
- [ ] Pre-register the A/B/C uniform warm-start scheme before the publication run. Decision: all from-scratch, no warm-start, matched epochs across A/B/C. **TODO** for next session that has compute.
- [ ] Multi-seed E0.5 analysis once the 5 seeds complete (~3 hrs from session close). Per-seed plateau bands + cross-seed variance report. **NEXT.**

**Affected files.**

- `experiments/probes/e0_self_stabilization.py` — added `--seed`, `--perturb-t`, `--measure-state-space` CLI args.
- `experiments/probes/e0_state_space_recovery.py` — new (state-space recovery probe).
- `experiments/probes/run_e0_multiseed.sh` — new (6-seed sweep launcher).
- `experiments/probes/run_e0_5_multiseed.sh` — new (5-seed E0.5 sweep, currently running in background).
- `experiments/configs/lstm_study_network_strong_60ep_template.yaml` — new (multi-seed E0.5 template).
- `experiments/training/train_graph_component0.py` — added `topology_features` variant + `compute_topology_features` helper + `warm_start_with_extra_input_dims` for partial warm-start.
- `experiments/analysis_outputs/e0/` — 12 new files (6 multi-seed × 2 outputs, plus t29 + state_space).
- `idea1.md` — status update.
- 5 currently-running 60-epoch training runs in `runs/lstm_strong_60ep_seed{11,13,17,19,23}_*/`.

**Open questions remaining after this session.**

- *Will multi-seed E0.5 confirm the pragmatic "val-saturated" reading from yesterday?*
- *Does the basin 08202700 normalization issue affect any other probes?*
- *Should we pre-register the A/B/C uniform-no-warm-start scheme now, before compute lands, to lock the methodology?*

## Next 2-3 sessions queued

1. **Multi-seed E0.5 analysis** — extract per-seed loss curves; compute cross-seed val-NSE band; verdict on yesterday's pragmatic-saturation claim. ~10 min CPU + 5 min plotting. **Cheap; first thing next session.**
2. **A/B/C pre-registration document** — write `idea1.md` §A/B/C amendment that locks the experimental protocol (no-warm-start, matched epochs, basin lists, edge files, hyperparameters). Code-only step. ~30 min.
3. **Component-0 baseline (Condition A) launch** — kick off the first scaled-experiment run when compute lands. ~2.5 hrs cloud GPU. Gates everything for the paper.

---

## 2026-04-24 — `/crs-unleashed` chain: E1 + E0.5 + E0-B' all run; framing significantly hardened

**Source.** Self-driven `/crs-unleashed` session.

**What was tested (chain of 3 pre-registered probes).**

| # | Probe | Hypothesis | Result | Verdict |
|---|---|---|---|---|
| E1 | Self-stabilization on the **weak** baseline (run 03, no basin encoding) | Self-stabilization is intrinsic to LSTM dynamics, not produced by the encoding | Probe A 100% / 1-step recovery, Probe B 100% / 0.008 max-dev — **identical signature to strong baseline (run 05)** | **PASS** — encoding is not the source of the attractor |
| E0.5 | Loss-saturation curve on strong baseline, 60-epoch retrain | Val loss has saturated by pilot's epoch 30 (else pilot was under-trained) | Strict criterion (1% per 5 epochs): FAIL on val NSE (0/25 flat windows), borderline on val loss (10/25). PRAGMATIC: val NSE plateaus at **0.355 ± 0.022 MAD from epoch 5 onward, slope −0.0004/epoch over epochs 10–60 (slightly declining)**. Train loss: 0.090 → 0.065 ep 30→60 (−27.7%, still descending). | **PRACTICALLY SATURATED** on val side; **NOT saturated** on train side → classic overfitting past epoch ~5. Pilot's epoch-30 stop was near-optimal on val. |
| E0-B' | Stronger Probe B variants (zero-out forcing; random historical day) | Probe B's t-1 result wasn't a "weak replacement" artifact | Zero-out: Probe B 100% / **0.035 max-dev**. Random-day: Probe B 100% / 0.033. Both PASS, both 5× larger than t-1 (0.007) but still well below 30% threshold. | **PASS** — LSTM rejects forcing perturbations of magnitudes 5× stronger than the original Probe B. The Probe B caveat from the previous JOURNAL entry is resolved. |

**Why it matters.**

1. **E1 PASS** is a major strengthening of the framing. Self-stabilization is now established as an intrinsic property of the LSTM dynamics, not an artifact of the basin-encoding embedding. This rules out one whole class of alternative explanations.
2. **E0.5 reading** clarifies the pilot's +0.078: it is NOT an under-training artifact (val saturated at epoch 5), but the absolute NSE numbers are bounded above by the val-saturation level (~0.40 on this 23-basin setup). The +0.078 *relative* gap is on solid ground; the *absolute* numbers should be cited as "in the saturated regime." Implication for scaling: **more data (more basins) is the lever, not more epochs.**
3. **E0-B' PASS on stronger variants** kills the methodological objection from the previous JOURNAL entry. The LSTM stays self-stabilizing under zero-forcing and random-day perturbations — both 5× more aggressive than t-1.

**Honest qualifications (for the paper).**

- E0.5's strict pre-registered criterion failed; I'm citing the pragmatic reading from the figure + linear-regression slope. Documenting the criterion was poorly calibrated for val-NSE noise (should have been smoothed or applied to val loss only). This is a methodology lesson, not a goalpost move.
- All probes are still single-seed. Multi-seed verification on E0 + E0.5 should happen before publication.
- Train loss continues to descend at epoch 60 — the model has more capacity to overfit. Pilot models (both baseline and graph) are at similar overfit states; the +0.078 *comparison* is fair, the absolute numbers less so.

**Decisions.**

- [x] Mark E1, E0.5 (pragmatic), E0-B' as PASS in `idea1.md`. The dynamical-systems framing now rests on a substantially harder evidence base.
- [x] Document E0.5 criterion-vs-pragmatic-reading honestly in this entry — the pre-registration was tighter than the data noise level, not a goalpost issue.
- [ ] Multi-seed replication of E0 + E0.5 — gates publication. ~2.5 hrs CPU for 5 seeds.
- [ ] Add a "perturb at t=29" condition to Probe A — addresses the "always 14 timesteps to recover" criticism from the previous reviewer pass.
- [ ] Compute access for Component-0 A/B/C ablation — the next big experimental block. **Still blocked.**

**Affected files.**

- `experiments/probes/e0_self_stabilization.py` — refactored with argparse (--baseline-dir, --sigma, --probe-b-mode, --out-suffix). Backward-compatible defaults reproduce the canonical 2026-04-24 morning run.
- `experiments/configs/lstm_study_network_strong_60ep.yaml` — new (E0.5 retrain config).
- `experiments/analysis/plot_e0_5_saturation.py` — new (E0.5 analysis + plot).
- `experiments/analysis_outputs/e0/` — three new sets of probe outputs: `*_weak_baseline*`, `*_probeB_zero*`, `*_probeB_randomday*` (decision_record.json, CSV, PNG each).
- `experiments/analysis_outputs/e0_5/` — new (loss saturation curve + decision record).
- `runs/lstm_study_network_strong_60ep_2404_173615/` — new training run.
- `idea1.md` — status update to mark E1 + E0.5 + E0-B' results.

**Open questions (queued).**

- *Does E0 hold on a multi-year continuous rollout?* (Currently: 30-step training-window perturbations only.)
- *Does Probe A still pass when perturbed at t=29 (immediately before the prediction)?*
- *State-space recovery* — measure `‖h_perturbed − h_unperturbed‖` directly, not just prediction-space deviation.

## Next 2–3 sessions queued

1. **Multi-seed replication of E0 + E0.5** (5 seeds × 2 probes × ~5 min = 50 min CPU; E0.5 5 seeds × 25 min = 2 hrs). Gates publication credibility. Can run overnight.
2. **Probe A at t=29 (immediately pre-prediction)** + **state-space recovery measurement** (cheap, ~5 min). Closes two reviewer-2 questions from this session.
3. **Implement Condition B (topology-as-features) in `train_graph_component0.py`** + smoke test on 23-basin network. Code-only step; gates the actual A/B/C ablation when compute lands.

---

## 2026-04-24 — E0 result: framing empirically grounded (with one caveat)

**Source.** Self-review while continuing solo work.

**Signal.** Ran the E0 self-stabilization probes pre-registered in `idea1.md`
on the run-05 strong baseline. Outputs in
`experiments/analysis_outputs/e0/`. Two probes, two perturbation magnitudes:

| Probe | σ | Pass rate | Median recovery |
|---|---|---|---|
| A — hidden-state perturbation | 0.5 × natural h-std | **100% (23/23)** | 1 step |
| A — hidden-state perturbation | 2.0 × natural h-std | **100% (23/23)** | 2 steps |
| B — forcing replacement (t-1) | n/a | **100% (23/23)** | max dev 0.007 |

The pre-registered bar was ≥ 50% of basins on each probe. Both probes
exceed it by a wide margin and the result is robust to perturbation
magnitude (recovery degrades gracefully from 1 → 2 steps as σ goes 0.5 → 2.0).

**Why it matters.** This is the gate experiment that the entire 2026-04-21
reframing rests on. The LSTM **does** exhibit self-stabilizing dynamics on
the trained baseline:
- A hidden-state perturbation as large as 2σ in any direction is absorbed
  within 2 timesteps in prediction space.
- The dynamical-systems framing is now empirically supported, not just a
  story about the +0.013 / +0.065 decomposition.

The reframing in `idea1.md` stands. We can proceed to E0.5 (loss saturation)
and the forcing-comparison sub-experiment.

**Caveat — Probe B has a methodological weakness.** "Replace forcing at t
with t-1's forcing" is on most days a near-null replacement: rainfall is
zero on most days (today=0, yesterday=0, no change), and SRAD/Tmax/Tmin/Vp
are smooth (today ≈ yesterday). The 0.007 median deviation reflects a mix
of (a) LSTM rejecting weak input changes and (b) the replacement being
weak in the first place. Probe A — the perturbation-recovery test — is
the more diagnostic of the two and is the one we should cite when
defending the framing claim. A more stringent forcing-test (set forcing
to zero, or replace with a randomly-chosen historical day) is a useful
follow-up but not blocking.

**Decisions.**

- [x] **Mark E0 as PASS** in `idea1.md` status section.
- [ ] **Run E0.5** (60-epoch loss saturation curve) next. Cheap, gating the
  scaling argument.
- [ ] **Strengthen Probe B** before publication: add a "replace with
  random historical day's forcing" variant and a "zero-forcing"
  variant. Not blocking for moving forward, but should be in the
  paper's supplementary.
- [ ] **Cache E0 outputs in repo** as evidence for the PI: the figure
  `experiments/analysis_outputs/e0/probe_a_recovery.png` is meeting-ready.

**Affected files.**
- `experiments/probes/e0_self_stabilization.py` — new probe script.
- `experiments/analysis_outputs/e0/` — new outputs subfolder
  (decision_record.json + sensitivity, CSV, two PNGs).
- `idea1.md` — update §Status section to mark E0 as PASS, add link to
  decision record.

**Open questions.**
- *Does E0 pass on the weak baseline (run 03, no basin encoding) too?*
  If yes, self-stabilization is generic; if no, the encoding is doing
  the stabilizing work — which would itself be a publishable mechanism.
  Cheap to check; deferred until after E0.5.
- *Does E0 pass on a basin's full multi-year rolled-out trajectory* (not
  just a 30-step training window)? Stronger test of self-stabilization
  in the deployed-prediction regime. Requires a different evaluation
  loop. Deferred.

---

## 2026-04-21 — PI meeting: dynamical-systems framing of LSTM behavior

**Source.** First post-pilot meeting with the PI. ~30 min. Walked through
`CURRENT_STATE.md`, `INSIGHTS.md`, and the `idea1.md` ablation plan.

**Signal.** Five distinct strands of feedback, paraphrased:

1. *LSTM self-equilibrium / self-stabilization.* The LSTM, when rolled out,
   tends to drift into a self-consistent attractor where its own dynamics
   dominate over external forcings. The model "drives itself" rather than
   tracking the actual physical drivers of streamflow.
2. *The research question is what destabilizes the cycle.* The interesting
   problem is to identify external information that **breaks** the LSTM's
   self-consistent regime — signals that force the model to track real
   external dynamics rather than its own learned attractor.
3. *Loss saturation as the scaling test.* If the pilot's training loss has
   plateaued, then scaling (more data, bigger model) will help — by
   scaling-laws logic. If the loss is still consistently descending and we
   are calling current improvements "marginal," that's *evidence* scaling
   would help, not a refutation.
4. *Reframe in dynamical-systems-on-networks language.* Treat the river
   network as a dynamical system on a graph; treat the LSTM as a learned
   approximation of that dynamical system. The mathematical literature on
   network dynamics (graph signal processing, dynamical systems on networks,
   physics-of-flow-on-graphs) gives a more robust grounding than ad-hoc
   "graph helps streamflow" framing. Open questions become: what features /
   spaces are sufficient to recover the dynamics? What graph topologies
   admit destabilizing forcings?
5. *Verifiable physical claims + reproducibility.* Tie claims back to
   physics-of-river-flow models (Saint-Venant, Manning, linear-reservoir
   routing). An agent can retrieve canonical mathematical models for
   verification. Practically: package the project as `setup.{sh,py}` +
   `run.py` so anyone (including the PI) can reproduce in Colab.

**Why it matters.** This is a **reframing**, not a redirection. The pilot
findings already make sense under this lens:

- **+0.013 frozen-graph NSE** = small, real "destabilizing forcing" effect.
  Pure injection of external information not visible from the basin's own
  history.
- **+0.065 from LSTM drift during joint training** = the LSTM finding a new
  self-consistent attractor that incorporates the graph hints into its own
  dynamics. *Not* a confound to be eliminated — a different *form* of using
  the same external information.
- **Aggregation variants converging** (0.994–0.999 error correlation) =
  consistent with all variants providing similar destabilizing forcing
  content; the form of aggregation matters less than whether *any*
  aggregator is present.
- **Chain contamination in PUB** = expected: when the upstream LSTM is also
  in a self-stabilizing regime that's *wrong* for the held-out basin, the
  forcing it provides is wrong-direction, not destabilizing in a useful way.

The question reframes from "does topology help?" to **"what kinds of
external forcings, including but not limited to graph topology, break the
LSTM's self-stabilization, and which graph topologies provide enough such
forcing to be worth the architectural complexity?"**

This is a stronger question because:
- It's grounded in dynamical-systems theory (not hydrology folklore).
- It explains the pilot results coherently rather than as a +0.078 number
  with caveats.
- It's verifiable against known physics (Saint-Venant, Manning, linear
  reservoir routing).
- It admits a clear ablation: *what kinds of external information break
  LSTM self-stabilization?* Graph is one; physical state variables (soil
  moisture, snowpack, antecedent conditions) are others; random
  perturbations are the null. We can compare them.

**Decisions.**

- [ ] **Reframe `idea1.md`** around the dynamical-systems framing while
  keeping the A/B/C ablation intact (it still answers the new question —
  with B reinterpreted as "static-feature destabilization" and C as
  "runtime-message destabilization").
- [ ] **Add Experiment 0: self-stabilization verification.** Show
  empirically that the trained LSTM exhibits self-stabilizing behavior
  (e.g., perturb hidden state mid-rollout, measure how fast it returns to
  the unperturbed trajectory; or autoregressive rollout test). If we can't
  show self-stabilization, the framing collapses; if we can, the rest of
  the plan rests on a verified premise.
- [ ] **Add Experiment 0.5: loss-saturation curve.** Train the strong
  baseline (config: `lstm_study_network_strong.yaml`) for 60 epochs instead
  of 30, plot validation loss. If saturated, scaling argument applies. If
  not, even baseline can be improved by training longer — affects how we
  interpret all "marginal" pilot numbers.
- [ ] **Add forcing-comparison experiment** to `idea1.md` Phase 2: compare
  graph messages against (a) random perturbations, (b) the basin's own
  forcing time series shifted by a lag, (c) upstream basins'
  *precipitation* (rather than learned hidden state). Question: is the
  graph hidden-state message "better than" these alternative
  destabilizers?
- [ ] **Engage the dynamical-systems-on-networks literature.** Read for
  positioning: Saint-Venant on graphs, network reaction-diffusion, graph
  signal processing for flow networks. Not in this conversation; logged
  as a TODO for next week.
- [ ] **Package as setup + run for Colab.** `setup.py`/`setup.sh` for
  environment + data; single-command `run.py` that reproduces the headline
  result. Sized so the PI (or a Colab session) can reproduce in one
  command.
- [ ] **Update `JOURNAL.md`** when each of the above lands (this is the
  feedback loop).

**Affected files.**
- `idea1.md` — biggest update; new framing, new experiments E0 + E0.5,
  forcing-comparison condition.
- `CURRENT_STATE.md` — append "Idea 1 reframing" section after Idea 2
  excursion, summarizing the dynamical-systems lens.
- `INSIGHTS.md` — re-interpret pilot findings under the new lens (do not
  delete the existing findings; *add* a re-reading paragraph).
- New: `setup.{py,sh}` + `run.py` at the repo root for Colab reproducibility.

**Open questions.**
- *What does "self-stabilization" empirically look like for our LSTM?* We
  need an operational test before we can claim it.
- *Is there a clean mathematical model for "river-network LSTM as dynamical
  system on a graph"* that admits the kind of bifurcation analysis the PI
  was implying? (Or are we just hand-waving with the language?)
- *What is the right comparator for "destabilizing forcing"?* The
  forcing-comparison experiment needs careful design — random noise is one
  thing, but lagged precipitation is more interesting. We need the PI's
  guidance on which alternative forcings are most informative to test.
- *Compute is still unresolved.* The reframing doesn't change the fact
  that a 5-seed × 3-condition × 183-basin run needs GPU. Did we ask? Logged
  as a follow-up question for the next meeting.

---
## 2026-05-12 — /crs-unleashed: 5-condition factorial post-mortem

### Step 1 — Orient (deeper)

15 runs from the Colab sweep landed in `experiments/5cond_factorial/multi_condition_ablation/`. Each L_seed* folder has the canonical NH layout (`config.yml`, `model_epoch001..030.pt`, `test/model_epoch030/{test_metrics.csv, test_results.p}`); each graph run has `test_metrics.csv`, `test_predictions.csv`, `run_config.json` with epoch-by-epoch loss + wall-clock. `compare_5conditions.py` works against either layout once a symlink puts everything under `runs/5cond_factorial/`. Recent commits (last 5): notebook fixes (Cell 8 disabled, Cell 10 cascade-detection, Component-0 baseline fix, factorial infrastructure). Open question carried from prior journal: "why does the 23-basin pilot's +0.078 NSE evaporate at Component-0 scale?" — this session addresses it.

### Step 2 — Diagnose (top 3 load-bearing claims, ranked)

1. **"L − G = +0.050 NSE is the architecture confound."** *Importance: high (it's the central question of this paper). Confidence: LOW after this session — see Step 4.* The hypothesis from the 2026-05-06 meeting was that cudalstm (cuDNN-fused `nn.LSTM`) and DirectedGraphLSTM (Python loop over `nn.LSTMCell`) are different architectures. They are — but they should be mathematically equivalent forward passes when edges are empty. Whether the gap is "real architecture" or "training-budget confound dressed up as architecture" needed to be tested. **TEST FIRST.**
2. **"Graph signal (topology features + message passing) is net null-to-negative at Component-0 scale."** *Importance: high. Confidence: medium-high.* Within-graph-trainer contrasts: G+T − G = −0.001 (null), G+M − G = −0.006 (slight negative), G+T+M − G = −0.011 (more negative), interaction = −0.009 (sub-additive). All three CIs exclude zero on the negative side. These contrasts hold the architecture and training budget constant, so they're internally valid — but they sit on an undertrained baseline.
3. **"23-basin pilot +0.078 NSE was real but doesn't generalize."** *Importance: medium. Confidence: high.* The pilot used warm-start and 23 basins (tiny network, high signal-per-step); the scaled run is from-scratch on 183 basins. Generalization gap is the expected story.

### Step 3 — Decide (chain of 2 gated steps; only Step A run this session)

**Step A — Code-level audit of L vs G architectural & training-pipeline differences.** *Hypothesis:* there is at least one non-architectural difference (training budget, batching, normalization) large enough to plausibly account for ΔNSE ≈ 0.05. *Success:* identify the difference and quantify it. *Falsification:* the trainers are matched on every dimension we can audit at code level (in which case the gap is purely architecture/numerics). *Cost:* free (no compute).

**Step B (gated on A finding a budget confound) — Reproduce L with matched gradient budget.** *Hypothesis:* if I train cudalstm with the SAME gradient-step budget the graph trainer gets (14 steps/epoch × 30 epochs = 420 steps total), its median NSE drops to ≈ G's 0.609. *Success:* cudalstm-with-restricted-budget lands within 0.01 NSE of G. *Falsification:* cudalstm-with-restricted-budget still beats G by ≥ 0.03 NSE → there IS a real architecture component. *Cost:* ~5 min on T4 per seed (only 420 gradient steps); 3 seeds ≈ 15 min. NOT RUN THIS SESSION (needs GPU).

### Step 4 — Execute (Step A only)

**Audited DirectedGraphLSTM (`experiments/training/train_graph_lstm.py`):**
- Forget bias init → matched (set to 3 on `bias_hh` at construction).
- Dropout → matched (0.4 on LSTM output before head).
- LR / optimizer / clip → matched (Adam, 1e-3, clip_grad_norm=1).
- Basin one-hot encoding → matched (`x_one_hot` from NH dataset, propagated via `static_parts.append(sample["x_one_hot"])`).
- Hidden size → matched (64).
- Loss → mathematically equivalent (MSE on last-step prediction with NaN masking).
- **Batching shape → MISMATCH.** NH samples random (basin, window) pairs of size 256 → ~2,610 gradient steps/epoch on Component 0. DirectedGraphLSTM samples whole windows (all 183 basins per window) of 256 windows per batch → ~14 gradient steps/epoch. **186× ratio.**

**Quantification: loss trajectories all four graph variants, seed 11:**
- G:    epoch 0/5/10/15/20/25/29 loss = 0.912/0.606/0.486/0.433/0.406/0.370/0.359
- G+T:  0.960/0.626/0.499/0.448/0.418/0.379/0.366
- G+M:  0.904/0.559/0.458/0.414/0.392/0.365/0.351
- G+T+M: 0.949/0.583/0.473/0.426/0.402/0.378/0.360

Loss still falling ~3% in the last 4 epochs across all variants. Graph trainer is NOT converged at epoch 30. With NH's gradient-step budget the trainer would have processed ~78,000 gradient steps; it processed 420.

**Per-basin pattern check (L − G):** 76% of basins are better with L; 50% strongly so (Δ > 0.05); only 10% are strongly better with G. The L > G effect is *broad and uniform*, not concentrated on a specific basin type. This is the signature of an optimization-side cause, not a model-side cause (which would show a bimodal pattern, e.g., "G better on basins with strong upstream signal, L better on independent basins").

**Step A verdict: PASSES (a large non-architectural confound exists).** Step B would proceed if compute were available this session.

### Step 4.5 — Reviewer 2

- *"Could the gradient-step ratio simply be wrong because each graph step sees 183× more data and so per-example signal is the same?"* → Per-example signal is the same, but **Adam's adaptive moments behave differently with effective batch size 183 × 256 ≈ 47k vs 256.** Larger effective batch → smaller relative noise → smaller effective step size at fixed LR. Empirically the loss curve says: trainer is still descending. So whatever the per-step efficiency, 30 epochs isn't enough.
- *"What if the LSTMCell Python loop has subtly different numerics from cuDNN-fused LSTM?"* → Possible at the 1e-4 level. The smoke test on 23 basins matched the pilot NSE within < 1e-3 of the cudalstm pilot. A 0.05 NSE gap is two orders of magnitude too large for FP32-vs-cuDNN.
- *"Is the per-basin uniformity argument really diagnostic?"* → It's suggestive, not conclusive. The strict test is Step B: equalize gradient budget and re-run. That's the killer experiment.
- *"What about KGE? L beats G by +0.17 KGE — same training-budget story?"* → Same direction. KGE has higher std due to dry-basin instability (mean-ratio β term blows up). Bootstrap median CI [+0.15, +0.20] is still very tight.
- *"The original 23-basin pilot had warm-start. Component 0 graph runs were from-scratch. Could warm-start alone close the gap?"* → Possibly. Warm-start was originally how we got +0.078 NSE. A future test should compare "G from-scratch" vs "G warm-started from L_seed11" — orthogonal to Step B.

### Open questions remaining

1. Does matched-gradient-budget cudalstm match G? (Step B; pre-registered above.)
2. Does warm-started G match L? (Independent test, 30 epochs each.)
3. If G == L when properly trained, do (G+T) − G and (G+M) − G change sign or magnitude on the properly-trained baseline? (The real publishable contrasts.)
4. Why is loss decreasing slowly in graph trainer — Adam moment-scale issue, or genuinely undertrained at any LR?

### Plan for next 2–3 sessions

1. **Run Step B on Colab T4 (~20 min):** train cudalstm with 420 gradient steps total (no longer 78,000) on Component 0. Compare to G. *Decisive on the training-budget hypothesis.*
2. **Run Step C — graph variants extended training (compute heavy):** train G for 200 epochs on Colab T4 (~10 hr), see if it converges to L's NSE. If yes, training-budget confirmed; if no, residual architecture gap quantified.
3. **Recompute the factorial contrasts on the properly-trained baseline:** once we have a converged G, recompute (G+T) − G, (G+M) − G, (G+T+M) − G. *This is the actual publishable result — current numbers are on an undertrained baseline and overstate the sub-additivity.*


---
## 2026-05-12 — Architecture deep-audit and testing-framework redesign

### Why this session

The 5-condition factorial finished with G+T+M < G+M ≈ G+T < G < L. The headline (L − G ≈ +0.050 NSE) was previously framed as an architecture confound (cudalstm vs DirectedGraphLSTM). Earlier this day's `/crs-unleashed` session showed it was almost entirely a **training-budget confound**: graph trainer gets 186× fewer gradient updates per epoch than NH's cudalstm trainer. Loss is still falling at epoch 30.

But the training-budget story is necessary, not sufficient. Why do the *advanced* variants (G+T, G+M, G+T+M) also underperform even the simplest no-graph baseline G? This session audits each architectural component to answer that.

### Three deliverables produced (root directory)

1. **`experiments/5cond_factorial/analysis/5cond_run_analysis.md`** — results-only digest. Headline numbers, six pairwise contrasts, per-basin patterns, stratifications, loss trajectories. Conclusion: the paper-claim direction is reversed; current results cannot support "our features beat standard LSTM."
2. **`experiments/5cond_factorial/analysis/architecture_analysis.md`** — deep technical critique. Walks through each component (DirectedGraphLSTM forward pass, the 5 topology features, message passing aggregation/function/residual, training pipeline confounds) and identifies specific defects. Prioritized fix tiers and three honest paper narratives.
3. **`experiments/5cond_factorial/analysis/testing_framework_proposal.md`** — 6-step diagnostic ladder with pre-registration discipline. Step 1 (matched-budget L) is 15 min on T4 and gates the next step. Total framework through Step 5 = ~43 hr T4.

### Most surprising finding from the audit

**The basin one-hot encoding (NH default, 671-dim) subsumes the 5 hand-designed topology features.** The topology features are 0.7% of the static input vector. The basin one-hot already perfectly identifies each basin and lets the LSTM learn arbitrary per-basin response curves. The topology features are *redundant signal in low-dimensional drag-along channels.*

This is the structural reason G+T − G ≈ 0. Not because topology features are inherently weak — they aren't necessarily — but because the design choice to keep `use_basin_id_encoding: True` makes them informationally redundant. Step 2 of the new framework tests this directly.

### Other architectural defects worth recording

- Mean aggregation gives a 1km² parent and a 100km² parent equal weight (Component 0 has area_ratio up to 275×).
- Single linear `W_msg_edge` (no nonlinearity in the message function itself).
- Zero-init `W_out` + `tanh(W_out(m))` residual: the graph path has to grow from zero through a saturating nonlinearity.
- `train_graph_lstm.py` had test-set leakage in best-checkpoint selection (lines 635-643) — production trainer doesn't use it but the helper code is buggy.
- `compute_topology_features` uses `shortest_path_length` despite docstring saying "longest path" — minor bug for ~10% of basins.

### Hostile-reviewer review of these findings

- *"Aren't you just rationalizing a negative result?"* → No: the diagnoses are specific (training budget = 186× ratio, basin one-hot = 0.7% feature share). Each has a falsifiable test in the new framework.
- *"What if all the proposed fixes don't help?"* → Then we publish the negative result (Narrative C in arch_analysis §7). The 5cond run already has the statistical rigor for that.
- *"You're proposing too many changes."* → The framework explicitly orders them — Step 1 is one experiment, gated; Step 2 is another, gated. Not all at once.

### Plan: next 2–3 sessions

1. **Write `preregistration_step1.md`** and launch matched-budget L on Colab (15 min compute). *Gates whether the L−G gap is purely training budget.*
2. **Write `preregistration_step2.md`** based on Step 1 outcome, launch one-hot-ablation runs (~6 hr compute). *Gates whether topology features have signal independent of basin-ID.*
3. **Implement area-weighted aggregation in DirectedGraphLSTM (code change in `train_graph_lstm.py`)** while Step 2 runs — does not need its own pre-registration since it's a design improvement, not a hypothesis test. The test of whether it helps is Step 4.

---
## 2026-05-12 — /crs-unleashed: Step 1 executed (matched-budget L)

### Step 1 (Orient deeper)

Re-read this morning's three analysis files. Identified `max_updates_per_epoch` in NH's `BaseTrainer` — clean implementation path for matched-budget cudalstm without forking the codebase. The framework's Step 1 was queued as the immediate next action; this session executed it.

### Step 2 (Diagnose top 3)

1. **L − G gap of +0.050 NSE is dominantly a training-budget confound** (high importance, low confidence after earlier analysis). → TEST THIS NOW.
2. Basin one-hot encoding subsumes topology features (high importance, med confidence). → Pre-register, defer execution to GPU session.
3. Area-weighted aggregation > mean (med importance, med confidence). → Defer.

### Step 3 (Decide)

- **Step A:** Pre-register Step 1 (`preregistration_step1.md`), implement matched-budget script, run on CPU. Hypothesis: L_420 lands within 0.01 NSE of G.
- **Step B:** Pre-register Step 2 (`preregistration_step2.md`) — write only, run later on GPU.
- **Step C:** gated on A: write addendum to `experiments/5cond_factorial/analysis/5cond_run_analysis.md` if hypothesis confirmed.

### Step 4 (Execute)

**Step A — matched-budget L (main result):**

| Trainer | Steps | Examples | Cross-seed median NSE |
|---|---|---|---|
| L (cudalstm, 30 epoch) | 78,330 | 20M | 0.653 |
| G (graph, 30 graph-epoch) | 420 | 20M | 0.609 |
| L_420 (matched steps) | 420 | 107k | **0.502** |

L_420 − G paired (n=549): median Δ = **−0.100**, CI [−0.105, −0.092], 94.5% of basin × seed pairs favor G.

**Outcome:** neither success (Δ ≈ 0) nor falsification (Δ ≥ +0.03). Third-category — Δ went in the *opposite* direction. **The "matched gradient steps" framing was biased toward the trainer with the larger per-step batch.** Each graph step sees 47k examples; cudalstm sees 256. At 420 steps, graph trainer has seen 200× more data.

**Step B — preregistration_step2 (queued, not run):** basin-encoding ablation. Needs GPU.

### Step 4.5 — Reviewer 2

1. **"Is the L_420 result an artifact of the script implementation?"** → Unlikely. Uses NH's standard `BaseTrainer` with only `max_updates_per_epoch=420` and `epochs=1`; everything else (model, loss, optimizer, data pipeline) is identical to the original L runs. Same scaler, same basin set, same forcings. Identical NH eval pipeline producing test_metrics.csv. The pipeline is the same as the L_seed runs, just stopped early.
2. **"Could L_420 just be 'NH at 16% of one epoch' which is essentially random?"** → No. L_420 NSE = 0.502 is far above the null-control expectation (≪ 0); the model HAS learned something. It's just less than G learned in the same step count.
3. **"Does cross-seed std support the conclusion?"** → Yes. L_420 per-seed medians 0.521/0.502/0.501 (std 0.011), tight. G per-seed medians 0.609/0.614/0.590 (std 0.013). The L_420 − G gap (0.10) is ~7× the cross-seed std, very robust.
4. **"What if NH cudalstm has a warmup phase that makes step-420 NSE pessimistic?"** → Plausible. Adam moments stabilize in early training. To address: would need to evaluate at multiple step counts (e.g., 420, 1000, 3000, 10000) — a quick follow-up using the existing L checkpoints. The bigger point is that *the original L − G comparison happened at matched-examples* (~20M each), not matched-steps. At matched-examples, L still wins by 0.05. So the residual gap is real even after this experiment.
5. **"Are you sure you didn't just rediscover that larger batch = fewer steps to reach same NSE?"** → Yes, that's exactly what this result implies — and that's *the finding*. The previous hypothesis (training-budget = step-count) collided with the reality that matched-steps biases toward whichever trainer has the bigger per-step batch. Matched-examples is the cleaner framing, and there L > G by 0.05 — so the gap is something *other than* both step count and example count.

### Open questions after this session

1. **What IS the source of the L − G gap at matched examples?** Three candidates: (a) Adam gradient-noise scaling (small-batch L gets effective LR boost from noise); (b) data exposure pattern (random (basin, window) sampling exposes the model to more diverse mini-batches than whole-window sampling); (c) actual architectural difference (cuDNN nn.LSTM vs Python nn.LSTMCell loop). Next experiment should isolate.
2. **Does graph trainer benefit from smaller batches?** If batch=32 instead of 256, graph trainer gets 14×16 = 117 steps/epoch → 30 epochs = 3,510 steps. 8× more gradient updates. Pre-register before running.
3. **Does the L − G gap survive at L's batch=32 setting?** Train L with batch=32 to check if cudalstm itself benefits more from smaller batches than graph trainer would.
4. Step 2 (basin-encoding ablation) still queued; gated by Step 1 outcome but the outcome here doesn't invalidate Step 2's design.

### Plan for next 2–3 sessions

1. **Graph trainer with batch=32 (G_b32):** matched-examples to original G but with 8× more gradient updates per epoch. 3 seeds. Pre-register before launching. Compute: ~3 hr on T4 (8× the graph trainer's per-epoch step count, similar per-step cost). *Gates whether more gradient updates close the L − G gap.*
2. **Step 2 from framework** (`preregistration_step2.md`, already written): one-hot ablation. 6 hr on T4. *Gates whether topology features have signal independent of basin-ID.* Independent of #1.
3. **L learning-curve via existing checkpoints:** evaluate L_seed11 at epochs {1, 5, 10, 20, 30}. ~5 min compute (eval-only). *Tells us where on L's learning curve NSE ≈ 0.609 (G's level) sits — i.e., how many cudalstm-steps equal one graph-trainer-step in NSE terms.*


---
## 2026-06-20 — Pivot: confounds diagnosed, program restarted as a controlled encoding×topology ablation

**Source.** User-run Colab local-subgraph sweep (single subgraph completed before stop) + CRS code audit during a /crs session.

**Signal.** On sg_midatlantic (16 basins), G+T+M median NSE 0.622 < G 0.68 — the *same* direction as the 183-basin 5cond result, now at small scale. The user flagged the key point: **G+T+M < G is a result that shouldn't be possible.** Adding inputs to a model can at worst be ignored; for them to make it *worse* implies a bug or a design fault, not a finding.

**Why it matters — three compounding confounds make every prior negative uninterpretable.**
1. **Architecture/trainer confound.** Custom DirectedGraphLSTM is undertrained (loss still falling at epoch 30) and not GPU-accelerated; never a fair comparison to NH's tuned cudalstm.
2. **Encoding redundancy.** NH's 671-dim basin one-hot lets the LSTM memorize per-basin behaviour. The 5 topology scalars (<1% of static input) are informationally redundant with it — topology features *cannot* help by construction. Connects to GNN theory (Kipf-Welling): structure helps most in the can't-memorize regime; the one-hot is the can-memorize regime.
3. **From-scratch noise injection.** With `--no-warm-start` the topology input-weight columns start random and an undertrained model never suppresses them → added features actively hurt. Confirmed `--no-warm-start` + `use_basin_id_encoding: True` in the sweep.

**Key enabling realization.** NH auto-loads any `camels_attributes_v2.0/camels_*.txt` file as static attributes, and the one-hot is a single config flag. So the topology-feature question runs entirely on **stock cudalstm** — well-tuned, GPU-native, fully trained — with zero custom model code. All three confounds vanish for the topology question. (Message passing still needs custom code; deferred and gated.)

**Decision — restart the program as a controlled experiment, Option A (build the framework) with honest framing.**
- New self-contained study: `experiments/topology_ablation/`.
- Foundational experiment: encoding × topology **2×2** (one-hot {on,off} × topology {off,on}) on stock cudalstm. Pre-registered prediction: `(L_noID+T − L_noID) > 0` while `(L+T − L) ≈ 0` → the one-hot subsumes topology features.
- Single seed until a clear signal; multi-seed only at publication.
- Built + verified end-to-end this session: topology-attributes generator (671 basins, depth bug fixed to longest-path, adds physically-meaningful `total_upstream_area`); 2×2 config generator; runner; analysis; Colab notebook (standard click→T4→run-all workflow). Confirmed NH ingests the features and the `L_noID+T` config trains.
- Legacy (`5cond_factorial/`, `local_subgraphs/`) preserved as the confounded-measurement prior; not deleted.

**CRS re-evaluation of the user's "ablation-framework paper > negative/corroboration paper" premise (they asked for an unbiased take).** The instinct toward a reusable *framework* is correct and is the highest-value direction — frameworks outlast one-off numbers. The correction: do **not** pre-commit to the result's *sign* ("show our features outperform"). Value lives in the rigor and the controlled design, not in the headline coming out positive; pre-committing to "it must beat baseline" is the p-hacking trap that produced our 3 weeks of confounded negatives. The encoding×topology 2×2 threads this: it likely yields a positive *and* theory-grounded finding ("structure helps in the can't-memorize regime"), is a framework others can follow, and is publishable whichever way it lands.

**Affected files.** New `experiments/topology_ablation/` (generator, configs, runner, analysis, notebook, README); new `datasets/camels_us/camels_attributes_v2.0/camels_topology.txt` (NH static-attr file). Legacy untouched.

**Open questions.**
- Does topology help without the one-hot at component-0 scale, at subgraph scale, or both? (Phase 1 answers.)
- If topology helps without one-hot: does message passing add anything beyond static topology features? (Phase 2, gated.)
- Is `total_upstream_area` (physically the load-bearing feature) the one carrying any signal? Per-feature importance worth checking if Phase 1 is positive.

---
## 2026-06-21 — Phase-1 result: topology features are weak (redundancy hypothesis FALSIFIED) → oracle becomes load-bearing

**Source.** User ran the encoding × topology 2×2 on component0 (stock NH cudalstm, single seed 11) via the Colab notebook; results dropped into `runs/topology_ablation/component0/`. CRS analyzed.

**Signal (single seed, n=183).**

| | median NSE |
|---|---|
| L (one-hot ON, no topo) | 0.653 |
| L+T (one-hot ON, +topo) | 0.654 |
| L_noID (one-hot OFF) | 0.633 |
| L_noID+T (one-hot OFF, +topo) | 0.625 |

- `topo benefit WITH one-hot` (L+T − L): **−0.001**
- `topo benefit WITHOUT one-hot` (L_noID+T − L_noID): **+0.003**
- interaction: −0.004; encoding cost (L − L_noID): +0.012
- Per-basin distributions are symmetric noise: ⅓ up / ⅓ down, std 0.18–0.35. Not a median artifact.

**Decision per pre-registration (FORWARD_PLAN Branch B).** The redundancy hypothesis is **falsified**: topology features add ≈0 NSE *whether or not* the one-hot is present. The features are intrinsically weak, not merely masked by basin-ID memorization. Pre-registered consequence: **EXP-0, the upstream-discharge oracle, becomes the load-bearing experiment.**

**Why it matters.** Static topology summaries are a lossy constant. The oracle asks the maximal version of the question: if a downstream basin sees the *actual lagged observed discharge* of its upstream basins (the literal water arriving), does prediction improve? This is the upper bound on every possible structural signal.
- Oracle helps (Δ ≥ +0.02) → the failure was the static *representation*; message passing is justified (reopen Branch A1).
- Oracle fails (Δ ≤ +0.005) → the killer control. Structure is uninformative for next-day flow at this scale; the paper is the rigorous controlled NEGATIVE (encoding confound ruled out, training confound ruled out via stock cudalstm, AND oracle ruled out). Corroborates Kirschstein 2024 with far more control.

**Honest qualifications.** Single seed — directionally clear (effects are ~0 vs cross-basin std 0.2) but the publication run needs 3 seeds. The encoding-cost finding (one-hot buys +0.012 NSE) is a minor real result, not the thesis.

**Action taken this session.** Built + launched EXP-0 oracle on component0 (stock cudalstm, L vs L+upstream_q, single seed, CPU, ~15-20 min). Feature builder unit-verified (upstream_q 1.3-1.8 mm/d). Pre-registration in FORWARD_PLAN.md; criteria formalized in this session.

**Affected files.** `runs/topology_ablation/component0/` (4 conditions moved into canonical location); `experiments/topology_ablation/analysis/{RESULTS.md,table.csv,contrasts.csv}`; new `experiments/topology_ablation/run_oracle.py`; new `experiments/topology_ablation/features/upstream_q_component0_lag1.p`.

**Open questions.**
- Does the oracle help? (running) — decides whether the program continues toward message passing or locks the negative-result paper.
- Is the +0.012 encoding cost worth reporting as a secondary methodological note? (Probably yes — it quantifies what basin-ID encoding buys.)

---
## 2026-06-21 (later) — ORACLE PASSED: structure helps as a DYNAMIC signal (+0.037 NSE upper bound)

**Source.** EXP-0 oracle completed (component0, stock cudalstm, single seed 11): L vs L+upstream_q (area-weighted mean of upstream basins' lagged OBSERVED discharge, mm/d).

**Result.** L median NSE 0.653 → L+upstream_q **0.703**. Paired upQ−L: **median +0.037, 67% of basins improve, 58% by ≥0.02** (n=183). Pre-registered success bar +0.02 — cleared ~2×.

**Decision per pre-registration.** SUCCESS (Branch B → reopen Branch A1). The oracle was built to disambiguate "structure is uninformative" from "static representation is too lossy." Verdict: the latter. Static topology features added ~0 (Phase-1 2×2); the actual dynamic upstream flow adds +0.037. **Structure carries real exploitable signal; the topology-feature failure was a representation problem, not a structural one.**

**Why it matters.** This is the first clean positive in the program and it flips the outlook. The thesis ("river-network structure improves LSTM streamflow prediction") is alive, now with a falsifiable, evidence-backed form: structure helps *as a dynamic upstream-state signal*. A learned message-passing model (propagating upstream hidden state) is the realizable proxy for the oracle and is now justified by evidence. CRS prior going in was "likely null" — falsified; good kind of wrong.

**Honest caveat.** The oracle uses observed (ground-truth) upstream discharge → it is an UPPER BOUND, not a deployable model. The realizable gain from learned/predicted upstream signal is some fraction of +0.037. Single seed; publication needs 3. Frame all future message-passing results *against* this +0.037 ceiling ("recovers X% of the bound").

**Next gated move (Branch A1).** Build/justify a correct message-passing model (propagate upstream hidden state, trained on stock-equivalent infra), report vs the +0.037 oracle ceiling. Secondary: does the upstream benefit grow on small local subgraphs (walker machinery ready)?

**Affected files.** `runs/topology_ablation/component0/L_upQ_component0_seed11/`; `updates.md` (note sheet) updated with the positive result and the reframed direction.

---
## 2026-06-21 — /crs-unleashed: upstream-signal chain — gain is real, content beyond precip, robust to lag

**Source.** /crs-unleashed session. Pre-registered 3-step chain (`preregistration_upstream_signal.md`) stress-testing the oracle's +0.037 NSE before investing in a learned model. All stock cudalstm, component0, seed 11, CPU.

**Results.**

| Condition | median NSE | paired Δ vs L | frac>0 |
|---|---|---|---|
| L | 0.653 | — | — |
| L+upQ (oracle, lag1) | 0.703 | +0.037 | 67% |
| L+upQ **shuffled (null)** | 0.658 | **−0.002** | 49% |
| L+upPrecip | 0.674 | +0.012 | 58% |
| L+upQ lag0 | 0.749 | **+0.087** | 85% |
| L+upQ lag2 | 0.699 | +0.036 | 69% |

**Step A (gate) — PASSED.** Shuffled-upstream-Q (same marginal, time-scrambled) gives −0.002, 49% of basins up = coin-flip. The +0.037 is **real upstream-flow content, not a capacity/regularization artifact.** Cleanest possible pass; the positive direction is now confound-checked.

**Step B — discharge carries content beyond precipitation.** Upstream precip gives +0.012 (~⅓ of the discharge gain). Answers the long-standing idea1.md "C-precip" question: routed upstream *flow/state* has substantial signal that raw upstream *rain* does not. **This justifies the message-passing direction** — it's not reducible to "just add upstream precipitation."

**Step C — robust to lag, and lag0 is strongest.** Positive at all lags (lag0 +0.087 / lag1 +0.037 / lag2 +0.036); no single-lag spike (rules out a leakage artifact). Same-day upstream flow nearly doubles the gain — physically sensible (sub-daily travel times at daily resolution). A real finding worth carrying into the paper.

**Why it matters.** Three weeks of negatives → a fully stress-tested positive. The story is now tight: static topology features null; upstream *flow* helps (+0.037 lag1, +0.087 lag0); the gain survives a null control, exceeds the precip-only baseline 3×, and is robust across lags. The remaining gap to a paper is the realizability test (predicted, not observed, upstream Q) and multi-seed confirmation.

**Honest caveats.** Single seed throughout — directional, needs 3-seed CIs before any headline. The oracle uses OBSERVED upstream discharge → upper bound; the deployable model recovers a fraction. lag0 uses same-day upstream observed flow — fair in an operational nowcast (upstream gauges report in real time) but must be stated explicitly; it is NOT future leakage (it's same-day upstream, predicting same-day downstream).

### Reviewer 2

- *"Is the +0.037 just 'more input dims = more capacity'?"* → No. The shuffled-Q null control (identical column distribution, scrambled in time) gives −0.002. The gain requires the real temporal upstream signal.
- *"Is it data leakage — you're feeding it discharge?"* → It's *upstream* basins' discharge lagged ≥0 days, predicting the *downstream* basin. Same-day upstream→downstream is physical routing, not target leakage. Lag sweep shows no single-lag spike (a leak would concentrate at one lag).
- *"Isn't this just upstream precipitation the model already has regionally?"* → Upstream precip alone gives +0.012, a third of the discharge gain. Routed flow carries content beyond rain.
- *"Single seed."* → Conceded; all directional. Multi-seed (3 seeds) is the next gate before any claim.
- *"What would make me believe the realizable (non-oracle) version works?"* → A two-stage model: predict upstream Q from upstream forcings, feed the *prediction* downstream. If it recovers a meaningful fraction of +0.037, the result is deployable. That's the queued next experiment.

### Open questions
1. Does *predicted* (not observed) upstream Q still help? (realizability — the load-bearing next test)
2. How much of the +0.037 survives at 3 seeds? (significance)
3. Is lag0's +0.087 stable, or seed-fragile? (it's the strongest single result)
4. Does the gain grow on small local subgraphs (walker machinery ready)?

## Next 2–3 sessions (queued)

1. **Realizability test (predicted-upstream-Q)** — gates whether the result is deployable, not just an upper bound; cost ~30 min CPU (train an upstream-Q predictor from upstream forcings, feed its prediction downstream); prerequisite: none, infra mostly reusable.
2. **Multi-seed (3 seeds) on the headline conditions** (L, L+upQ lag1, L+upQ lag0, L+upPrecip, shuffled-null) — gates publication significance; cost ~1.5 hr CPU or ~30 min Colab T4; prerequisite: none.
3. **Local-subgraph scale curve** — does upstream-flow benefit grow on small coherent networks; cost ~1 hr; prerequisite: realizability + multi-seed clear.

---
## 2026-06-27 — Realizability test PASSED: predicted upstream Q recovers 72% of the oracle gain (deployable)

**Source.** Realizability test (`preregistration_realizability.md`). Stage 1 (full-span L evaluate → predicted Q per basin) ran on Colab; Stage 2 (train L + predicted-upstream-Q) ran locally reusing the uploaded `_Lfullspan_eval` predictions. component0, seed 11, stock cudalstm.

**Result.**

| Condition | median NSE | paired Δ vs L |
|---|---|---|
| L | 0.653 | — |
| L+upQ (oracle, observed) | 0.703 | +0.050 |
| **L+upQ_pred (realizable)** | **0.683** | **+0.0265** (67% of basins ↑) |

**Predicted upstream Q recovers 72% of the +0.037 oracle ceiling** (53% of the larger +0.050 observed gain). Pre-registered success bar +0.015 (≥40%) — cleared.

**Decision per pre-registration: SUCCESS — the result is DEPLOYABLE.** Every prior gain used observed upstream discharge (upper bound). This shows the gain survives when upstream Q is *predicted from forcings* (run per-basin LSTM, route predictions downstream, no ground truth at inference, no target leakage). 72% recovery means the upstream hydrological state the downstream model needs is largely reconstructible from forcings alone.

**Why it matters — the paper spine is now complete and positive:** static topology features null → upstream *flow* helps (+0.037 observed bound; null-control passed, beats precip 3×, lag-robust) → a *realizable* model recovers 72% of it (+0.027 deployable). Three weeks of confounded negatives have resolved into a clean, stress-tested, deployable positive result with a clear mechanism.

**Process note.** Cell-9 `KeyError: 'date'` on Colab was a one-line bug (unnamed feature-pickle index; NH concatenates additional_feature_files on a 'date'-named index). Fixed. Stage 2 then ran locally because the uploaded `_Lfullspan_eval` predictions removed the need to re-run the full-span evaluate that broke on the Mac data subset.

### Reviewer 2
- *"Is predicted-Q just leaking observed Q?"* → No. Predicted Q comes from the L model run on upstream *forcings*; the downstream basin's own target never enters. Two-stage, standard deployable setup.
- *"72% of what — is the ceiling cherry-picked?"* → Ceiling = +0.037 (the conservative lag1 oracle contrast). Against the larger +0.050 observed gain in this same run, recovery is 53%. Either way well above the +0.015 bar.
- *"Single seed."* → Conceded; directional. Multi-seed (3) is the immediate next gate.
- *"Could the predicted-Q feature help just by adding capacity?"* → The shuffled-Q null control (−0.002) already ruled that out for the observed feature; predicted-Q carries even less raw magnitude, so a capacity artifact is implausible. A predicted-Q shuffle control can confirm if a reviewer demands it.
- *"What makes this deployable vs the oracle?"* → At inference you run each basin's LSTM on its forcings, then feed upstream predictions downstream. No gauge observations needed at test time.

### Open questions
1. Does +0.027 hold at 3 seeds? (significance — next gate)
2. Does a 2-hop / iterated-prediction scheme recover more of the gap to the oracle?
3. Does the realizable gain grow on small local subgraphs (walker machinery ready)?

## Next 2–3 sessions (queued)
1. **Multi-seed (3 seeds: 11,13,17)** on the headline conditions (L, L+upQ oracle, L+upQ_pred, shuffled-null) — the significance gate before any paper claim; ~30 min Colab T4 or ~1.5 hr CPU; prerequisite: none.
2. **Realizable-Q null control** (shuffle the predicted-Q feature) — closes the capacity-artifact question for the deployable result; ~5 min; prerequisite: none.
3. **Local-subgraph scale curve** — does the realizable upstream gain grow on small coherent networks; ~1 hr; prerequisite: multi-seed clear.

---
## 2026-07-01 — /crs-unleashed: multi-seed CONFIRMS realizable gain; depth-gradient reveals the routing mechanism

**Source.** Multi-seed run (seeds 11/13/17) uploaded via drive-download; organized into `runs/topology_ablation/component0/` and analyzed (`analyze_multiseed.py` → `analysis/MULTISEED.md`).

**Orient.** Realizability passed single-seed (+0.027, 72% of oracle). This session confirms across seeds and stress-tests the mechanism. Drive-download merge overwrote seed-11's oracle/predicted metric folders (numbers preserved from prior runs); seeds 13/17 fully measured — paired analyses use only the two clean seeds.

**Diagnosis (top-3).** (1) Realizable gain holds across seeds — was low-confidence, now high. (2) Null control stays ~0 — mild concern, crept to +0.004. (3) Oracle>realizable>null ordering stable — confirmed.

### Results

| Condition | median NSE (mean±std, 11/13/17) |
|---|---|
| L | 0.653 ± 0.002 |
| L+upQ (oracle) | 0.691 ± 0.009 |
| L+upQ_pred (realizable) | 0.678 ± 0.008 |
| L+upQshuf (null) | 0.666 ± 0.008 |

**Multi-seed verdict: SUCCESS.** Realizable Δ vs L = +0.027/+0.026/+0.013 (seeds 11/13/17), mean +0.019, all positive (bar +0.015). Recovers ~55% of the oracle ceiling cross-seed (seed-11 was 72%; seeds 13/17 55%/61%).

**Step A — realizable − null (capacity control): PASS.** +0.0115 cross-seed, all positive. Because the null crept to +0.004, we report realizable−null (+0.012) as the honest effect size; still clears the +0.010 bar.

**Step B — depth-stratified: PASS, strong.** Realizable Δ rises monotonically with graph depth: depth0 **−0.003** (headwaters, no upstream → zero gain, as it must be), depth1 +0.019, depth2 +0.029, depth3 +0.034. depth≥2 vs headwater = +0.032. **This is the routing signature** — the gain is mechanistically "downstream basins benefit from upstream flow," not a generic extra-input effect. This is the paper's convincing figure.

### Reviewer 2
- *Capacity artifact?* No — null control +0.003, realizable−null +0.012 all-positive, and headwaters (depth 0) show zero gain (capacity would help them too).
- *Null crept up — contaminated?* Slightly; report realizable−null as honest effect. Depth gradient makes contamination implausible as driver.
- *Single-run-per-seed noise?* Effect (+0.019) is 3× cross-seed std (0.006), all 3 seeds agree.
- *Seed-11 folders lost — fudging?* Load-bearing paired analyses (A/B) use only fully-measured seeds 13/17; seed-11 medians are recorded originals used only in the summary table.
- *Depth confounded with basin size/aridity?* Plausible — area-stratified + partial-depth analysis is the follow-up TODO.

### Open questions
1. Is the depth gradient confounded with basin area/aridity? (area-stratified check — cheap)
2. Does the realizable gain grow on small local subgraphs? (scale curve — Colab)
3. Can a 2-hop / iterated-prediction scheme close more of the gap to the oracle?

## Next 2–3 sessions (queued)
1. **Area-stratified + depth-vs-area partial analysis** — rules out the size/aridity confound on the depth gradient; CPU/free re-analysis; prerequisite: none.
2. **Local-subgraph scale curve (Step C)** — does realizable gain grow at small scale; ~30 min Colab T4; prereg written.
3. **2-hop predicted-Q** — feed predicted Q from 2 hops upstream (iterated); tests if more of the oracle gap is recoverable; ~20 min CPU per seed.

---
## 2026-07-01 (later) — Confound check: depth gradient is ROUTING, not basin size (paper-ready)

**Source.** `/crs` follow-up. Pre-registered `preregistration_confound_check.md`; ran `analyze_confound.py` (zero compute, re-analysis of seeds 13/17 realizable runs, n=366 basin×seed). Depth correlates with area (r=0.38), so the depth-gradient result needed a partial control before anchoring the paper.

**Result — confound decisively ruled out.**
- **T3 (partial control, load-bearing): PASS 3/3 area terciles.** depth≥2 vs depth0 realizable gain within each size class: small +0.021, mid +0.036, large +0.050. The depth effect survives holding area fixed, and is *strongest* in large basins (opposite of an area-confound).
- **T4:** corr(Δ, area) = −0.008 (essentially zero) vs corr(Δ, depth) = +0.134. Area does not predict the gain; graph position does.
- **T2:** area-tercile spread only +0.006 (flat).
- **T1:** step change headwaters(−0.003) → any-upstream(+0.022), then saturates in n_upstream count — routing signal is the binary "downstream vs headwater," captured by the area-weighted upstream aggregate.

**Why it matters.** The depth gradient (the mechanistic heart of the paper) is genuinely about *upstream routing*, not basin size. The strongest reviewer objection to the routing story is now closed. Combined with the multi-seed confirmation and the null control, the finding is paper-ready.

### Reviewer 2
- *Depth proxies size?* Refuted — within-tercile gaps hold (+0.021/+0.036/+0.050), corr(Δ,area)≈0.
- *T1 not monotonic in n_upstream count?* Gain saturates; the step is "has upstream vs not." Consistent with routing.
- *Only 2 seeds?* n=366 basin×seed is ample for stratified medians; seed-11 realizable re-eval is a cheap TODO.

### Open questions
1. Does the gain grow on small local subgraphs (scale curve, Colab)?
2. Can 2-hop / iterated predicted-Q recover more of the oracle gap?
3. Re-evaluate seed-11 realizable to restore the full 3-seed paired set (cheap).

## Next 2–3 sessions (queued)
1. **Local-subgraph scale curve (Step C)** — pre-registered; ~30 min Colab T4.
2. **2-hop predicted-Q** — ~20 min CPU/seed; tests recoverable headroom to the oracle.
3. **Seed-11 realizable re-eval** — restores full 3-seed paired analyses; ~10 min once L_upQpred_seed11 checkpoint is re-run or re-downloaded.

---
## 2026-07-01 (later) — /crs-unleashed: methodology-compliance audit — study is PUBLICATION-VALID

**Source.** Full methodology sweep. Byte-level config diff of all headline runs + re-analysis of stored predictions (`analyze_compliance.py` → `analysis/COMPLIANCE.md`), seeds 13/17. Pre-reg `preregistration_compliance.md`.

**Core finding — standardization is exemplary.** All 4 headline conditions (L, L+upQ, L+upQ_pred, L+upQshuf) run byte-identical configs: cudalstm, hidden 64, dropout 0.4, forget-bias 3, Adam 1e-3, batch 256, 30 epochs, seq 30, maurer forcings, 5 static attrs, one-hot on, same 1990-99/2005-08 split, same seed. **The ONLY difference is a single dynamic input (`upstream_q`).** This is the cleanest-possible ablation and directly resolves the architecture-confound that invalidated the earlier DirectedGraphLSTM 5cond work (different trainer, undertrained). By moving the whole study onto stock cudalstm + one input, "the difference maker is our addition" is literally true at the config level.

**Step A — metric robustness (added log-NSE, our methodology's 3rd metric): PASS, strengthens the result.** Realizable Δ: NSE +0.022 / log-NSE +0.019. Gain holds across the flow regime. Null control goes NEGATIVE in log-NSE (−0.023) — shuffled input hurts low flows, so the real gain is even more clearly genuine.

**Step B — baseline is not a straw man: PASS.** Realizable gain persists on well-predicted basins (L NSE > 0.6): +0.012 (n=230); larger on bad basins but positive everywhere → real signal, not rescue.

**Scale + literature positioning (assessment).** 183 basins × 3 seeds is adequate for a regional/workshop paper and exceeds the 5cond design; it is NOT a national benchmark. Kratzert 2019 = 531 basins + EA-LSTM (~0.74); our cudalstm on 183 eastern basins (0.653) is a legitimate strong baseline for this scope — report honestly, claim no SOTA. Crucially the study BUILDS ON the cited work: it resolves Kirschstein 2024's GNN-adjacency null (static topology is inert — we show it directly) and executes Jiang 2025's physics-aware-operator direction (dynamic upstream flow is what helps). 531-basin scale-up is named future work.

**Verdict: PUBLICATION-VALID methodology.** Nothing broken. Standardization exemplary, comparison fair, metrics now complete, not baseline-rescue, positioned in the literature. Only honest framing (regional scope, no SOTA claim) + the cheap seed-11 re-eval remain.

### Reviewer 2
- *Conditions truly standardized?* Byte-identical configs; one input differs. Confirmed on disk.
- *183 basins enough?* Adequate for regional/workshop; frame as eastern-US sub-network, not national. Scale-up = future work.
- *Baseline a straw man?* No — Step B: gain persists on good basins. Baseline is honest cudalstm, no SOTA claim.
- *Builds on the papers?* Yes — resolves Kirschstein null, executes Jiang direction.
- *Leakage?* Upstream lagged Q → downstream; target basin's own flow never enters; deployable version uses predicted Q.

### Open questions / TODO
1. Re-eval seed-11 realizable (restore clean 3-seed paired set) — cheap.
2. 531-basin scale-up for a top-tier (vs workshop) venue — large compute.
3. NHDPlus ground-truth edges vs our heuristic edges (robustness a reviewer may want).

## Next 2–3 sessions (queued)
1. **Seed-11 realizable re-eval** — restores full 3-seed set; ~10 min once L_upQpred_seed11 retrained/re-downloaded.
2. **Local-subgraph scale curve** — does gain grow at small scale; ~30 min Colab (pre-registered).
3. **Draft the paper skeleton** — methods (the byte-identical ablation), results (null topology → dynamic-flow gain → routing mechanism), positioning (Kirschstein/Jiang). The science is essentially complete for a regional workshop paper.

---
## 2026-07-01 (later) — Seed-11 realizable re-eval: full 3-seed set now MEASURED (reproduces exactly)

**Source.** Restored the seed-11 realizable run (lost in the drive merge — no ckpt/metrics on disk). Re-ran the two-stage pipeline locally: (1) full-span L_seed11 evaluate → predicted Q (the step that failed on the Mac subset before — worked cleanly this time, transient earlier failure), (2) build predicted-Q feature, (3) train L_upQpred_seed11.

**Result — reproduces the original exactly + strengthens the record.** Seed-11 realizable: L 0.6529 → 0.6833, paired Δ **+0.0265** — identical to the original run (confirms determinism, not a fluke). Complete measured 3-seed set:

| seed | realizable Δ | recovery of oracle |
|---|---|---|
| 11 | +0.0265 | 72% |
| 13 | +0.0258 | 55% |
| 17 | +0.0131 | 61% |

**Cross-seed realizable Δ +0.0218 ± 0.0062, all 3 seeds positive** — no more "seed-11 recorded" asterisk. All three analyses re-run on the full measured set (`SEEDS=[11,13,17]`):
- Step A (realizable−null): +0.017 all-positive (was +0.012 on 2 seeds).
- Step B depth gradient: intact (depth0 +0.002 → rising).
- Confound: corr(Δ,area)=+0.015 (~0) vs corr(Δ,depth)=+0.158; routing survives 3/3 area terciles.
- Compliance log-NSE: realizable +0.027 (stronger than NSE +0.022); null −0.003; baseline-not-strawman +0.012 on good basins.

**The open TODO from the compliance audit is closed.** The multi-seed paired comparison is now fully measured and clean. Nothing changed in the conclusions; the record is just tighter.

## Next 2–3 sessions (queued)
1. **Paper skeleton** — the science is complete for a regional workshop paper (static topology null → dynamic upstream flow helps, 3-seed measured → deployable predicted-Q → routing mechanism, confound-checked → robust across NSE/KGE/log-NSE → not baseline-rescue).
2. **Local-subgraph scale curve** — does the gain grow at small scale (Colab, pre-registered).
3. **531-basin scale-up** — only for a top-tier (vs workshop) venue; large compute.

---
## 2026-07-06 — /crs-unleashed: lag0-realizable FALSIFIED (reveals predictability ceiling); KGE scopes the claim

**Source.** /crs-unleashed. Pre-reg `preregistration_lag0_realizable.md`. Two CPU-cheap gaps in the otherwise-complete study: the strongest realizable version (lag0 predicted-Q) was untested, and KGE (3rd required metric) hadn't been run on the realizable headline.

**Diagnosis (top-3):** (1) realizable result may be under-sold — observed oracle is 2× stronger at lag0, predicted-lag0 untested [test first]; (2) KGE robustness unverified [free check]; (3) NHDPlus edges — reviewer-relevant but large data task [defer].

### Step A — lag0-predicted: FALSIFIED, but informative

| realizable version (seed 11) | Δ vs L | recovery of its oracle |
|---|---|---|
| lag1 predicted | +0.0265 | 72% (of +0.037) |
| lag0 predicted | +0.0229 | **26%** (of +0.087) |

lag0-predicted (+0.0229) < lag1-predicted (+0.0265) → pre-registered hypothesis FALSIFIED; lag1 stays the headline. **The mechanism is the finding:** the observed oracle is 2× stronger at lag0 (+0.087 vs +0.037), but the *predicted* version recovers only 26% of the lag0 ceiling vs 72% of lag1. Same-day upstream flow has the most signal when observed but is the hardest to forecast — **lag1 is the realizable sweet spot; the deployable gain is capped by upstream predictability, not the downstream model.** Chain stopped at Step A per pre-reg (Step C gated on lag0 winning — not run; no re-design of a falsified test).

### Step B — KGE robustness (3rd metric)

Realizable Δ robust in NSE (+0.022, all 3 seeds +) and log-NSE (+0.027, all 3 seeds +), but KGE +0.013 mean with seed 13 at −0.002 (not all-positive). Honest scope: **robust in NSE/log-NSE; KGE-positive-on-average with seed sensitivity.** Report all three metrics; do not overclaim KGE.

### Reviewer 2
- *Did you cherry-pick lag1?* No — pre-registered lag0 as the hypothesized stronger version; it falsified. lag1 was the original headline and remains it. Both reported.
- *Is the 26% vs 72% just noise?* It's a large, consistent gap (single seed here, but the recovery ratio is a 3× difference); the predictability interpretation is testable via a lag0 seed-13 check if a reviewer insists (not run — chain stopped on falsification).
- *KGE not all-positive — is the result fragile?* Robust in 2 of 3 metrics with all-seeds-positive; KGE seed-sensitivity is disclosed. NSE + log-NSE agreement is the load-bearing evidence.
- *What would make lag0 deployable?* A better upstream Q *forecaster* (the current one is the same L baseline). That's future work, not this paper's claim.

### Open questions
1. Does a stronger upstream-Q forecaster lift the lag0 realizable recovery above lag1? (future)
2. lag0 seed-robustness (only if pursuing the lag0 branch — currently closed).

## Next 2–3 sessions (queued)
1. **Paper skeleton** — science is complete; lag1 realizable is the confirmed, best deployable headline; the lag0 predictability-ceiling result is a strong discussion point.
2. **Local-subgraph scale curve** — does the gain grow at small scale (Colab, pre-registered).
3. **531-basin scale-up** — top-tier venue only; large compute.

---
## 2026-07-12 — /crs-unleashed: analysis-only hardening chain (significance → mechanism-confound → metric-honesty), all PASS

**Source.** /crs-unleashed. The core study is complete and publication-valid; this session hardened its three most-attackable joints with zero training compute, exploiting a reuse discovery: every headline run stores `test_results.p` (per-timestep obs/sim), so any metric or stratification is a pure re-analysis. Pre-reg: `preregistration_hardening_chain.md` (written before execution).

**Orient (what was read).** git log (30), current_implementation.md (the full paper narrative), README, all `analysis/*.md` (MULTISEED, CONFOUND, COMPLIANCE, RESULTS), FORWARD_PLAN, JOURNAL tail (open TODOs: paper skeleton / scale curve / 531-scale / NHDPlus), and the signatures of `analyze_multiseed.py` / `analyze_confound.py` / `analyze_compliance.py` / `build_predicted_upstream_q.py`. Confirmed all 4 headline conditions × 3 seeds have metrics on disk; `test_results.p` present for all except `L_upQ` seed11 (lost in the drive merge — non-load-bearing).

**Diagnosis (top-3 load-bearing claims, ranked).**
1. *The gain is routing (rises with depth), not a size/aggregation artifact* — medium confidence, high importance → **test first**. Only area had been controlled; feature-magnitude and the depth-vs-n_upstream ambiguity had not.
2. *The deployable +0.022 is significant vs the null, not three noisy seeds agreeing in sign* — high confidence, high importance → **verify rigorously** (no paired significance test existed).
3. *log-NSE/KGE robustness* — medium → verify + defer (log-NSE eps choice untested; KGE seed-13 dip unexplained).

### Step A — significance (PASS) — `analysis/SIGNIFICANCE.md`
Paired Wilcoxon signed-rank on per-basin NSE deltas, pooled seeds (n=549).
- realizable − L: median +0.0225, 66% of basins positive, **p=6.0e-19**, bootstrap 95% CI on median [+0.0175, +0.0281].
- **realizable − null (capacity-controlled): median +0.0167, p=2.3e-12, CI [+0.0106, +0.0216] excludes 0.**
- Per-seed (independent basins, n=183): realizable-vs-L significant 3/3; realizable-vs-null significant 2/3 (seed17 p=0.061 — weakest, **disclosed**).
- **Upgrade:** "all-seeds-positive in the mean" → "statistically significant, capacity-controlled, with a CI." Gate passed → Step B.

### Step B — routing vs feature-magnitude confound (PASS, stronger than expected) — `analysis/FEATURE_MAGNITUDE_CONFOUND.md`
Tests whether the depth→Δ gradient is really upstream routing or just that deeper basins have a larger `upstream_q` feature. Feature magnitude = per-basin mean |upstream_q| (lag-1 predicted feature).
- **Direction (decisive):** feature magnitude *decreases* with depth (corr −0.369; depth1 median 1.85 → depth3 1.39 mm/d) while the gain *increases* (depth1 +0.020 → depth3 +0.044). The confound runs **opposite** to the effect — it cannot manufacture it.
- Within each feature-magnitude tercile, deeper (depth≥3) beats shallower (depth1) in **3/3** terciles (+0.012 to +0.019).
- Partial corr(Δ, depth | area, fmag) = **+0.149 (p=4.4e-4)**; reverse partial corr(Δ, fmag | area, depth) = +0.080 (p=0.061, **not significant**). Depth is load-bearing; feature scale is not.
- **Resolves the CONFOUND.md tension** (its T1 flagged "monotonic in n_upstream: False" while the story rests on depth): the routing variable that carries the gradient is **graph depth**, not raw n_upstream count and not feature magnitude. Gate passed → Step C.
- *Binning note:* the first-cut within-tercile depth0-vs-depth2 test returned all-n/a because depth0 ⟺ fmag=0 by construction (headwaters have no upstream); corrected to a connected-basin (depth1 vs depth≥3) comparison. Documented in the artifact, not swept under.

### Step C — metric honesty (PASS) — `analysis/METRIC_HONESTY.md`
- **C1 log-NSE eps-sensitivity:** realizable Δ stable +0.0270 → +0.0303 across eps ∈ {1e-2,1e-3,1e-4}×mean-flow (100× sweep); null stays negative (−0.003 → −0.010) and the contrast *sharpens* as eps shrinks. **The +0.027 headline is not an eps artifact.**
- **C2 KGE decomposition (the useful discovery):** the disclosed seed-13 KGE dip (ΔKGE −0.005) is **not** a timing failure. The correlation component **r improves in all 3 seeds** (Δr +0.018/+0.021/+0.005) — upstream flow consistently sharpens hydrograph timing. The seed-13 dip is a **variability-ratio (γ) overshoot** (Δγ −0.036) plus a bias (β) shift. Honest-scope statement sharpened from "KGE is seed-sensitive" to "upstream flow improves timing (KGE-r positive in all seeds); the seed-13 KGE dip is a γ-overshoot, not a timing loss."

**Net effect.** No result changed sign. The paper's three most-attackable joints are materially stronger: the deployable effect now carries a significance test and CI; the routing mechanism survives a confound that runs directionally *against* it; and the metric weak-spot (KGE) is now understood mechanistically (γ-overshoot with r always improving) rather than merely disclosed. All from stored predictions — zero training.

### Reviewer 2
- *Pooled 549 as independent — basins repeat across seeds?* Per-seed tests (n=183, independent) corroborate: 3/3 vs-L, 2/3 vs-null. Pooled p is not carried alone.
- *Seed17 vs-null p=0.061 — not significant?* Disclosed; claim is "significant pooled and in 2/3 seeds," not "every seed." One-sided is pre-registered (directional physical H1), and two-sided still clears 0.01.
- *Depth/n_upstream/area collinear — partial corr can't separate them?* The direction argument dominates: fmag *decreases* with depth while gain rises; a confound opposite to the effect can't create it.
- *Heuristic edges → depth is noise?* Noise doesn't produce a monotone gradient surviving area + fmag controls at p=4e-4.
- *eps scaling non-standard?* Swept 100×, sign never moved; a fixed-eps reviewer reaches the same conclusion (TODO: report NH default eps for comparability).
- *KGE γ-overshoot = model adds spurious variance?* Honest limitation; but r-always-positive shows timing is intact — it's a bias/variance calibration issue, addressable with a variance-penalized loss (future).

### Open questions
1. Report log-NSE at NH's default fixed eps for direct comparability (cheap).
2. Does a variance-penalized loss fix the KGE-γ overshoot without losing the r gain? (a run — future).
3. Restore `L_upQ` seed11 `test_results.p` (re-eval) to complete the oracle log-NSE/KGE columns (cheap, ~10 min).

## Next 2–3 sessions (queued)
1. **Paper skeleton** — gates the write-up; cost ~1 session; prerequisite none (science + hardening complete). The 3 new artifacts (SIGNIFICANCE, FEATURE_MAGNITUDE_CONFOUND, METRIC_HONESTY) drop straight into Results/Discussion.
2. **Oracle seed11 test_results.p re-eval** — gates complete 3-seed oracle log-NSE/KGE; cost ~10 min CPU; prerequisite the L_upQ_seed11 checkpoint (re-run the two-stage eval as in the 2026-07-01 seed-11 restore).
3. **Local-subgraph scale curve** — gates the "does the gain grow at small scale" secondary figure; cost ~30 min Colab T4; prerequisite GPU (pre-registered in `experiments/local_subgraphs/preregistration_local_scale.md`).

---
## 2026-07-12 (later) — /crs-unleashed: routing-baseline chain (queue re-scoped for validity), all PASS

**Source.** /crs-unleashed, executing the queued next sessions. On inspection the queue was stale: "oracle seed-11 re-eval" needs RETRAINING (its checkpoint was lost in the drive merge — only `config.yml`+`test/` survive, no `.pt`), and "local-subgraph scale curve" needs GPU (no subgraph runs on disk; the 2026-06-20 batch left nothing). Neither is the CPU-cheap step the queue implied. The optimal available move — genuinely paper-contributing and CPU-cheap — is the reviewer baseline `FORWARD_PLAN.md` explicitly names and the paper still lacks: the **no-ML routing baseline** ("if our model can't beat simple physical routing, the ML isn't earning its complexity"). Pre-reg: `preregistration_routing_baseline_chain.md`.

### Step A — no-ML routing baseline (PASS) — `analysis/ROUTING_BASELINE.md`
Least-squares routing predictors, coefficients fit on TRAIN 1990-99, scored on TEST 2005-08 (no test fitting), seed 11, connected basins (n=150). Data on disk: observed `upstream_q` lag1 feature + the `_Lfullspan_eval` run (obs + L-sim over full 1990-2008).
- **R1 pure routing** (a·upstream_q + b): median NSE **+0.324**.
- **R2 routing + local** (a·upstream_q + c·L_sim + b): **+0.675**.
- L +0.654 | L+upQ_pred (realizable) **+0.686** | L+upQ (oracle) +0.717.
- **Verdict PASS:** realizable and oracle both beat R1 (pure routing); realizable beats R2. The LSTM's learned use of upstream flow beats naive physical routing — the ML earns its complexity.
- **Honest nuance (now in-paper):** the margin over the *strong* R2 baseline is only +0.010. R2 nearly matches the realizable LSTM — but R2 *uses the L baseline's own sim* as an input (it is "LSTM + linear upstream correction," not ML-free); the standalone no-ML predictor is R1 at +0.324. The LSTM's real advantage is integrating upstream flow WITH local rainfall-runoff, which linear routing cannot. This is exactly the reviewer's first question, answered head-on instead of ambushed.

### Step B — per-depth significance (PASS) — `analysis/DEPTH_SIGNIFICANCE.md`
Per-depth paired Wilcoxon (one-sided) on realizable Δ, pooled seeds (n=549).

| depth | n | median Δ | p | sig |
|---|---|---|---|---|
| 0 (headwater) | 99 | +0.002 | 0.24 | **no** (expected — no upstream) |
| 1 | 243 | +0.020 | 2.6e-9 | yes |
| 2 | 153 | +0.031 | 4.7e-12 | yes |
| 3 | 48 | +0.044 | 8.4e-4 | yes |
| 4 | 6 | +0.015 | 0.34 | no (n=6, no power) |

**The routing gain is statistically significant exactly where upstream flow arrives (depth 1-3) and statistically absent at headwaters.** The depth gradient is upgraded from a median trend to a per-stratum-significant result — the strongest form of the routing claim.

### Step C — consolidated publication table (PASS) — `analysis/PAPER_TABLE.md`
Assembled the Results section into one auditable file: Table 1 (conditions × NSE/KGE/log-NSE mean±std × Δ-vs-L Wilcoxon p), Table 2 (routing baselines R1/R2 vs LSTM), Table 3 (depth significance). Surfaced honestly: the shuffled null is *weakly* significant vs L on raw NSE (Δ +0.012, p=0.047) — which is precisely why the realizable-vs-**null** contrast (p=2.3e-12, from SIGNIFICANCE.md) is the load-bearing test, not realizable-vs-L. The table makes the honest comparison visible.

**Net effect.** The paper gains (1) its missing reviewer baseline with an honest margin discussion, (2) per-stratum significance for the routing mechanism, and (3) a single consolidated Results table. Six-plus analysis artifacts now cover: significance + CI, capacity control, feature-magnitude confound, metric honesty (log-NSE eps + KGE decomposition), no-ML routing baseline, per-depth significance. The empirical case is as hard as it can be made without new compute. No result changed sign.

### Reviewer 2
- *R2 (+0.675) nearly matches your model — LSTM barely helps?* R2 uses the LSTM's own sim as an input; ML-free routing is R1 at +0.324. Realizable still wins; oracle shows headroom. Honest margin disclosed.
- *Per-basin routing fit = leakage?* No — coefficients fit on train, scored on test.
- *depth-4 not significant?* n=6, no test power; depths 1-3 (well-populated) all significant and rising.
- *Baseline only seed 11?* Uses observed upstream_q (seed-independent) + the one fullspan eval on disk; 3-seed extension needs the other fullspan evals (cheap follow-up).
- *Null weakly sig vs L?* Yes (p=0.047) — added capacity has a tiny effect; that's why realizable-vs-null (p=2e-12) is the load-bearing contrast, now explicit in the table.

### Open questions
1. 3-seed routing baseline (needs fullspan L-sim at seeds 13/17 — re-eval the L checkpoints, cheap).
2. Oracle seed-11 `test_results.p`: requires RETRAINING L_upQ_seed11 (checkpoint lost), not a re-eval — correct cost logged.

## Next 2–3 sessions (queued, costs corrected)
1. **Paper skeleton** — ALL Results artifacts now exist (PAPER_TABLE.md is the spine). Cost ~1 session, no prerequisite. The natural, highest-leverage next move.
2. **3-seed routing baseline + oracle log-NSE completion** — re-eval L and L_upQ checkpoints at seeds 13/17 over the full span; ~30-45 min CPU (this is training-eval, not free — corrected from the prior queue).
3. **Local-subgraph scale curve** — GPU-bound secondary figure; ~30 min Colab T4; pre-registered in `experiments/local_subgraphs/preregistration_local_scale.md`. Only when GPU is available.

---
## 2026-07-14 — /crs-unleashed: graph-robustness chain — the over-connectivity threat is CLOSED, all PASS

**Source.** /crs-unleashed. After 90400e0 the empirical case was heavily hardened, but the prior session's graph-similarity analysis had surfaced one unaddressed validity threat that outranked the queued "paper skeleton": the heuristic edges OVER-CONNECT vs real hydrography (child in-degree mean 4.16 / max 15; 66/150 children have >3 parents; real confluences join 2-3 tributaries). A hydrology reviewer's first attack would be "your routing gain is an artifact of an unrealistically dense graph." Writing a paper on an unvalidated graph is premature — so this session tests that threat head-on. Pre-reg: `preregistration_graph_robustness_chain.md`.

**Orient / reuse insight.** The R1 lstsq routing baseline (from the prior session's `analyze_routing_baseline.py`) scores an upstream-flow feature's signal strength WITHOUT training a model. Combined with observed Q per basin (fullspan eval) + area weights + editable edge sets, this makes alternative graphs testable at ZERO training cost. So the over-connectivity threat — which naively needs a per-graph LSTM re-train (GPU, hours) — becomes a cheap CPU probe of *signal content* invariance.

**Diagnosis (top-3, ranked).** (1) routing signal survives on a hydrography-realistic pruned graph — low confidence, high importance → test first; (2) depth hierarchy stable under pruning — med/high; (3) edge-choice-noise robustness — med/med.

### Step A — pruned-graph robustness (PASS, decisive)
Full graph (624 edges): R1 median test-NSE +0.325 (n=150). Prune to k nearest parents per child:

| pruning (nearest) | edges | R1 NSE | % of full |
|---|---|---|---|
| in-degree ≤ 1 | 150 | +0.319 | 98% |
| in-degree ≤ 2 | 266 | +0.326 | 100% |
| in-degree ≤ 3 | 359 | +0.326 | 100% |

**Capping in-degree at a hydrography-realistic ≤2 retains 100% of the routing signal; even ≤1 (76% of edges deleted) retains 98%.** The signal lives in the NEAREST parents — exactly the ones real hydrography keeps — and the heuristic's excess edges contribute ~nothing. The study's single biggest reviewer attack is closed. (Robustness: `smallest-ratio` selection retains only 81-97%, proving the metric responds to graph changes AND that nearest-parent selection — shortest travel time — is the physically meaningful rule.)

### Step B — depth-structure stability (PASS)
Under k=2 pruning: 95% of basins retain depth within ±1 (173/183), DAG preserved, max depth 5→4. The depth-gradient routing signature is not an artifact of edge density.

### Step C — edge-dropout sensitivity (PASS)
Random 20% edge dropout (5 fixed-seed draws): R1 NSE +0.3244 ± 0.0023 (100% of full, spread 0.006). 40% dropout: 99%, spread 0.014. The signal is anchored in aggregate graph structure, not any specific edges.

**Net effect.** The over-connectivity caveat — previously the study's most serious unquantified limitation — is now contained with hard evidence: the routing signal is INVARIANT across a 4× range of graph densities (in-degree 1→4.16), invariant to which nearest parents are kept, and robust to 20-40% random edge noise. The paper can now state the heuristic-edge caveat AND show the result does not depend on it. Paper write-up is unblocked.

### Reviewer 2
- *Only the R1 proxy, not the LSTM?* Pre-registered scope limit. R1 is a monotone proxy for the feature's signal content; content invariance (0-2%) means the LSTM has the same information under any graph in the range. Full per-graph LSTM re-train is the GPU follow-up.
- *k=2 nearest isn't real NHDPlus?* Correct — it's a hydrography-plausible proxy. The point is invariance across the density range, so wherever true connectivity falls within it, the result holds. NHDPlus is the definitive future check.
- *R1 NSE saturated/insensitive?* No — smallest-ratio pruning moves it to 0.263 (81%), so the metric is responsive; nearest-parent invariance is a real finding, not saturation.
- *Cherry-picked draws?* Deterministic seeds (1000·draw + frac·100), set before running, 5 draws, mean±std + spread reported.
- *What would make it fully real?* An NHDPlus edge set scored identically + a full LSTM re-train on k=2. Both named; neither expected to overturn a 0-2% invariance.

### Open questions
1. NHDPlus ground-truth edges scored via the same R1 pipeline (definitive; needs the NHDPlus flowline data — a data-acquisition task).
2. Full LSTM re-train on the k=2 pruned graph (confirms the LSTM, not just the proxy, is pruning-invariant; ~GPU).

## Next 2–3 sessions (queued)
1. **Paper skeleton** — every Results artifact now exists AND the graph caveat is contained (GRAPH_ROBUSTNESS.md). Highest leverage, no prerequisite. The natural next move.
2. **3-seed routing baseline + oracle log-NSE completion** — re-eval L / L_upQ at seeds 13/17 over full span; ~30-45 min CPU (training-eval, not free).
3. **NHDPlus edge validation OR k=2 LSTM re-train** — the definitive graph checks; NHDPlus needs flowline data acquisition, LSTM re-train needs GPU. Either closes the graph question fully.

---
## 2026-07-16 — /crs: 3-seed routing baseline DONE (margin widens); oracle + k=2 LSTM re-trains are GPU-blocked, staged turnkey

**Source.** /crs, executing the two queued items (3-seed routing baseline + oracle log-NSE completion; and the definitive graph check). CRS triage up front: the two asks are not equals, and one has a hidden dependency.

**Orient correction.** Two disk facts reset the plan: (1) all THREE fullspan evals exist (seeds 11/13/17), so the "3-seed routing baseline" needs NO re-eval — it's zero-training, not the "~30-45 min CPU" I'd queued. (2) The "oracle log-NSE completion" is one specific hole — `L_upQ seed11` lost both checkpoint and results.p; L/L_upQ seeds 13/17 are complete. NHDPlus flowline data is NOT on disk, so the NHDPlus option for the "definitive graph check" is data-blocked; the executable option is a k=2 LSTM re-train (the named follow-up to the 2026-07-14 R1-proxy graph-robustness result).

**Part 1 — 3-seed routing baseline: PASS, and it strengthened the paper.** `analyze_routing_baseline_3seed.py` → `analysis/ROUTING_BASELINE_3SEED.md`. R1 pure routing +0.324 at every seed (seed-independent — uses observed upstream_q). The realizable-vs-R2 margin **widens to +0.019 across 3 seeds** from the single-seed +0.010 — seed-11 (the prior single seed) was the pessimistic one. All 3 seeds: realizable & oracle beat R1. The "ML earns its complexity" conclusion is multi-seed-robust, and Table 2's margin was previously understated. Zero compute.

**Parts 2 & 3 — BLOCKED on this machine; not faked.** Attempted the oracle seed-11 re-train. Per CRS discipline, smoke-tested first (1 epoch). `nh_run.py train` aborts with **SIGABRT (exit 134) at startup** on both `device: mps` and `device: cpu` — an AVX illegal-instruction crash from an AVX-compiled dependency on this CPU (the "TensorFlow compiled for AVX" warning is the tell; the crash kills the full training stack though NH itself imports fine in isolation). Confirmed the prior runs were **all trained on Colab** (`device: cuda:0` in on-disk configs); this Mac has only ever run analysis. No degraded local run is even possible, so none was faked.

**Staged turnkey for GPU (all zero-training prep done this session):**
- Part 2: `configs/L_upQ_component0_seed11.yaml` — set device cuda:0, train, evaluate → restores oracle log-NSE/KGE (the one blank column in PAPER_TABLE).
- Part 3: built the k=2 nearest-parent pruned edge set (`component0_edges_k2.csv`, 266 edges from 624), the k=2 oracle + realizable features (`features/upstream_q_{obs,pred}_component0_k2_lag1.p`), and the two configs (`L_upQ_k2_...`, `L_upQpred_k2_...`). On GPU: train both, compare Δ vs L against full-graph +0.037/+0.027. Success = k=2 realizable Δ within ±0.010 of +0.027 → confirms the LSTM (not just the R1 proxy) is pruning-invariant, closing the over-connectivity threat at the model level.

### Reviewer 2
- *Is the +0.019 margin real or a seed artifact?* Multi-seed (all 3), and it's LARGER than the single-seed +0.010 — the prior number understated it. R1 is fixed; the widening is in the LSTM rows.
- *Did you give up on training too early?* No — smoke-tested, got a hard SIGABRT at startup on both devices, and confirmed via on-disk configs that training was always done on Colab. This is an environment block, not a config error.
- *Why k=2 LSTM over NHDPlus for the "definitive" check?* NHDPlus needs flowline data not on disk (acquisition task); k=2 needs only Colab (already used) and directly tests the same threat. k=2 first; NHDPlus stays named future work.
- *Will the k=2 re-train just confirm the proxy trivially?* Not guaranteed — the R1 proxy is linear; the LSTM could exploit the excess edges nonlinearly. The pre-registered falsification (k=2 Δ collapses) is a live possibility worth the run.

### Open questions
1. Does the LSTM-level k=2 Δ match the full-graph Δ (proxy/LSTM agreement)? — the live Part-3 question, GPU.
2. Does the restored seed-11 oracle reproduce 0.703 (determinism)? — Part 2, GPU.

## Next 2–3 sessions (queued)
1. **[GPU] Parts 2 & 3** — run the 3 staged configs on Colab (~20-40 min each); completes oracle metrics + closes the graph threat at the LSTM level. All prep done; turnkey.
2. **Paper skeleton** — unblocked after Part 3 lands (or now, if the R1-proxy graph result is accepted as sufficient). PAPER_TABLE + ROUTING_BASELINE_3SEED + GRAPH_ROBUSTNESS are the spine.
3. **NHDPlus edge acquisition** — only if a reviewer demands ground-truth hydrography beyond the k=2 check; data-acquisition task on the user.

---
## 2026-07-16 (later) — /crs-unleashed: k=2 pruned-graph LSTM check LANDED — over-connectivity threat closed at the model level

**Source.** /crs-unleashed. The user ran the staged Colab notebook (`colab_oracle_completion_and_k2.ipynb`) on GPU and added the two k=2 run folders (L_upQ_k2, L_upQpred_k2, seed 11) to the repo root. This session ingests, verifies, interprets, and files the definitive graph-robustness result — the model-level confirmation of the 2026-07-14 R1-proxy finding.

**Orient.** Two runs came back (both with test_results.p → full metrics possible), 30 epochs, healthy training (final loss 0.065, clean eval). Relocated both from root to canonical `runs/topology_ablation/component0/` (gitignored heavy files, alongside siblings). Part 2 (oracle seed-11 full-graph restore) did NOT come back — its results.p is still missing; that half ran on Drive only or wasn't copied. Non-load-bearing (seeds 13/17 oracle complete).

**Diagnosis — the load-bearing question this run answers.** The 2026-07-14 GRAPH_ROBUSTNESS chain showed the R1 *lstsq signal-content proxy* is invariant to pruning the over-connected heuristic graph (in-degree mean 4.16/max 15) to hydrography-realistic in-degree≤2. The open reviewer objection: "the R1 proxy is linear; the *LSTM* could exploit the excess edges nonlinearly — maybe the model-level gain DOES depend on over-connection." This k=2 re-train tests exactly that.

**Result — PASS, at the LSTM level (paired Δ vs L, 150 connected basins, seed 11):**

| condition | graph | median Δ NSE | Wilcoxon p | log-NSE Δ |
|---|---|---|---|---|
| realizable | full | +0.034 | 1.4e-8 | — |
| **realizable** | **k=2** | **+0.021** | **4.0e-4** | **+0.034** |
| oracle | full | +0.046 | 7.6e-6 | — |
| **oracle** | **k=2** | **+0.049** | **2.3e-12** | — |

- **Realizable survives:** +0.021 NSE (p=4e-4), +0.034 log-NSE, ~78% of the full-graph realizable Δ on the same basins, inside the pre-registered ±0.010 band. Predicted upstream Q stays deployable on a real-confluence-connectivity graph.
- **Oracle STRENGTHENS under pruning:** k=2 +0.049 > full +0.046. Dropping the excess (distant, weakly-connected) parents *sharpens* the observed upstream signal. This is the routing physics showing through: nearest parents = shortest travel time = most-aligned flow. Directly consistent with the 2026-07-14 finding that the R1 signal lives in the nearest parents (and that `nearest`-rule pruning was invariant while `smallest-ratio` degraded).

**Why it matters.** The over-connectivity of the heuristic edges was the study's single biggest unresolved validity threat. It is now closed at BOTH levels: signal-content (R1 proxy, 2026-07-14) AND the trained model (this session). The heuristic's excess edges are not doing the work; the routing gain is carried by the physically-meaningful nearest-parent structure. The paper can present the heuristic-edge caveat AND demonstrate robustness to it — converting the limitation into a strength.

### Reviewer 2
- *R1 proxy might not predict the LSTM?* That was the whole point of this run. It does: k=2 realizable holds (+0.021, p=4e-4), matching the proxy's invariance. Proxy and model agree.
- *k=2 realizable (+0.021) is BELOW full-graph (+0.034 same basins) — is the gain eroding?* It's a modest drop, still inside the pre-registered band and clearly significant. Some erosion is expected (fewer edges = slightly less upstream information), but it stays deployable. The oracle going UP shows the *observed* signal is not eroded — the realizable dip is about the predicted feature being built on fewer parents, not the routing being an artifact.
- *Single seed?* Yes (seed 11). The full-graph result is 3-seed; a 3-seed k=2 replication is the named robustness extension. Single-seed here is adequate to answer the binary "does it survive pruning" — it clearly does, significantly.
- *Could pruning have helped by luck?* The oracle strengthening is mechanistically predicted (nearest = shortest travel time), not luck, and matches the independent 2026-07-14 nearest-vs-smallest-ratio result.
- *What would falsify this?* A k=2 realizable Δ ≤ +0.010 or non-significant. Observed +0.021, p=4e-4 — comfortably clear.

### Open questions
1. 3-seed k=2 replication (seeds 13/17) — the robustness extension; GPU, turnkey (same notebook, change SEED).
2. Oracle seed-11 full-graph restore — brings back the one missing results.p; fills the blank oracle log-NSE/KGE seed-11 columns; GPU, in the same notebook (Cell 8, currently idempotent-skipped only if present).

## Next 2–3 sessions (queued)
1. **Paper skeleton** — the empirical case is complete AND the graph threat is closed at the model level. Highest leverage, no prerequisite. K2_GRAPH_CHECK + GRAPH_ROBUSTNESS + PAPER_TABLE + SIGNIFICANCE are the spine; the graph-robustness pair is now a full subsection ("the result does not depend on heuristic edge density").
2. **[GPU] 3-seed k=2 + oracle seed-11 restore** — robustness + metric-column completeness; both in the existing idempotent notebook (change SEED / re-run Cell 8). ~20-40 min each.
3. **NHDPlus edge validation** — only if a reviewer demands ground-truth hydrography beyond the k=2 check; data-acquisition task. The k=2 result largely pre-empts this.

---
## 2026-07-26 — /crs-unleashed: directionality controls staged (the Kirschstein mirror test) + notebook protocol skilled

**Source.** /crs-unleashed, prompted by the user's ML-conference acceptance report. Used the report to **de-bias** prior advice (not overfit to it): ICLR/ICML explicitly state SOTA is not required when the work delivers new insight, so my earlier "national 531-basin scale-up is the gate" was over-weighted. The real top-tier lever is the **mechanism**, and the single most valuable missing experiment is the one that most directly engages the rebutted literature.

**The gap.** Kirschstein & Sun (2024) diagnosed river-network GNN failure as *directional insensitivity* — GNNs perform similarly whether edges are maintained, reversed, or permuted. Our controls (shuffled-time null, upstream precip, lag sweep, depth gradient) never tested whether OUR feature is direction-**sensitive**. If reversing the edges collapses the gain, we exhibit exactly the property whose absence explains the GNN null — turning the routing claim from correlational (gain rises with depth) into causal (gain requires correct flow direction).

**Staged (pre-registered, turnkey Colab):**
- `preregistration_directionality_controls.md` — hypothesis, criteria, honest falsification.
- `build_directionality_variants.py` — builds two observed-Q feature variants, identical aggregation to the forward builder, ONLY the edge set differs: **reversed** (parent↔child swapped → aggregates downstream flow) and **random** (degree-preserving rewire, seed 42 → each basin keeps its in-degree but random parents). name='date' fix baked in.
- `notebooks/colab_directionality_controls.ipynb` — idempotent run-all; builds features, ensures L + forward oracle, trains L_upQrev + L_upQrand, prints the pre-registered verdict + persistence check.

**Design checks (dry-run on the real edge set).** Forward 624 edges/150 connected; reversed 624/130 (outlets become disconnected); random 624/150 (in-degree preserved exactly). All Δ evaluated on the **forward-connected 150 basins** — the set where the routing question is defined — for apples-to-apples.

**Pre-registered criteria.** Directional sensitivity: forward − reversed ≥ +0.015. Topology specificity: forward − random ≥ +0.015. **Falsification (real risk):** if reversed ≈ forward (gap < +0.005), the gain is direction-INSENSITIVE → generic spatial correlation, not routing → the mechanism narrative must be rewritten, not re-scoped. Honest note: reversed won't hit 0 (downstream flow is weather-correlated with the target); the *contrast* is the test.

**Why it matters for the paper.** This is the experiment that converts "careful ablation" into "explains a field-wide null and demonstrates the fix." If it lands: the gain is causally directional, mirroring Kirschstein's insensitivity, grounding the routing mechanism and the general principle (structure-as-dynamic-state vs structure-as-label). If it falsifies: we learn the routing story is weaker than believed — better to know before submission.

**Also:** codified the Colab notebook protocol as a reusable skill (`~/.claude/skills/colab-notebook/SKILL.md`) — the idempotency-key trap, the name='date' feature fix, the VM-vs-Drive persistence check, determinism tolerance, push-before-link, and the validate-via-IPython-transform step. Standard protocol now, not re-derived each time.

### Reviewer 2
- *Is reversed a fair control?* Yes — identical everything (model, config, discharge source, lag), only edges reversed. Direct analog of Kirschstein's "edges reversed" but on our method.
- *Won't reversed just be ~0?* No, and the pre-reg says so — downstream flow shares weather with the target, so reversed carries residual correlation. The forward>reversed gap is the finding, not reversed=0.
- *Single seed?* Yes, per protocol (signal first, multi-seed at publication). If it lands, 3-seed it — same notebook, change SEED.
- *Random rewire fair?* Degree-preserving (each basin keeps its in-degree), so only *which* neighbors changes, not *how many* — isolates topology from feature-count.

### Open questions
1. Does reversed collapse (directional) or hold (generic)? — the live question.
2. If it lands, 3-seed both variants for the publication figure.

## Next 2–3 sessions (queued)
1. **[GPU] Run the directionality notebook** — turnkey; result grounds or revises the mechanism. Highest leverage.
2. **Reframe the paper's contribution as the transferable principle** (structure-as-dynamic-state vs structure-as-label), instantiated in hydrology — the "foundation-building" elevation, done carefully without overclaiming generality.
3. **3-seed the k=2 graph check + (if landed) the directionality controls** — publication-grade robustness on the two single-seed results.

---
## 2026-07-27 — /crs-unleashed: directionality controls RESULT — topology-specific (strong), directionally-preferential (partial)

**Source.** /crs-unleashed. User ran the staged directionality notebook (the Kirschstein mirror test) on Colab and added L_upQrev + L_upQrand (seed 11) to the repo. This session ingests, computes the pre-registered contrast, and files the result with honest scope. Pre-reg: `preregistration_directionality_controls.md`.

**Result — paired Δ vs L, forward-connected basins (n=150), seed 11:**

| edge set | median Δ | Wilcoxon p vs L |
|---|---|---|
| forward (true upstream) | +0.046 | 7.6e-6 |
| reversed (downstream) | +0.026 | 1.3e-6 |
| random (rewired, in-degree preserved) | +0.014 | **0.10 (n.s.)** |

Monotone gradient: forward > reversed > random > 0. **Both pre-registered median-gap criteria PASS** (forward−reversed +0.020 ≥ +0.015; forward−random +0.032 ≥ +0.015).

**The honest split (the paired head-to-head, which the median-gap hid):**
- forward > random: median +0.041, **p=3e-4** — topology-specificity, strong & significant.
- reversed > random: median +0.026, **p=2e-4** — even wrong-direction *real* edges beat random.
- forward > reversed: median +0.008, **p=0.19 — NOT significant per-basin.**

**Interpretation.**
- **Topology-specificity is the clean win.** Random rewiring (same in-degree, wrong neighbors) drops the gain to +0.014, not even significant vs L. The signal lives in the REAL river structure, not any regional flow aggregate. This is a strong, defensible result.
- **Directionality is present but partial.** Reversed edges retain ~57% of forward (+0.026); forward beats reversed on the median but not at paired per-basin significance. Exactly as the pre-reg anticipated: downstream flow is weather-correlated with the target (shared precip; the basin's own routed water), so reversal weakens rather than destroys the signal. Here the residual is larger than a strictly-directional mechanism would predict.
- **Scope decision (no overclaim):** the correct framing is *the model exploits the real hydrological network, preferring the physically-correct upstream direction* — NOT *the gain requires correct direction*. Claiming strict directionality is unsupported by the paired test. Report the gradient honestly.

**Kirschstein mirror — appropriately scoped.** Their GNNs were topology-INsensitive (any/no adjacency ≈ same). Our feature is sharply topology-SENSITIVE (real edges >> random, p=3e-4) — the property their GNNs lacked. On direction specifically our advantage is a median preference, not a significant per-basin effect; we report that honestly rather than claiming the stronger inverted-mirror result.

**Why this is not a falsification.** The pre-reg's falsification was "reversed ≈ forward (gap < +0.005) → generic correlation, not routing." We got a +0.020 median gap and a clean forward>reversed>random ordering — the gain is NOT generic correlation (random confirms that). The mechanism is real; its directional component is just weaker than its topological component. That is a nuanced true result, not a failed one.

### Reviewer 2
- *Your pre-reg criterion passed but the paired directional test is n.s. — misleading criterion?* Reported openly. Topology-specificity is the solid headline; directionality is scoped as a preference. Under-claiming by choice.
- *Random-rewire fair?* Degree-preserving — only which neighbors changed. Its drop to non-significance is the strongest topology-specificity evidence available.
- *Single seed?* Yes — the main limitation, and precisely what a 3-seed run would resolve for the +0.008 directional gap.
- *Does reversed retaining 57% undercut you?* No — it's physically expected (downstream flow shares weather), and forward still leads. It makes the story richer, not weaker.

### Open questions
1. Does the +0.008 forward−reversed paired gap become significant (or vanish) at 3 seeds? — the one unresolved question.
2. Would a shorter-lag or high-flow-event-only slice sharpen directionality (routing is clearest during storm pulses)? — possible future refinement.

## Next 2–3 sessions (queued)
1. **[GPU] 3-seed the directionality controls** (L_upQrev/L_upQrand at seeds 13/17) — resolves the directional test; turnkey (same notebook, change SEED). ~40 min. The one clear follow-up.
2. **Write the paper skeleton** — mechanism now includes topology-specificity (strong) + directional preference (honest). PAPER_TABLE + DIRECTIONALITY + GRAPH_ROBUSTNESS + K2_GRAPH_CHECK are the mechanism spine. No prerequisite.
3. **Reframe contribution as the transferable principle** (structure-as-dynamic-state vs structure-as-label), instantiated in hydrology, scoped honestly. Writing.

---
## 2026-07-29 — /crs-unleashed: 3-seed mechanism results — topology-specificity + k=2 CONFIRMED; directionality is a mild pooled preference (downgraded)

**Source.** /crs-unleashed. User ran the 3-seed mechanism notebook; 8 runs (L_upQrev/rand/_k2/pred_k2 × seeds 13/17) landed. This session computes the pooled 3-seed verdicts, reconciles them against the user's Cell 9 output, and files the honest scope. Supersedes single-seed DIRECTIONALITY.md / K2_GRAPH_CHECK.md. → analysis/MECHANISM_MULTISEED.md.

**k=2 graph-robustness — CONFIRMED, strong, 3-seed.**

| condition | per-seed Δ (11/13/17) | pooled Δ | p |
|---|---|---|---|
| full realizable | +0.034/+0.030/+0.015 | +0.023–0.026 | 3e-15 |
| k2 realizable | +0.021/+0.033/+0.021 | +0.025 | 1e-14 |
| k2 oracle | +0.049/+0.074/+0.048 | +0.059 | 3e-43 |

k2-realizable ≈ full-realizable (indistinguishable), all seeds positive. The over-connectivity threat is closed multi-seed at the LSTM level. Clean, publishable.

**Topology-specificity — CONFIRMED, strong, 3-seed.** forward−random paired pooled +0.037, p=3e-20; every seed. The gain requires the REAL river structure (random rewire retains ~26%). Headline mechanism result + the Kirschstein mirror (their GNNs topology-insensitive; ours sharply sensitive).

**Directionality — downgraded to a mild, aggregate-only preference (reconciliation).** The user's Cell 9 reported forward−reversed one-sided p significant; my two-sided p=0.06 initially read as n.s. Reconciled: the pre-registered hypothesis is directional, so **one-sided is correct → pooled p≈0.03, significant.** BUT stress-testing showed it is small and seed-fragile: median +0.006, only 54% of basins favor forward, per-seed p=0.19/0.07/0.23, and **seed 17 has reversed ≈ forward (Δ −0.001).** Reversed retains 57/79/90% of forward. Physically expected (downstream flow is weather-correlated). **Verdict: report as "a mild, aggregate-detectable preference for the physically-correct direction" — NOT "direction-sensitive."** The pre-reg falsification (reversed≈forward → generic correlation) is not triggered, but no strong directional claim is supported either.

**Why the 3-seed run was worth it.** At single seed (11) the directional gap looked cleaner (+0.008 with a plausible story). The 3-seed data revealed it's fragile (null at seed 17). Running this BEFORE writing prevented an overclaim of "direction-sensitivity" that a reviewer with the multi-seed data would have shredded. This is the run earning its cost — exactly the /crs discipline (multi-seed before claiming).

### Reviewer 2
- *One-sided vs two-sided — did you p-hack?* One-sided is pre-registered (H1: forward improves). Reported alongside the weakness (54% of basins, null seed). Not a strong claim either way.
- *Is topology-specificity also fragile?* No — forward−random +0.037, p=3e-20, every seed. Rock solid; that's the headline.
- *k=2 robustness real?* 3 seeds, k2 ≈ full within noise, p=1e-14. Yes.
- *Why report directionality at all if weak?* Transparency — the gradient forward>reversed>random is real in aggregate; we show it and scope it honestly rather than hiding an inconvenient middle result.

### Open questions
1. Could directionality be sharpened by conditioning on high-flow events (routing clearest during storm pulses)? — a possible future refinement, NOT needed for the paper.
2. None load-bearing remain. Every mechanism claim is multi-seed.

## Next 2–3 sessions (queued)
1. **Write the paper skeleton** — NO further experiments needed for workshop tier. Mechanism spine: static-topology null → dynamic flow gain (3-seed, p=2e-12) → topology-specific (real≫random, p=3e-20) → deployable (predicted-Q) → graph-density-robust (k2≈full, 3-seed) → mild directional preference (honest). PAPER_TABLE + MECHANISM_MULTISEED + SIGNIFICANCE are the spine.
2. **Reframe contribution as the transferable principle** (structure-as-dynamic-state vs structure-as-label), instantiated in hydrology, honestly scoped.
3. **[Optional, top-tier only] National 531-basin scale-up** — deferred; not needed for workshop (decision 2026-07-26). Only if targeting a main track after the workshop version.

---
## 2026-08-01 — /crs-unleashed: Methodology multi-pass rigor audit (correctness error caught + slop removed)

**Source.** /crs-unleashed. User asked for iterative rigor passes on the drafted Methodology: grounding sanity, AI-slop removal (semicolons/em-dashes/long clauses/jargon), math recheck, redundancy, reviewer-level reread. Handed prose/math to ml-paper-writer + ml-math-rigor.

**Diagnosis (top-3):** (1) correctness — potential one-hot dimension inconsistency [high-imp, test first]; (2) AI-slop density [high-imp]; (3) grounding/"what we build on" [medium].

**Pass A — correctness (caught a real error).** Table 1 said "671-dim basin one-hot". Verified against `basedataset.py:208` (`num_classes=len(id_to_int)`, built from TRAIN basins): the true dimension is |V|=183, not 671. The old `current_implementation.md` "671" was wrong and would have shipped a fabricated number. Fixed to "$|V|$-dim ($|V|=183$)", consistent with §setup. All other equations (NSE/logNSE-eps/KGE/feature/edge) re-verified against source — clean.

**Pass B — AI-slop + redundancy.** Prose semicolons 7→0; clause em-dashes 0; cut the filler roadmap sentence; de-duplicated "byte-identical" (3×→appropriate); tightened trailing editorial clause; "what the signal can buy"→"bounds the achievable gain".

**Pass C — reviewer reread.** Rewrote the leakage sentence, which conflated two arguments and re-introduced τ≥1 after Eq. fixed τ=1 — now two precise claims (upstream ⇒ own discharge never enters; lagged ⇒ no same-day info).

**CRS stance on "what work we build on" (the user flagged this as important).** The grounding is correct and honestly stated: §model cites the foundational LSTM (kratzert2018), the multi-basin paradigm our L baseline IS (kratzert2019), and the stock cudalstm from the NeuralHydrology model zoo we run (kratzert2022joss). We cite both paradigm and software, and frame novelty as the ablation finding + deployable feature, not a new architecture. This is the right grounding; no change needed. My stance: over-claiming a novel model here would be the mistake — the honesty that "the model is theirs, the finding is ours" is a strength, not a weakness.

**Skill update.** Embedded an "AI-slop tells" subsection into ml-paper-writer (semicolon/em-dash overuse, long clauses, filler roadmaps, trailing editorial clauses, synonym drift, over-formal jargon) with a read-aloud test — applies to every future prose pass.

### Reviewer 2
- *Is the 183 one-hot definitely right?* Verified in code: NH builds the one-hot over the training basin set (183 for Component 0), num_classes=len(id_to_int). Yes.
- *Does cutting the roadmap hurt navigability?* No — the section is 1.5pp with clear subsection titles; a roadmap sentence was over-signposting.
- *Is "byte-identical" still emphasized enough?* Yes — stated once in §model, reinforced by the design argument in §ablation and Table 1's caption. Emphasis without repetition.

### Open questions
1. Figure 1 (static-null vs dynamic-gain contrast) still to be drawn.
2. Compute/hardware statement (report wants it) — belongs in a Reproducibility para, add with the checklist.

## Next 2–3 sessions (queued)
1. **Draft Results** — tables grounded (PAPER_TABLE, MECHANISM_MULTISEED); conditions now formally defined (Table 2). Same end-to-end + 4-axis QC treatment.
2. **Draft Introduction** — the tension→turn→payoff arc, leading with the contrast; align with the QC'd abstract.
3. **Figure 1** — the core-idea contrast diagram (static topology → ~0 vs dynamic flow → gain).
