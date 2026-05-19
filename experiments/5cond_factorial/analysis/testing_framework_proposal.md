# Testing Framework Proposal — How to Test "Our Features Beat Standard LSTM"

**Date:** 2026-05-12
**Goal:** A diagnostic ladder + experimental protocol that can credibly support (or falsify) the paper claim *"our added graph features outperform a standard LSTM on discharge prediction."* Built to expose confounds, not hide them.
**Companion to:** `5cond_run_analysis.md` (what we found) and `architecture_analysis.md` (why).

---

## 1. Why we need a new framework

The current 5-condition factorial is **internally** valid (paired contrasts, multi-seed, bootstrap CIs, the whole stats stack) — but it answered the *wrong* question. It answered "what does our DirectedGraphLSTM trainer + variants produce on Component 0?" — not "does the graph signal add value when fairly compared to a standard LSTM?"

The two problems that broke the inference:

| Problem | Source | Fix |
|---|---|---|
| **Training-budget confound:** L gets 186× more gradient updates than G | Per-window batching forces all 183 basins into one sample | Below: **Step 1** — matched-budget control |
| **Inductive-bias confound:** basin one-hot encoding subsumes the topology features | NH's `use_basin_id_encoding: True` default | Below: **Step 2** — one-hot-off ablation |

A "do our features help" framework has to *isolate* each effect cleanly. The proposal below is a **diagnostic ladder**: each step is cheap, each step gates the next, and each step has a written success/falsification criterion.

---

## 2. Design principles

1. **Pre-register before running.** Every step has its hypothesis, success criterion, and falsification criterion written *before* the data is seen. No re-fitting hypotheses to results.
2. **One factor at a time, then factorials.** Change one variable per step. Once each factor is isolated, *then* run a factorial over the surviving subset.
3. **Always include the strongest baseline.** NH cudalstm at full strength stays in every comparison — it's the publishable reference.
4. **Always include the architecture-matched control.** The DirectedGraphLSTM-with-no-graph (G) tells us what the trainer + architecture can do on its own, holding the architecture constant.
5. **Three seeds minimum; five for publication.** Below three is anecdotal.
6. **Paired per-basin contrasts, not just medians.** Two models with identical median NSE can disagree on every basin (we've seen this in earlier ensemble experiments — correlations 0.994 between variants).
7. **Validation set distinct from test.** No best-checkpoint selection on test data, ever.
8. **Compute-budget honest.** Each step has a stated compute cost; don't over-promise.

---

## 3. The diagnostic ladder

Six steps, each pre-registered. **Stop and re-plan at the first step that fails its falsification criterion.** Do not skip ahead.

### Step 1 — Training-budget control

**Pre-registration**
- **Hypothesis:** The current L − G gap of +0.050 NSE is dominantly a training-budget confound. If L is constrained to the same gradient-step budget as G, the gap closes to within 0.01 NSE.
- **Setup:**
  - Train NH cudalstm for **only 420 gradient updates total** (matching what the graph trainer got from 30 epochs of batch=256 over 3,652 windows). Achievable by setting `epochs: 1` and `batch_size: 4382` (≈ 183 × 24, so each step uses 24 days × 183 basins = analogous to graph trainer's per-window). Or, the cleaner approach: use NH's normal pipeline but stop after 420 update steps and report NSE.
  - 3 seeds {11, 13, 17}, Component-0 basin set.
  - Otherwise identical to current L config.
- **Success criterion:** `(L_matched_budget median NSE) − (G median NSE) ≤ +0.01` AND CI excludes both +0.03 and −0.03.
- **Falsification criterion:** `(L_matched_budget) − (G) ≥ +0.03 NSE`. If L still beats G by 3 NSE points at matched budget, there is a real architectural component to the gap.
- **Cost:** ~5 min per seed on T4 (420 gradient steps total). 3 seeds ≈ 15 min.
- **Output:** one row in the headline table — "L (matched budget): NSE 0.X ± 0.0Y". Settles the architecture-confound debate.

**Decision tree:**
- If success: training budget is the entire L − G story. Continue to Step 2; G's "true" performance (with proper training) is unknown but ≥ matched-budget L.
- If falsified: there IS a real architecture gap. Continue to Step 3 (which still needs to happen regardless), but also flag a separate investigation into the architecture-side of L − G.

### Step 2 — Basin-encoding ablation

**Pre-registration**
- **Hypothesis:** The G+T variant adds zero signal because the 671-dim basin one-hot encoding subsumes the 5 topology features. With one-hot OFF, G+T should show a real lift over G.
- **Setup:**
  - Train G and G+T with `use_basin_id_encoding: False` in the NH config that the graph trainer pulls. 3 seeds. 30 epochs (or matched budget per Step 1's lesson).
  - Comparison: G_no_onehot vs G+T_no_onehot (paired per-basin).
- **Success criterion:** Median paired Δ (G+T_no_onehot − G_no_onehot) ≥ +0.01 NSE AND CI excludes zero.
- **Falsification criterion:** Δ remains ≤ +0.005 (i.e., topology features still null even without the redundancy hypothesis).
- **Cost:** 6 runs × ~1 hr each on T4 (more if Step 1 found we need extended training). ≈ 6 hr.
- **Output:** answers whether the topology features have *any* signal, or whether they're poorly designed regardless of one-hot.

**Decision tree:**
- If success: topology features are a real contribution when properly used. Either (a) keep one-hot and accept that G+T = G in our actual setup, OR (b) drop one-hot and have G+T > G as the paper claim.
- If falsified: topology features are intrinsically weak. Need redesign per `architecture_analysis.md` §2 (embed discrete topo features, replace network-relative with absolute, etc).

### Step 3 — Message-passing diagnostic (does it learn anything?)

**Pre-registration**
- **Hypothesis:** The current message-passing implementation produces near-zero effective gradient flow into the graph component (`W_out`, `W_msg_edge`). The model never actually *uses* the graph signal regardless of how informative it could be.
- **Setup:**
  - Take a converged G+M run (or train one to convergence per Step 1's lesson).
  - Measure per-epoch:
    - `||W_out||_F` (Frobenius norm of W_out matrix) — should grow from 0 if the graph path matters.
    - `||∇W_out||_F / ||∇LSTM||_F` — gradient norm ratio.
    - `corr(h_parent, h_child)` averaged over edges — does the model align parent and child representations?
  - Also: shuffle edges (random pairing) and re-train. If random edges give the same NSE as real edges, the model isn't using the graph structure.
- **Success criterion (graph IS being used):** `||W_out||` grows to ≥ 1% of `||LSTM weights||` by epoch 15; random-edges NSE ≤ real-edges NSE by ≥ 0.005.
- **Falsification criterion (graph is dead-on-arrival):** `||W_out||` stays < 0.1% of LSTM norms throughout training; random-edges and real-edges have indistinguishable NSE.
- **Cost:** 2 trainings (real + shuffled edges) × 3 seeds ≈ 6 hr T4. Plus the diagnostic plotting (CPU, free).
- **Output:** binary answer to "is the message-passing dead?" If yes, no amount of "more training" or "better edges" will help — needs redesign first.

### Step 4 — Message-design A/B (only if Step 3 says graph is alive)

**Pre-registration**
- **Hypothesis:** Replacing mean aggregation with area-weighted aggregation produces a measurable lift on basins with high in-degree (where mean is most biased).
- **Setup:**
  - Two variants of G+M:
    - `G+M-mean` (current): unweighted mean over parents.
    - `G+M-area`: `m_v = sum_u (parent_area_u / total_parent_area_v) · msg_u`.
  - 3 seeds each. Stratify results by in-degree (compute per-basin Δ NSE within in-degree buckets {1, 2, 3-4, 5+}).
- **Success criterion:** Median Δ NSE for in-degree ≥ 3 basins is +0.005 or larger; overall median Δ ≥ 0.
- **Falsification:** No improvement at any in-degree level.
- **Cost:** 6 runs × 1 hr ≈ 6 hr T4.
- **Output:** validates (or rejects) the area-weighting design improvement.

### Step 5 — The clean 5-condition factorial (re-run)

**Pre-registration**
- **Hypothesis:** With the corrections from Steps 1–4 applied, G+T+M outperforms L (full-budget cudalstm) by a small but real margin (+0.005 to +0.020 NSE).
- **Setup:**
  - L (cudalstm full-budget) — unchanged from 5cond.
  - G, G+T, G+M, G+T+M — all with (a) matched training budget to L (if Step 1 said budget matters), (b) area-weighted aggregation (if Step 4 said it helps), (c) appropriate basin-encoding setting (per Step 2).
  - 5 seeds {11, 13, 17, 19, 23}.
  - 30 epochs at matched per-step budget, OR until validation NSE plateaus for 3 consecutive checks.
- **Success criterion:** Median paired Δ (G+T+M − L) ≥ +0.005 NSE; CI excludes zero on the positive side.
- **Falsification criterion:** Δ remains ≤ 0 even after all corrections from Steps 1–4.
- **Cost:** 5 conditions × 5 seeds = 25 runs × ~1 hr each on T4 ≈ 25 hr. One large Colab session or split across 2.
- **Output:** the paper's headline numbers.

### Step 6 — Robustness checks (publication-quality)

If Step 5 hits success:
- **Seed expansion:** 10 seeds total per condition (gives tighter CIs for the headline).
- **NHDPlus edges:** replace heuristic edges with ground-truth NHDPlus edges; re-run G+M, G+T+M. Should be ≥ the heuristic-edges result.
- **Holdout component:** train on Component 0, test on a held-out smaller component (e.g., Texas 23-basin pilot). Out-of-network generalization.
- **Time generalization:** train 1990–1999, test 2010–2018 (10-year shift). Drought-flood-pattern generalization.

These are nice-to-haves for revision; the core paper claim sits on Step 5.

---

## 4. Statistical protocol (carried over from `compare_5conditions.py`)

The current analysis script (`experiments/analysis/compare_5conditions.py`) is already publication-quality for the contrasts we need. Specifically:
- Three metrics (NSE, KGE, log-NSE) computed identically from raw predictions across all conditions.
- Bootstrap 95% CIs for cross-seed medians (n_boot=2000).
- Paired per-basin contrasts with bootstrap CIs.
- Interaction term computation.
- Depth- and area-stratified plots.
- Outlier-trimmed (bottom 5%) sensitivity.

The only additions needed for the new framework:
- **Per-basin attribute joins** (drainage area, mean elevation, aridity index from CAMELS attrs) so we can do additional stratifications.
- **Time-segmented evaluation** (per-year NSE) so we can report low-flow vs high-flow season behavior.
- **A "delta vs in-degree" plot** for Step 4 specifically.

These are 1–2 hours of analysis script work, not a redesign.

---

## 5. Pre-registration hygiene

For each step:
1. Write the pre-registration into a file (e.g., `experiments/5cond_factorial/preregistration_stepN.md`) BEFORE launching any runs.
2. Commit it to git so the timestamp is provable.
3. After the run completes, write the result + interpretation into the same file (append-only, dated).
4. If a re-design is needed, write a *new* pre-registration with explicit text "this supersedes preregistration_stepN.md" — never edit the original.

This guards against the most insidious failure mode: re-running until you get the result you want and reporting only that run. Especially important for Step 5, where 25 runs × 3 seeds = 75 trainings is a lot of randomness to dredge through.

---

## 6. What this framework deliberately does NOT do

1. **No "train longer until it works" without a pre-stated budget.** Step 1 sets the budget; Step 5 stays within it.
2. **No new metrics introduced post-hoc.** NSE, KGE, log-NSE are the metrics; if a new metric shows a positive result, that's a separate hypothesis requiring its own pre-registration.
3. **No basin filtering** (e.g., "drop the 20 worst basins"). Every step reports on the full Component-0 set.
4. **No ensembling.** Each condition's headline number is from a single training per seed. Ensembling can be a follow-up but is not part of the core claim.
5. **No pivot to a different dataset** if Component 0 doesn't cooperate. We pre-committed to Component 0 in the 5cond design; the paper either supports its thesis on Component 0 or it doesn't.

---

## 7. Compute and timeline estimate

| Step | Runs | Cost (Colab T4 hr) | Cumulative |
|---|---|---|---|
| Step 1 (matched-budget L) | 3 | 0.25 | 0.25 |
| Step 2 (one-hot off ablation) | 6 | 6 | 6.25 |
| Step 3 (graph-alive diagnostic) | 6 + analysis | 6 | 12.25 |
| Step 4 (area-weighted aggregation) | 6 | 6 | 18.25 |
| Step 5 (clean 5cond factorial × 5 seeds) | 25 | 25 | 43 |
| Step 6 (robustness; if Step 5 succeeds) | 30+ | 30 | 73 |

Total compute through Step 5: **≈ 43 hr on T4**, or **≈ 12 hr on L4** (3.5× faster), or **≈ 5 hr on A100** (8× faster).

At Colab Pro (100 units/month, T4 ≈ 1.5 units/hr): Step 1–5 = ~65 units. Affordable in one month with room for unexpected reruns.

---

## 8. The decision diagram

```
                  Step 1: Train budget the problem?
                    /                              \
                   YES                              NO (residual architecture gap)
                    |                                |
                Continue to Step 2                  Flag separate investigation;
                    |                                still continue to Step 2
                    |                                |
                  Step 2: Does T help w/o one-hot?
                    /                              \
                   YES                              NO
                    |                                |
              Step 3-5 with one-hot OFF         Topology features need redesign
                                                    (per arch_analysis §2);
                                                    drop T from the headline claim
                    |
                  Step 3: Is the message path alive?
                    /                              \
                   YES                              NO
                    |                                |
              Step 4-5 with current MP        Redesign MP per arch_analysis §3
                                                    BEFORE Step 4 / 5
                    |
                  Step 4: Does area-weighting help?
                    /                              \
                   YES                              NO
                    |                                |
              Step 5 with area-weighted        Step 5 with mean aggregation;
                  aggregation                       narrative depends on whether
                                                    G+T+M > L still emerges
                    |
                  Step 5: G+T+M > L?
                    /                              \
                   YES                              NO
                    |                                |
              Step 6 + paper writeup            Pivot to Narrative C
              (Narrative A from arch_analysis)  ("clean negative result")
                                                from arch_analysis §7
```

---

## 9. Why this framework is honest

The current 5-condition design said "we eliminate the architecture confound by running everything in DirectedGraphLSTM." The new framework says "we eliminate the *training-budget* confound by giving L the same step budget as G, and we eliminate the *inductive-bias confound* by testing the one-hot subsumption hypothesis explicitly."

Both confounds were unknown to us when we designed 5cond. We are now eyes-open about them.

The framework is honest in that it has **two failure modes baked in** (Step 5 falsification → pivot to negative result; Step 3 falsification → fundamental redesign required) and does not pretend the paper-claim direction is the only acceptable outcome.

It is also honest about its limitations: it does not test against the wider ML4hydrology literature (Kratzert 2022, Kirschstein 2024, etc.) head-to-head, only against our own cudalstm baseline. A revision pass for publication would add those comparisons.

---

## 10. Next concrete actions

1. **Today:** read this framework + `architecture_analysis.md` + `5cond_run_analysis.md` end-to-end. Decide whether to commit to Step 1 or pivot first.
2. **This week (after decision):** write `preregistration_step1.md`, push to git, launch Step 1 on Colab (15 min compute).
3. **Next week:** based on Step 1 outcome, write `preregistration_step2.md`, launch (6 hr compute).
4. **Two weeks out:** Step 5 ready to launch if Steps 1–4 cleared their gates.
5. **One month out:** paper draft skeleton with whichever narrative the data supports.

If at any step the data points the other way, follow the decision diagram. Document the pivot in `JOURNAL.md`. Do not retrofit hypotheses.
