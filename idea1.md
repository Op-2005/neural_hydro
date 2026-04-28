# Idea 1 — Does river-network topology destabilize the LSTM's self-equilibrium enough to help?

**Master file. Current active direction. Last updated 2026-04-21 post-PI-meeting (see `JOURNAL.md` 2026-04-21 entry for the reframing trigger).**

---

## One-sentence version

Treat the multi-basin LSTM as a learned dynamical system that drifts toward a
self-consistent attractor; test whether river-network topology supplies an
**external forcing** that breaks the LSTM's self-stabilization in a way that
recovers real physical dynamics — and whether that forcing is delivered better
as static topological identity or as runtime inter-basin message passing.

## The research question (plain)

*"Does river-network topology supply external forcing that breaks the
LSTM's self-stabilizing dynamics? If yes, is the destabilization delivered
better as a static signal about where the basin sits in the network, or as
runtime hidden-state messages from upstream neighbors?"*

This is the reframing of the original "does topology help streamflow
prediction?" — anchored in dynamical-systems-on-networks language that the
PI recommended on 2026-04-21. The pilot's +0.013 frozen-graph NSE is now
read as *the small, real destabilizing-forcing effect*, and the +0.065 of
LSTM weight drift during joint training is read as *the LSTM finding a new
self-consistent attractor that incorporates the forcing more deeply* —
not a confound, but a different mode of the same mechanism.

## Why this question is worth asking

**Dynamical-systems framing (PI, 2026-04-21).** The trained LSTM, when
rolled out on a basin's time series, tends to settle into a self-consistent
attractor where its own hidden-state dynamics dominate the prediction.
Unless an external forcing breaks that self-consistency, the model "drives
itself" rather than tracking real physical drivers. This is consistent
with what we already saw on the 23-basin pilot — the LSTM-drift component
of the +0.078 headline (≈ +0.065 of it) is the model finding a new
self-stable trajectory that happens to incorporate the graph signal,
rather than passing graph information through cleanly. **Treating the
problem as a dynamical system on a graph reframes "does topology help" as
"what graph topologies admit external forcings strong enough to
destabilize the LSTM's self-consistent regime in a useful direction?"** —
a sharper, more verifiable question.

**Physical intuition.** Water flows downstream. A basin's discharge at time
`t` is physically a function of its own recent precipitation *and* the
discharge of its upstream neighbors hours-to-days earlier. Standard
multi-basin LSTMs (Kratzert 2019, the field's reference baseline) are
trained on every basin's time series *independently* during each forward
pass — two basins that sit on the same river get the same treatment as two
basins on opposite coasts. That seems like it ought to leave signal on the
table.

**Literature status.** Whether it actually leaves signal on the table is an
open empirical question with mixed published evidence:
- **Kratzert et al. 2019 (HESS):** Multi-basin LSTM with static attributes
  matches or beats physics-based models on CAMELS-US — establishing the
  strong baseline that any topology-aware method must beat.
- **Kirschstein & Sun 2024 (ICML):** GNNs on LamaH-CE river networks showed
  **no** benefit over a plain MLP, even with multiple adjacency definitions.
  The null result has not been fully explained.
- **Jiang et al. 2025 (ICML):** A specifically-designed directional operator
  + physics-regularized loss recovers a +31.6% gain on a river dataset,
  diagnosing Kirschstein's failure as low-pass filtering destroying
  high-frequency flow signals.

**The gap we address.** No one has run a **clean ablation at scale** that
separates the two distinct channels through which topology could help:
(1) *static identity* — "my graph position tells the LSTM what kind of basin
I am," and (2) *runtime information flow* — "my upstream neighbor's current
hidden state tells me something my own forcings do not." Our 23-basin pilot
(see `INSIGHTS.md`) suggested (1) and (2) overlap heavily, but n=23 with
heuristic edges was too small to be definitive. The scaled three-way
ablation below is designed to answer this at a statistical-power level we
can publish.

**Why it's worth a workshop paper either way.** The ablation result is
informative in all four directions:
- If C beats B by a margin → message passing is doing real work → positions
  against Kirschstein's null with a specific mechanism.
- If B ≈ C → topology is an identity signal, not a communication signal →
  consistent with Kirschstein but gives a clean explanation of *why*.
- If B ≈ A → topology as a static feature is redundant with basin encoding
  → negative result with a precise scope.
- If C > A but B > A too → both channels contribute; rare-but-publishable.

## Plan

The plan is now staged. **E0 and E0.5 are gates** — the rest of the program
only proceeds if their results are consistent with the dynamical-systems
framing. If they aren't, the framing is wrong and we revisit (the pilot's
A/B/C ablation still stands as a backup deliverable, just framed differently).

### E0 — Verify LSTM self-stabilization (gate experiment)

**Question.** Does our trained baseline LSTM actually exhibit
self-stabilizing behavior on held-out time series? If not, the entire
"destabilizing forcing" framing collapses.

**Test design (two complementary probes).**
1. *Hidden-state perturbation recovery.* At a random test-period timestep,
   add Gaussian noise to the LSTM's hidden state. Continue the rollout.
   Measure how many timesteps until the perturbed trajectory rejoins the
   unperturbed one (in hidden-state L2 distance and in output prediction
   space). Self-stabilization predicts: fast convergence back to the
   unperturbed trajectory.
2. *Forcing-replacement test.* Replace the basin's true forcing
   (precipitation, temperature, etc.) at some test timestep with the
   *previous day's* forcing repeated. Measure how prediction error grows
   vs. a control. If self-stabilization dominates, prediction stays
   plausibly close (LSTM ignores the wrong forcing); if external forcing
   dominates, prediction immediately tracks the wrong input.

**Success criterion.** Both probes show self-stabilization signatures
(perturbations decay; misforcing has bounded effect) on the run-05 baseline
on at least 50% of test basins.

**Cost.** Cheap. Reuses run-05 weights + existing test data. CPU-feasible
in an afternoon.

### E0.5 — Loss-saturation test (gate experiment)

**Question.** Has the pilot baseline saturated its training-loss floor? If
not, our "marginal gains" interpretation is partly an under-training
artifact.

**Test design.** Re-train the run-05 strong baseline for 60 epochs (vs the
pilot's 30) with the same config, save loss curves on both train and
validation. Plot.

**Success criterion (interpretation key).**
- *Saturated* (val loss flat for ≥ 15 epochs): scaling laws apply, scaling
  is the right move.
- *Still descending*: pilot results were under-trained; some of our
  "marginal" effect attributions need re-checking.

**Cost.** ~20 min of CPU on the 23-basin network. Trivial.

### The ablation — three conditions, same LSTM backbone

(Unchanged from the pre-PI plan; reinterpreted under the new framing.
Each condition tests a different *channel of external forcing.*)

All three use the NH-framework LSTM (CudaLSTM) with basin ID encoding, trained
under identical hyperparameters on the same basin set.

| Condition | What it has | What it tests |
|---|---|---|
| **A. LSTM baseline** | Standard multi-basin LSTM + basin ID encoding (Kratzert 2019 style). No topology. | The strong published-style baseline. |
| **B. LSTM + topology-as-features** | 5 topology-derived scalars appended to the static attribute vector (see list below). **No message passing between basins.** | Does knowing your network position help, without any runtime communication? |
| **C. LSTM + topology + message passing** | DirectedGraph-LSTM: lagged upstream-parent hidden states aggregated as messages, edge features, warm-started from A. | Does runtime information flow between basins add anything beyond position-as-features? |

**What each margin tells us**

- **B − A** : the value of *topological identity* as a static signal
- **C − B** : the value of *runtime message passing* specifically
- **C − A** : total effect of topology + message passing combined

The pilot 23-basin result (C − A = +0.078 NSE median) collapsed largely into
B-like territory (graph ≈ basin encoding). Scaling and adding B as an explicit
condition resolves this ambiguity.

**Concrete Condition B feature list** (the 5 topology-derived scalars
appended to each basin's static attribute vector):

1. **Graph depth** — longest directed path length from any root to this
   basin. 0 = headwater.
2. **In-degree** — number of CAMELS basins immediately upstream.
3. **Out-degree** — number of CAMELS basins immediately downstream.
4. **Transitive upstream count** — number of CAMELS basins strictly upstream
   (all depths), divided by the network size for scale-invariance.
5. **Upstream-area ratio** — sum of upstream basins' areas divided by this
   basin's own area. Approximates "how much of my flow arrives from
   upstream." Log-transformed, z-score-normalized.

All five are computed from the same edge list used by Condition C, so A/B/C
rest on the same topology definition. They are *static* scalars (computed
once per basin) and are concatenated into the same static attribute vector
the LSTM already consumes — no architectural change required to the NH
CudaLSTM.

### Forcing-comparison condition (PI ask, 2026-04-21)

A separate sub-experiment, **only** if E0 confirms self-stabilization.
Compare Condition C's graph hidden-state messages against three other
candidate destabilizing forcings, on the same Component-0 setup:

| Comparator | What it injects | What it tests |
|---|---|---|
| **C** | Upstream basins' learned hidden states `h_u(t-1)` | The graph signal as we currently use it. |
| **C-rand** | Random noise of the same magnitude as C's messages | The *null* — does any signal at all destabilize the LSTM, or does the message content matter? |
| **C-precip** | Upstream basins' raw precipitation, not learned states | Is the destabilizing power in the upstream physical input, or in the learned representation? |
| **C-lag** | The basin's *own* forcing time series shifted by a longer lag | Self-forcing as a destabilizer — does the destabilization need to be inter-basin at all? |

If C beats all three comparators, the graph hidden-state message has
content beyond random noise / raw upstream input / temporal lag.
If C ≈ C-precip, the learned representation is replaceable by raw upstream
forcing — important simplification (no need for the DirectedGraph-LSTM).
If C ≈ C-rand, the destabilization is content-free — that would falsify
the framing.

## Scale target

**Component 0** — 183 basins, eastern US, 6 HUC regions, max graph depth 4.
Already extracted to
`topology_analysis/phase1_network_discovery/outputs/component0_basins.txt` and
`component0_edges.csv`. Basins-per-depth: 33 / 81 / 51 / 16 / 2 — enough
statistical power at depths 1–3 (unlike the 23-basin network, which had n=2 at
depths 2 and 3).

Why not CAMELS-531: many basins in that benchmark have no graph neighbors, so
the topology variable is ill-defined for a large fraction of the data.
CAMELS-531 is a good follow-up for direct comparability to Kratzert 2019, not
the first target.

## What we are NOT claiming (bounds of contribution)

- We are not claiming state-of-the-art on CAMELS-US. EA-LSTM median ≈ 0.74 is
  the ceiling for a possible follow-up, not a requirement here.
- We are not claiming a new architectural primitive. The DirectedGraph-LSTM
  already exists in the repo; the paper's contribution is the **evaluation**
  + the dynamical-systems-on-network framing.
- We are not claiming general-purpose GNN insights. The framing is hydrology-
  specific (rivers as networked dynamical systems with known governing PDEs).
- We are not claiming a *new* dynamical-systems theory. We are *applying*
  existing dynamical-systems-on-networks language to interpret an empirical
  ML result.

## Verifiable physical claims to anchor the paper (PI ask, 2026-04-21)

The paper should make claims that can be checked against canonical
hydrology physics, not just ML benchmark numbers. Concrete anchors to use:

- **Saint-Venant equations** for one-dimensional open-channel flow:
  `∂A/∂t + ∂Q/∂x = q`, `∂Q/∂t + ∂(Q²/A + gA·h̄)/∂x = gA(S₀ - S_f)`. These
  are the textbook physical model the LSTM is approximating; any
  destabilizing forcing we identify should be checkable against what the
  Saint-Venant model would say is the *physical* upstream→downstream
  transmission.
- **Manning's equation** for steady-state discharge: `Q = (1/n)·A·R^(2/3)·S^(1/2)`.
  Used as a reference for what the relationship between basin
  characteristics and flow magnitude should look like.
- **Linear reservoir routing** (`dS/dt = I - kS`, `Q = kS`): the simplest
  physically-grounded model of upstream→downstream flow. We can train an
  LSTM on synthetic data from this model with known ground-truth
  dynamics, and ask: does the LSTM-on-graph recover the routing parameter
  `k`?

The first two are *positioning* anchors (we cite them, we don't
re-implement them). The third is a candidate **synthetic experiment** to
add later — train on a known dynamical system, verify the framework
recovers its dynamics, then transfer the methodology to CAMELS.

## What a "meaningful finding" looks like

The reframing widens the publishable surface — there are now multiple
distinct findings the paper can headline, depending on what E0 + E0.5 +
the ablation + the forcing-comparison reveal.

1. **Positive — destabilization works, content matters:** E0 confirms
   self-stabilization. C beats both A and B at 5-seed 95% CI. C-rand and
   C-precip don't match C. Headline: *"river-network topology supplies an
   external forcing that breaks LSTM self-stabilization in a way that
   recovers physical dynamics — and the learned hidden-state messages
   carry content beyond raw upstream forcing."*
2. **Positive — destabilization works, content doesn't (the simplification
   result):** E0 confirms self-stabilization. C ≈ C-precip > A. Headline:
   *"the destabilizing forcing is the upstream physical input itself; the
   learned hidden-state representation is replaceable by raw upstream
   precipitation."* Removes the case for a complex graph-LSTM.
3. **Identity beats messaging:** B ≈ C > A. Headline: *"topology helps as
   a static identity signal, not as a runtime communication channel,"*
   consistent with Kirschstein 2024 with a precise mechanism.
4. **Pure null:** A ≈ B ≈ C and the forcing-comparison shows nothing
   destabilizes meaningfully. Replicates Kirschstein at scale with our
   stronger ablation. Workshop-publishable as a negative result.
5. **E0 fails:** the LSTM does *not* exhibit clear self-stabilization
   signatures. The framing is wrong but the original A/B/C ablation still
   delivers a paper — we publish the ablation alone with a discussion
   paragraph noting the framing didn't pan out.

## Required methodology — non-negotiable for publication

1. **Multi-seed.** ≥ 5 seeds per condition, bootstrap 95% CI on per-basin ΔNSE.
2. **Matched hyperparameters across A, B, C** — no condition gets a tuned
   advantage.
3. **Matched training time** (epochs + LR schedule). Eliminates the LSTM-drift
   confound we found on 23 basins.
4. **Held-out test period** unchanged from pilot: train 1990-1999,
   validate 2000-2004, test 2005-2008.
5. **Report KGE and log-NSE in addition to NSE.** The 08165300-style
   variance-outlier-basin problem repeats at scale.
6. **E0 (self-stabilization) and E0.5 (loss saturation) results reported
   regardless of outcome** — they are the foundation of the framing and
   their findings stand independent of the main ablation.

## Reproducibility — Colab packaging (PI ask, 2026-04-21)

The PI requested a one-command reproducibility setup. Plan:

- `setup.py` (or `setup.sh`) at repo root — installs the `nh` env,
  downloads a pinned subset of CAMELS-US, prepares Component 0 derivatives.
- `run.py` at repo root — single entry point with subcommands:
  `run.py baseline`, `run.py graph`, `run.py e0`, `run.py e0.5`,
  `run.py compare`. Each subcommand reproduces the corresponding
  experiment block.
- Colab notebook (`run_in_colab.ipynb` or similar) that does
  `git clone && python setup.py && python run.py <experiment>`.

Implemented after E0 and E0.5 land — no point packaging until the
gate experiments are done.

## Open questions for the next PI meeting

The 2026-04-21 meeting answered some questions and surfaced new ones. The
ones still standing for next time:

1. **Compute.** 3 conditions × 5 seeds × 15+ epochs on 183 basins, plus E0
   and E0.5 on the existing 23-basin runs (cheap). On CPU the graph runs
   are 8× slower than on 23 basins. Concrete ask still pending: lab
   machine, cloud credits, or HPC access.
2. **Forcing-comparison comparators.** PI ask was high-level
   ("destabilizing forcings"). The four comparators in the table above
   (C, C-rand, C-precip, C-lag) are my proposal — does the PI agree these
   are the right contrasts, or are there others (snow water equivalent,
   antecedent soil moisture, ENSO state) that are more informative?
3. **Verifiable physical claims to lead with.** Saint-Venant, Manning, and
   linear-reservoir routing are my proposed anchors. Which of these does
   the PI want as the **primary** verification target, and is the
   synthetic linear-reservoir experiment in scope or a follow-up?
4. **Methodology gaps unchanged from before.**
   - Heuristic vs. NHDPlus ground-truth edges (we have `pynhd` installed).
   - Maurer (ends 2008) vs. daymet/nldas (extend to 2014). Which forcing?
5. **Publication framing & venue.** Workshop is the target — Climate Change
   AI, AI4Earth, ML4PS, or other? Does the PI have a venue in mind?
6. **Positioning vs. prior work.** Kirschstein & Sun 2024 (null) and Jiang
   et al. 2025 (positive). Which is the primary point of comparison?
7. **Dynamical-systems-on-networks literature.** The PI gestured at this
   as "more robustly grounded." Concrete reading list to anchor the
   paper's introduction?

## Status as of 2026-04-21 (post-meeting)

**Done.**
- Pilot completed on 23-basin Texas network (`INSIGHTS.md`).
- Component 0 extracted: 183 basins, 624 edges, depth 4
  (`topology_analysis/phase1_network_discovery/outputs/component0_*`).
- NH baseline config for Component 0:
  `experiments/configs/lstm_component0_baseline.yaml`.
- DirectedGraph-LSTM training parameterized for Component 0:
  `experiments/training/train_graph_component0.py`.
- Reorganized `experiments/` into `configs/`, `basin_lists/`, `training/`,
  `analysis/`. Per-subfolder READMEs.
- `JOURNAL.md` started for ongoing decision/feedback log.
- Idea 1 reframed around dynamical-systems language (this file).

**Up next, in order.**
1. ~~**E0** (self-stabilization verification) on the existing run-05 baseline.~~
   **DONE 2026-04-24 — PASS.** Both probes 100%; robust to σ=0.5/2.0.
2. ~~**E0.5** (60-epoch loss-saturation curve).~~
   **DONE 2026-04-24 — PRAGMATIC PASS (val saturated, train still descending → overfit past epoch 5; pilot epoch-30 stop near-optimal on val).** See JOURNAL entry 2026-04-24 (`/crs-unleashed` chain).
3. ~~**E1** — self-stabilization on the **weak** baseline (run 03, no encoding).~~ **DONE 2026-04-24 — PASS.** Identical signature to strong baseline → self-stabilization is intrinsic, not encoding-induced.
4. ~~**E0-B'** — stronger Probe B variants (zero-out, random historical day).~~ **DONE 2026-04-24 — BOTH PASS** (max-dev 0.035 / 0.033 vs. 0.30 threshold).
5. ~~**Multi-seed replication of E0**~~ **DONE 2026-04-25** — 6 seeds, all 100% pass, zero variance.
6. ~~**Probe A at t=29 (worst-case timing)**~~ **DONE 2026-04-25** — median deviation 0.098 of natural pred std; 23/23 below threshold.
7. ~~**State-space recovery measurement**~~ **DONE 2026-04-25** — ‖Δh‖_norm 0.478 → 0.012 in 5 steps. True contracting dynamics, not head-orthogonality.
8. ~~**Condition B (topology_features) implementation in `train_graph_component0.py`**~~ **DONE 2026-04-25** — verified by pre-training NSE = 0.423 = exact baseline match.
9. ~~**Multi-seed E0.5**~~ **DONE 2026-04-26** — within-seed saturation CONFIRMED (slopes ≤ 0.0019/epoch). **Cross-seed spread: 0.111 NSE** (0.366 ↔ 0.478) — 2× the pre-registered bar. **Pilot's +0.078 headline is now multi-seed-contingent** — cross-seed variance in baseline alone is larger than the gap. Framing probes unaffected. See JOURNAL 2026-04-26 (later).
10. ~~**A/B/C pre-registration document**~~ **DONE 2026-04-25 (this section + Compute Spec below)** — methodology locked.

---

## A/B/C Publication-Run Protocol (LOCKED 2026-04-25)

This section is the definitive specification of how Conditions A, B, C will be
trained, evaluated, and compared in the publication run. Once compute lands,
no methodology debate — execute as written. Amendments only via dated entries
in `JOURNAL.md`.

### Conditions

| Condition | Model | Static features | Edges | Warm-start | What it tests |
|---|---|---|---|---|---|
| **A** Baseline | NH `cudalstm` + basin ID one-hot | 5 attrs + n_basins one-hot | none (no graph) | from scratch | Strong Kratzert-style baseline |
| **B** Topology features | DirectedGraphLSTM with `edges=[]` | 5 attrs + n_basins one-hot + **5 topology scalars** | none in model | **from scratch** | Does graph *position* help, without runtime communication? |
| **C** Topology + message passing | DirectedGraphLSTM | 5 attrs + n_basins one-hot | full network edges with edge features | **from scratch** | Does runtime hidden-state messaging add to B? |

**Crucial**: all three conditions train **from scratch** (`--no-warm-start`)
with **matched epochs** and **matched hyperparameters**. The earlier "warm-
start C from A" pattern from the pilot is retired for the publication run —
yesterday's hostile-reviewer Q5 flagged it as creating asymmetry between
A/B and C. With from-scratch training across all three, A − B and B − C
margins are interpretable as pure structural-information effects.

### Hyperparameters (matched across A / B / C)

| | Value |
|---|---|
| hidden_size | 64 |
| dropout | 0.4 |
| seq_length | 30 |
| predict_last_n | 1 |
| batch_size | 256 |
| optimizer | Adam |
| loss | MSE |
| learning_rate | 1e-3 (constant) |
| epochs | 30 (matches the 23-basin pilot's saturation point per E0.5) |
| clip_gradient_norm | 1.0 |
| initial_forget_bias | 3 |
| forcings | maurer |
| dynamic_inputs | PRCP, SRAD, Tmax, Tmin, Vp |
| target | QObs(mm/d), clipped to zero |
| static_attributes | elev_mean, area_gages2, slope_mean, p_mean, pet_mean |
| use_basin_id_encoding | True (Kratzert-style) |
| seeds | 11, 13, 17, 19, 23 (n=5) |
| train period | 1990-01-01 to 1999-12-31 |
| validation period | 2000-01-01 to 2004-12-31 |
| test period | 2005-01-01 to 2008-12-31 |

### Basin set + edges (locked)

- **Basin set**: Component 0 = 183 basins, eastern US, 6 HUC regions, max graph
  depth 4. List in `topology_analysis/phase1_network_discovery/outputs/component0_basins.txt`.
- **Edges**: 624 heuristic edges in `component0_edges.csv`. NHDPlus replacement is
  a follow-up ablation (see Compute Spec below), not blocking for the first run.

### Reporting

Primary metrics:
- **Median NSE per condition × seed** — bootstrap 95% CI across seeds.
- **Per-basin ΔNSE** for B − A and C − B; report distribution + signed-rank test.
- **Depth-stratified median NSE** across depths 0, 1, 2, 3, 4.
- **KGE and log-NSE** alongside NSE (per the methodology requirements).
- **Outlier-trimmed means** (drop bottom-5% basins) — published-style robust summary.

Headline test: **C − B > 0** at 5-seed 95% CI is the message-passing-helps claim.
**B − A > 0** at 5-seed 95% CI is the topology-as-identity-helps claim.

### Compute spec (transparent)

| Run | Per-run cost | × seeds | × variants | Total |
|---|---|---|---|---|
| Condition A (NH cudalstm, no graph) | ~10 min on GPU / ~30 min on CPU | 5 | 1 | 50 min GPU / 2.5 hrs CPU |
| Condition B (graph-LSTM with empty edges, augmented x_s) | ~2.5 hrs CPU per smoke estimate; ~30 min on GPU | 5 | 1 | 2.5 hrs GPU / 12.5 hrs CPU |
| Condition C (graph-LSTM with full edges + message passing) | same as B | 5 | 1 | 2.5 hrs GPU / 12.5 hrs CPU |
| **Headline run total** | | | | **~5.5 hrs on a single GPU; ~28 hrs on CPU** |
| NHDPlus-edges follow-up (Condition C only) | same as C | 5 | 1 | 2.5 hrs GPU |
| **All-in including NHDPlus follow-up** | | | | **~8 hrs on a single GPU** |

Concrete sizes:
- **Single GPU sufficient** (T4 / A10 / A100 — anything with ≥ 16 GB VRAM).
  No parallelism needed; the graph-LSTM is small (~35 K params).
- **Cloud cost estimate** (Lambda Labs A6000 @ $0.80/hr): ~$7 for the headline run; ~$10 all-in.
- **Local lab GPU**: any modern card works.
- **Free credits**: $1000 GCP for Research or AWS Research Credits would cover this run 100×.

### Execution checklist when compute lands

```bash
# Condition A (NH cudalstm — already buildable)
for seed in 11 13 17 19 23; do
  python neuralhydrology/nh_run.py train \
      --config-file experiments/configs/lstm_component0_baseline.yaml
  # (modify experiment_name + seed in the YAML per seed; or use the multiseed template pattern)
done

# Conditions B and C (graph-LSTM trainer — already buildable)
for cond in topology_features warm; do
  for seed in 11 13 17 19 23; do
    python experiments/training/train_graph_component0.py \
        --variant $cond \
        --seed $seed \
        --no-warm-start \
        --epochs 30
  done
done

# Analysis
python experiments/analysis/compare_results.py \
    --baseline runs/lstm_component0_baseline_seed*/test/model_epoch030/test_metrics.csv \
    --baseline-label "A: Baseline" \
    --graph runs/graph_c0_topology_features_seed*/test_metrics.csv:"B: topo-features" \
            runs/graph_c0_warm_seed*/test_metrics.csv:"C: graph+messages"
```

### Pre-registered falsification conditions

- **C − A > 0 fails (95% CI includes 0):** topology + message passing didn't help. Workshop-publishable as a negative result with the dynamical-systems mechanistic frame.
- **C − B ≈ 0 but B − A > 0:** topology helps as static identity, not as runtime messages. Cleanly positions against the "graph-LSTM is special" assumption.
- **All three saturate at the same NSE:** no benefit from topology in any form. Replicates Kirschstein 2024 at scale with stronger ablation. Still publishable.
3. **Add Condition B** (`--variant topology_features`) to
   `train_graph_component0.py`. Small edit.
4. **Compute decision** — block on this before launching the full A/B/C
   sweep at 183-basin scale.
5. **Forcing-comparison sub-experiment** designs — implement the four
   comparators only after E0 confirms the framing.
6. **Colab packaging** (`setup.py` + `run.py`) — only after gates pass.

**Not yet started.**
- Multi-seed runs.
- NHDPlus ground-truth edge replacement.
- Synthetic linear-reservoir experiment.
- Dynamical-systems-on-networks literature read.

## Pointers

- Decision/feedback log: `JOURNAL.md`
- Alternative direction (set aside): `idea2/`
- Pilot findings: `INSIGHTS.md`
- Chronological experiment log: `CURRENT_STATE.md`
- Experiment scripts and their status: `experiments/README.md`
- Per-run results: `runs/README.md`
