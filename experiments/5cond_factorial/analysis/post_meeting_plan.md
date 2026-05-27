# Post-Meeting Plan — Local Subgraphs + Loss-Distribution Invariant

**Date:** 2026-05-13 (post-professor meeting).
**Compute budget:** ~34 GPU hours on Colab Pro T4.
**Status:** PLAN ONLY. No code, no runs yet. To be reviewed before any execution.
**Companions:** `5cond_run_analysis.md` (what we found), `architecture_analysis.md` (why), `testing_framework_proposal.md` (the 6-step ladder this plan partially supersedes), `meeting_brief.md` (the questions we took into the meeting).

---

## 1. What the professor said, triaged as CRS

The professor's comments + my read of which to keep, which to nuance, which to drop.

| # | Comment | CRS verdict | Notes |
|---|---|---|---|
| 1 | First thing: compute mean and standard deviation across seeds, not just medians. | **KEEP** | We were reporting median + bootstrap CI. Mean ± std is complementary and is the framing the professor will read fastest. Use both. |
| 2 | Reduce data the model sees → expect mean to degrade and deviation to increase. | **KEEP** | Standard bias-variance intuition. Useful as a *sanity check*: if reducing data does NOT degrade NSE, something's wrong with the experiment. |
| 3 | Track how mean/deviation change through scaling and architectural revisions. | **KEEP — central to the plan** | This is the closest thing the professor offered to a "what does success look like" definition. The invariant: mean improves with model changes; std stays constant (or shrinks). |
| 4 | Shift mean/deviation to a reasonable window — increase the training window. | **KEEP, but with care** | Currently train period is 1990–1999 (10 years). Extending to 1980–2004 (25 years) is feasible if CAMELS goes back that far. We were already at a defensible 10-year window; extending helps but only after stabilizing the smaller experiments. |
| 5 | 3-seed loss-distribution as a stable across-runs invariant. | **KEEP — anchor metric** | Our headline becomes "mean NSE ± std across 3 seeds" rather than "median across basins with bootstrap CI". Both are still computed; the former is the meeting-facing summary. |
| 6 | Want runs on the order of 5–10 min, and 15–30 min for 3 models. | **KEEP — central to the plan** | The 5cond runs were ~30 hr; we want each follow-up experiment to be ~30 min for 3 seeds × 3 conditions. This means radically reducing scale (= smaller basin sets) AND keeping epochs similar. |
| 7 | Build basin graph with distances, use shortest-path random walker, trim by shortest-path distance, use as the standard test set. | **KEEP, with one modification** | Random walks are great for *generating candidates*. But for paper-grade results, we should **pre-commit to a small set of explicitly-named subgraphs** rather than re-sampling each session. Otherwise it's cherry-picking. Pre-commit 3–5 subsets and report on all of them. |
| 8 | Do not test on 183 basins — that may be what is causing the underperformance. Test on localized subsets that make hydrological sense. | **KEEP — biggest single change to the plan** | The 23-basin pilot (+0.078 NSE) was all in one Texas HUC region. The 183-basin Component 0 spans 6 HUC regions (HUC 01–06, Atlantic Northeast through Tennessee). Graph signal probably averages out across that heterogeneity. Localized subsets are the right unit. |

**Comments I'd add as CRS that the professor did not raise:**

- *"Stay falsifiable."* Pre-commit subset choice + success criterion before each run. Otherwise we'll subconsciously pick subsets where graph signal already helps.
- *"Same paper, different scope."* Moving to localized subsets is not abandoning the paper claim; it's testing it where it has the best chance of holding. If the headline-after-this-work becomes "graph features beat standard LSTM on coherent local river networks (n=10–30 basins, paired NSE Δ ≈ +X)" that is a defensible workshop paper.

---

## 2. The 3-seed loss-distribution invariant — what it actually means in practice

The professor's framing: as we change the model and the data, **the mean NSE across seeds should improve** (we're making the model better), **but the standard deviation should stay about constant or shrink** (we're not just adding random variance).

What this looks like operationally:

```
Run 0 (baseline):     mean = 0.610 ± 0.013 (3 seeds)
Run 1 (change A):     mean = 0.625 ± 0.012  ← improvement, std stable. GOOD.
Run 2 (change B):     mean = 0.635 ± 0.011  ← further improvement, std stable. GOOD.
Run 3 (change C):     mean = 0.640 ± 0.038  ← marginal mean lift, std blew up. SUSPICIOUS.
                                                Change C may be helping one seed and
                                                hurting others — not robust.
```

This becomes the reporting protocol for every architectural revision:
- 3 seeds minimum (we can afford more if budget allows; 5 is better)
- Mean ± std as the headline
- Per-basin median NSE still computed for the existing bootstrap-CI framework
- A change is "validated" only when mean improves AND std doesn't materially expand

**Subtlety the professor's framing doesn't fully address:** std *can* legitimately grow when adding capacity (more parameters = more sensitivity to init). The decision rule: a std-increase ≤ 50% relative is tolerable if mean improvement is ≥ 0.02 NSE. Larger std-blowups need investigation before being adopted.

---

## 3. Localized basin selection — methodology and pre-commitment

### 3.1 What's already available

Component 0's 183 basins are already discovered with directed edges (`topology_analysis/phase1_network_discovery/outputs/component0_edges.csv`). Inside Component 0 there are **14 natural root-rooted subgraphs of 10+ basins** (computed via descendants-of-root). The 4 largest:

| Root basin | Subgraph size | HUC region (approx) |
|---|---|---|
| 01594950 | 51 basins | HUC 02 (Mid-Atlantic) |
| 01516500 | 31 basins | HUC 02 (Mid-Atlantic) |
| 03455500 | 31 basins | HUC 06 (Tennessee) |
| 03450000 | 28 basins | HUC 06 (Tennessee) |
| 03026500 | 24 basins | HUC 05 (Ohio) |

These are already locally-coherent: a single river system + its descendants, by construction.

### 3.2 The shortest-path-walker variant (professor's specific request)

Implement as a *secondary* candidate-generator:

1. Pick a random seed basin.
2. Take a random walk on the graph (treating it as undirected for walker purposes; revisits allowed).
3. After K steps, collect all visited basins as the candidate subset.
4. Filter: size ∈ [15, 30], at least 2 graph depths represented, all from same HUC-2 region.

This generates more subgraph variety than just root-descendants (e.g., interior-rooted subgraphs that span tributary forks). Use it to expand the candidate pool beyond the 14 root-rooted ones.

### 3.3 Pre-commitment — final 5 subgraphs

CRS decision (see §10 for reasoning): walker-generated subgraphs seeded at the **4 largest roots in Component 0** (which is the special case of the random-walker seeded at roots; root-descendants on a DAG equal walker-output by construction), plus the historical Texas pilot.

**Note:** verified via direct query on `component0_edges.csv` — the root-descendant subgraphs each span 2–4 HUC regions, not a single one. They're **graph-coherent**, not **climate-coherent**. This is the correct unit for testing whether the message-passing structure helps, regardless of climate homogeneity. Climate homogeneity is a separate axis that subset #5 (Texas pilot) anchors.

| # | Identifier | Size | Depth range | HUCs spanned | Why on the list |
|---|---|---|---|---|---|
| 1 | root `01594950` | **51** | 0–4 (full) | 02 / 03 / 05 | Largest local subgraph in Component 0; contains the rare depth-4 basins where graph signal should matter most. |
| 2 | root `01516500` | **31** | 0–3 | 02 / 04 / 05 | Mid-size with 4 depths. Different root system from #1. |
| 3 | root `03450000` | **28** | 0–4 (full) | 02 / 03 / 05 / 06 | Most HUC-diverse root subgraph; also has the rare depth-4. |
| 4 | root `02055100` | **22** | 0–4 (full) | 02 / 03 | Smallest subgraph with full depth range. Tightest unit. |
| 5 | Texas pilot (Component 3) | **23** | 0–3 | 12 (all one HUC) | The set where +0.078 NSE was originally seen. Only climate-coherent option. Sanity anchor connecting to the historical result. |

Size range 22–51 basins → estimated training time 5–15 min per run on T4. Depth ranges include the 4-depth case for 3 of the 5, which is where the graph-signal hypothesis is most testable.

**Once committed, all architectural changes are reported across all 5 subsets.** No silent dropping of subsets where a method does badly.

---

## 4. The phased experiment plan (target ~34 GPU hours total)

Each phase is gated: do not advance until the prior phase has produced a clean, written result. Numbers below are estimates on T4.

### Phase 0 — Re-analyze existing 5cond data through the new invariant (~0 hr, free)

- Recompute mean ± std (cross-seed) for L, G, G+T, G+M, G+T+M from the existing 15 runs.
- Compute the **base invariant value** that all subsequent work will track relative to.
- Express the existing 5cond contrasts in the new format.
- Output: short addendum to `5cond_run_analysis.md` with the mean ± std table.
- **No new runs.** This is a free reframing of what we already have.

### Phase 1 — Subgraph candidate generation (~0 hr, free)

- Implement shortest-path-walker (or just use root-descendant subgraphs — both are valid).
- Generate basin lists for all 5 pre-committed subsets.
- Write basin files to `experiments/basin_lists/local_subgraphs/`.
- Pre-commit the subset list in `experiments/local_subgraphs_v1/preregistration.md`.
- **No GPU compute.** Pure data/file generation on CPU.

### Phase 2 — Subset characterization (~4 hr)

- For each of the 5 subgraphs:
  - Train L (cudalstm) and G (DirectedGraphLSTM, empty edges) at 3 seeds × 30 epochs.
  - Smaller subsets → ~5–10 min per run on T4.
- 5 subsets × 2 models × 3 seeds × ~10 min = **~5 hr** with overhead.
- Output: `mean ± std NSE` per (subset, model). Establishes the new invariant baseline.
- **Expected pattern:** subsets show mean NSE ≈ 0.55–0.70 with std ≈ 0.01–0.05 across seeds. The 23-basin pilot subset should reproduce the historical +0.078 NSE lift if everything is consistent (free smoke-test of the pipeline).

### Phase 3 — Replicate the 5-condition factorial on the chosen subgraph (~5 hr)

- Pick the SINGLE BEST subgraph from Phase 2 — best meaning: (a) reasonable size, (b) reproduces "L is non-trivial baseline", (c) has graph-depth diversity.
- Run all 5 conditions (L, G, G+T, G+M, G+T+M) × 3 seeds × 30 epochs.
- 5 × 3 × ~20 min = **~5 hr**.
- Output: the 5cond factorial table, but now on a localized subgraph. The headline question: **does G+T+M beat L on this subset?** If yes, the paper claim is salvaged. If no, we move to Phase 4 architectural revisions.

### Phase 4 — Iterative architectural revisions (~6 hr, tiered)

CRS decision (see §10 for reasoning): the 5 original revisions are re-tiered by *expected leverage × implementation cost*. Tier A is mandatory; Tier B is conditional on A's outcome; Tier C is fallback; Tier D is deferred to a future cycle.

For every revision: train on the chosen subgraph × 3 seeds × 30 epochs. Track mean ± std relative to Phase 3 baseline. **Adopt** only if mean improves ≥ +0.02 AND std doesn't expand by >50%.

**Tier A — must test (highest expected leverage):**

| Rev | Change | Tests which hypothesis | Cost |
|---|---|---|---|
| **4.A.1** | **Area-weighted aggregation** instead of mean aggregation. Replace `m_v = mean(msgs)` with `m_v = Σ_u (area_u / Σ area_v) · msg_u`. | Mean aggregation gives a 1 km² parent the same weight as a 275 km² parent (max ratio in Component 0). Direct fix for `architecture_analysis.md` Defect 3.2.1. | ~1 hr |
| **4.A.2** | **Drop basin one-hot encoding** for G, G+T, G+M, G+T+M (small subgraph → only 20–50 one-hot dims, so topology features become a meaningful share of the static input rather than 0.7%). | Tests whether basin one-hot is subsuming the topology features (the dominant explanation for G+T − G ≈ 0 from the 5cond run). | ~1 hr |

**Tier B — test if Tier A didn't close the gap:**

| Rev | Change | Tests which hypothesis | Cost |
|---|---|---|---|
| **4.B.1** | **2-layer MLP message function** instead of single linear. `W_msg_edge` → `Linear → ReLU → Linear`. | Single linear can't express conjunctive features ("if parent has high flow AND large area, send strong message"). Fix for Defect 3.2.3. | ~1 hr |
| **4.B.2** | **Learnable scaled residual** replacing `tanh(W_out(m))` saturation. New: `s * W_out(m)` where `s` is a single learnable scalar init at 0.01. | tanh saturates at ±1, capping the graph contribution magnitude. Fix for Defect 3.2.4. Tiny code change, cheap test. | ~1 hr |

**Tier C — fallback if Tier A + B insufficient:**

| Rev | Change | Tests which hypothesis | Cost |
|---|---|---|---|
| **4.C.1** | **Embed discrete topology features** (depth, in-degree, out-degree) as `nn.Embedding(max_val+1, 4)` each. | Z-normalizing discrete features to continuous loses categorical semantics. Fix for Defect 2.3.2. | ~1 hr |
| **4.C.2** | **K-hop message passing per timestep** (K=2). Lets a depth-3 basin receive root-signal within 1 timestep rather than 3. | Single-hop per timestep artificially couples graph-depth lag to temporal lag. Fix for Defect 1.2.6. | ~1 hr |

**Tier D — deferred to a future cycle (too large for this compute envelope):**

| Rev | Change | Why deferred |
|---|---|---|
| **4.D.1** | **Predict-then-route.** Replace learned message passing with explicit physical Muskingum-style routing on predicted runoff. | Requires a new model class (~half a day of implementation), changes the supervision signal, would be its own paper section. Pre-register separately. |

**Budget for Phase 4 (assumes A + B run, C as needed):**

- Tier A: 2 revisions × 3 models × 3 seeds × ~15 min = ~2 hr
- Tier B: 2 revisions × 3 models × 3 seeds × ~15 min = ~2 hr
- Tier C reserve: ~2 hr
- **Total Phase 4: ~4–6 hr GPU.** Down from original 7 hr estimate.

### Phase 5 — Hold-out validation (~3 hr)

- Take the best variant from Phase 4 and run on subsets #1–5 that WEREN'T used in Phase 3/4 (i.e., the 4 other subgraphs).
- Confirms the improvement generalizes across local subgraphs, not just the one we tuned on.
- 4 subgraphs × 1 best variant × 3 seeds × ~15 min = **~3 hr**.
- Output: mean ± std for the best variant on each held-out subset. Critical for honesty — if Phase 4 wins only show on the tuned subset, that's overfitting.

### Phase 6 — Scale-up sanity check (~5 hr)

- Take the best variant from Phase 4 and run on the full 183-basin Component 0 × 3 seeds × 30 epochs.
- ~30 min per run × 3 seeds × 4–5 conditions = ~5 hr.
- Tests whether the improvements found at local-subgraph scale carry over to the full network.

### Compute budget rollup

| Phase | Description | GPU hours |
|---|---|---|
| 0 | Recompute invariants from existing data | 0 |
| 1 | Subgraph generation | 0 |
| 2 | Subset characterization (L + G × 5 subsets × 3 seeds) | 5 |
| 3 | 5cond factorial on chosen subgraph | 5 |
| 4 | Architectural revisions Tier A + B (4 revisions × ~1 hr) | 4–6 |
| 5 | Hold-out validation on 4 unused subgraphs | 3 |
| 6 | Scale-up sanity check on 183-basin | 5 |
| **Subtotal** | | **22–24** |
| Buffer for re-runs / surprises / Tier C reserve | | 10–12 |
| **Total** | | **34** |

Fits in the 34-unit budget with ~30% buffer. The buffer is meaningful — Tier C of Phase 4 is held in reserve and only used if Tier A+B didn't produce a clear win.

---

## 5. Specific tests for "why topology / MPNN underperformed"

The 5cond run produced three negative contrasts. The plan above tests each against a specific hypothesis:

| Contrast | What we observed | Phase that re-tests | Hypothesis being tested |
|---|---|---|---|
| G+T − G ≈ 0 | Topology features inert. | Phase 4.3 (drop one-hot), 4.4 (embed features), Phase 3 on smaller subset (one-hot is naturally smaller). | Redundancy with basin one-hot encoding. |
| G+M − G < 0 (slight) | Message passing slightly hurts. | Phase 4.1 (area-weighted), 4.2 (MLP message), and the small-subset run itself (less basin-heterogeneity noise). | Mean aggregation washes out signal + single linear message function can't extract it. |
| G+T+M − G < 0 (sub-additive) | Combination is worse than either alone. | Naturally retested when Phase 4 revisions are applied. | The two pathways are stealing each other's gradient signal under poor training, not genuinely incompatible. |
| L − G ≈ +0.05 | Architecture-matched control loses to NH. | Phase 6 only (re-evaluated at 183-basin scale post-revisions). | Gradient-noise scale or data-exposure pattern. Lower priority since Step 1 (matched-budget) was inconclusive and the issue dilutes at smaller scale. |

**Crucially:** the order is intentional. We test the cheapest hypothesis-fixes first, on the smallest scale where the contrast can be measured. Only after Phase 4 do we re-engage with the 183-basin question.

---

## 6. Decision tree

```
After Phase 2:
  → Subgraphs show mean NSE 0.55-0.70, std 0.01-0.05.        → continue to Phase 3.
  → Subgraphs show wildly varying NSE / std > 0.10.           → STOP. Subgraphs are too small / heterogeneous.
                                                                Reconsider subset choices.

After Phase 3 (5cond on chosen subgraph):
  → G+T+M − G > +0.02 (graph helps on local subgraph).        → continue to Phase 4 to amplify.
  → G+T+M − G ≈ 0 (graph still neutral).                       → continue to Phase 4; redesigns may help.
  → G+T+M − G << 0 (graph clearly hurts even small).          → consider pivot to predict-then-route (4.5) directly.

After Phase 4 (each revision):
  → mean improves ≥ +0.02, std stable.                         → adopt that change; build on it for next revision.
  → mean improves but std blows up > 50%.                       → investigate why; do NOT adopt without explanation.
  → mean degrades or flat.                                       → revert; try next revision on the previous best.

After Phase 5 (held-out subgraphs):
  → Improvements generalize across 4 held-out subsets.        → real result; proceed to Phase 6 + writeup.
  → Improvements only on tuned subset.                         → overfitting. Step back; redesign requires
                                                                  cross-subgraph validation built in.

After Phase 6 (183-basin scale-up):
  → G+T+M variant matches or beats L on 183-basin.            → paper claim is saved at full scale.
  → G+T+M-variant still loses on 183-basin.                    → paper claim is "graph helps at local-network
                                                                  scale; doesn't yet scale to multi-HUC networks".
                                                                  Still publishable as a workshop paper.
```

---

## 7. Reframing the paper claim

The paper claim originally: *"our added features outperform a standard LSTM on discharge prediction."*

The actual evidence we'll likely produce in this work cycle: *"on locally-coherent river subgraphs (15–30 basins, single HUC region), graph-based features add measurable value over a strong LSTM baseline; the effect is visible at small-network scale and (does / does not) scale to multi-HUC networks."*

This is honest, narrower, and defensible. It also matches the original 23-basin pilot finding (+0.078 NSE) — connecting the new work to the historical signal that motivated the project in the first place.

The professor's "test on localized subsets that make sense" is not a retreat — it's a refocusing onto the scale where the hypothesis has the best chance of being true. If it doesn't hold there, it doesn't hold anywhere, and a clean negative result is the right paper.

---

## 8. What this plan does NOT include

To stay honest:

- No commitment to *what we'll write up* before we see the data. The Phase 4 decision tree tells us when to keep going and when to stop.
- No promise that any architectural revision will work. The plan budgets for 5 attempts and assumes 2–3 will give meaningful improvement.
- No promise of beating L on the 183-basin scale. Phase 6 may end the project's most ambitious claim.
- No work on validation-set best-checkpoint selection, NHDPlus edges, or seed expansion. Those are Phase-7 polish for a successful result.

---

## 9. Concrete first action (when execution begins)

Phase 0: in Python (no GPU), recompute mean ± std cross-seed for the existing 5cond data. Add as an addendum table to `5cond_run_analysis.md`. **Estimated time: 30 minutes of CPU work.** This produces the *anchor invariant* that every future change is compared against.

This is the lowest-cost, highest-leverage immediate step. Do it before launching any new runs.

---

## 10. CRS decisions — settled

All 5 questions resolved by CRS judgment. Brief reasoning per decision; override if you disagree.

### Q1 — Subgraph list: settled. See §3.3.

**Decision:** 5 subgraphs as listed in §3.3. The 4 walker-seeded-at-root subgraphs from Component 0 (sizes 22 / 28 / 31 / 51, depths 0–4 in three of four) + the Texas pilot (HUC 12).

**Reasoning:**
- Root-rooted-on-a-DAG = walker output by construction (walks from a root reach exactly its descendants). So the "shortest-path walker" the professor asked for is what these are, just with the seed pre-committed at the root.
- I deliberately picked subgraphs that include the rare depth-4 basins (3 of the 5). The graph-signal hypothesis is most testable at depth ≥ 2; depth-4 basins are where it matters most.
- Empirical finding from the audit: root-descendant subgraphs span 2–4 HUC regions (the topology was inferred via distance + area, not strict drainage). So these are **graph-coherent, not climate-coherent.** Climate coherence is owned by subset #5 (Texas pilot, all HUC 12).
- 5 subgraphs is enough to detect generalization without spending the entire budget on characterization.

### Q2 — Training window: settled. **Keep 10 years (1990–1999) for Phases 0–5.**

**Reasoning:**
- The 5cond runs all used 10 years. Extending the window simultaneously with changing basin selection makes the comparison muddier — we couldn't tell whether improvements came from "more data" or from "better subset choice".
- 10 years × ~365 days × 22–51 basins = 80k–186k training (basin, window) examples per subset. Adequate for a 30-epoch run at our model size.
- Window extension is queued as a Phase 7 sensitivity check if the rest of the plan produces a solid result. Don't do it now.
- **Exception:** if Phase 2 baseline mean NSE < 0.5 (clearly under-trained at the smaller subset size), then extend the window as a recovery move. Not the default.

### Q3 — Primary metric: settled. **Mean ± std as headline; median + bootstrap CI as backing.**

**Reasoning:**
- Professor explicitly requested mean ± std. It's the simplest one-line summary for the running invariant.
- Median + bootstrap CI stays in the analysis pipeline (free to compute) for the paired contrasts where it's the more robust choice.
- Reporting both: meeting-facing summary uses mean ± std; paper tables use both.
- No code change needed in `compare_5conditions.py` — both already get computed; just promote mean to the headline.

### Q4 — Phase 4 revision list: settled. See §4 "Phase 4" (re-tiered).

**Decision summary:** original 5 revisions re-organized into Tier A (must), Tier B (next), Tier C (fallback), Tier D (deferred).

**Reasoning:**
- Original list was unordered and stuffed all changes into one phase. As CRS that's bad triage.
- **Tier A (area-weighting, drop one-hot)** are the two with clearest mechanistic hypotheses and smallest implementation cost. Test these first.
- **Tier B (MLP message, learnable scaled residual)** add capacity / fix saturation. Cheap. Test if Tier A doesn't close the gap.
- **Tier C (embedding discrete features, K-hop messages)** are good ideas but lower-leverage. Reserved for if A+B don't get us there.
- **Tier D (predict-then-route)** is a redesign that wants its own pre-registration and ~half a day of implementation. Out of scope for this 34-hr cycle; queued for the next.
- Added 4.B.2 (learnable scaled residual) which I missed in the original list — `tanh(W_out(.))` saturation is a known defect I documented in `architecture_analysis.md`, and it's a 2-line code change.

### Q5 — Paper framing if Phase 6 fails: settled. **Default to "characterizing when graph signal helps".**

**Decision:** if Phase 6 (183-basin scale-up) shows the best-revised variant still loses to L:
- Default framing: **"Identifying conditions under which graph-LSTM features help streamflow prediction"** — a *study*, not a *claim of superiority*. Reports 5cond at 183-basin (negative), local-subgraph results (positive or characterized), and the boundary between them.
- Upgrade trigger: if Phase 4–5 produces a clear local-subgraph win (mean NSE Δ ≥ +0.03 across all 4 held-out subgraphs), pivot to **"Graph features outperform standard LSTM on locally-coherent river networks"** — narrower but stronger claim. Connects to the 23-basin pilot's +0.078 NSE.
- Fall-back trigger: if even local-subgraph results are uniformly null, do not publish on this thesis. Pivot the next cycle to Tier-D (predict-then-route) as the constructive alternative.

**Reasoning:**
- "Characterizing when X helps" is intellectually honest, builds on the rigorous statistical machinery we already have, and is defensible against reviewers who will rightly demand we account for the negative 5cond result.
- It's the framing that the existing evidence + the Phase 4-5 evidence will *jointly* best support, regardless of which way Phase 6 lands.
- Workshop deadlines don't make a stronger claim more publishable; honest framing of a result that holds is.

---

## 11. Concrete first action

Phase 0 begins now (no GPU needed). Recompute mean ± std cross-seed for the existing 5cond data and add as an addendum table to `5cond_run_analysis.md`. Then write Phase-1 basin-list files. After that, GPU work begins with Phase 2.
