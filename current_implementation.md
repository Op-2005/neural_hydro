# Current Implementation — Full Scope of the Research

*A complete, plain-language picture of what this research is, how it's done, and what it has
found. Written as a catch-up reference after several autonomous CRS sessions. Every number
here is on disk and reproducible; sources are cited to the analysis files.*

**Last updated:** 2026-07-06. **Active work:** `experiments/topology_ablation/`.

---

## 1. The research question

**Does river-network structure improve LSTM streamflow prediction over a strong multi-basin
LSTM baseline — and if so, how must that structural information be delivered?**

The answer we have arrived at (and the paper's thesis):

> **Yes, but only as a *dynamic* signal.** Static topology *features* (a basin's position in
> the network, as fixed numbers) add essentially nothing. But feeding a basin the *dynamic
> upstream flow* arriving from its upstream neighbors gives a real, multi-seed-confirmed,
> mechanistically-grounded improvement — and a *deployable* version (using predicted, not
> observed, upstream flow) recovers most of that gain.**

In one line: **structure helps streamflow LSTMs when delivered as dynamic upstream flow
(routing), not as static topology descriptors.**

---

## 2. How we got here (brief — why the current design exists)

The project spent months on a **custom DirectedGraph-LSTM** (an LSTM with learned
message-passing between basins). Those experiments (the 23-basin pilot, then the 183-basin
"5-condition factorial") produced **confusing negative results** — graph features appeared to
*hurt*. A methodology audit found those negatives were **confounded** by three problems:

1. **Trainer confound** — the custom graph model used a different, undertrained training loop
   (loss still falling at 30 epochs; ~186× fewer gradient updates than the baseline trainer).
2. **Encoding redundancy** — the baseline already includes a 671-dim basin one-hot; the 5
   topology scalars were <1% of the input and had no room to matter.
3. **Noise injection** — from-scratch topology-feature weights started random and were never
   suppressed by the undertrained model.

None of those negatives were interpretable. The whole program was **restarted as a clean,
controlled ablation** (details below). The old work is preserved for provenance under
`experiments/5cond_factorial/` and `experiments/local_subgraphs/` but is *not* the current
study.

**The enabling insight:** NeuralHydrology auto-loads any `camels_attributes_v2.0/camels_*.txt`
file as static attributes, and the basin one-hot is a single config flag. So the entire study
runs on **stock NeuralHydrology `cudalstm`** — no custom model, no custom trainer — where the
only thing that varies across conditions is *one input feature*. This dissolves all three
confounds at once.

---

## 3. The data and how it is used

**Dataset:** CAMELS-US (671 US basins with daily meteorological forcings, observed streamflow,
and static catchment attributes). Located at `datasets/camels_us/`.

**Study network — "Component 0":** a connected sub-network of **183 basins** in the eastern US,
spanning 6 HUC regions, with **624 inferred directed edges** and graph depth up to 4. This was
discovered in Phase 1 by inferring upstream→downstream edges from basin distance, area ratio,
and elevation (`topology_analysis/phase1_network_discovery/`). Basins-per-depth: 33/81/51/16/2
(depths 0–4). *Caveat: edges are heuristic (distance/area/elevation), not NHDPlus
ground-truth hydrography — flagged as future work.*

**Inputs to every model (identical across all conditions):**
- **5 dynamic forcings** (daily, Maurer product): `PRCP(mm/day)`, `SRAD(W/m2)`, `Tmax(C)`,
  `Tmin(C)`, `Vp(Pa)`.
- **5 static attributes:** `elev_mean`, `area_gages2`, `slope_mean`, `p_mean`, `pet_mean`.
- **Basin one-hot encoding** (671-dim) — Kratzert-2019-style per-basin identity.

**Target:** `QObs(mm/d)` — observed daily discharge, area-normalized, clipped to ≥ 0.

**Temporal split (identical across all conditions):**
- Train: 1990-01-01 → 1999-12-31
- Validation: 2000-01-01 → 2004-12-31
- Test: 2005-01-01 → 2008-12-31 (all reported numbers are on this held-out test period)

**The one variable that changes across conditions:** a single extra *dynamic input* column,
`upstream_q` — the area-weighted mean of a basin's upstream neighbors' discharge, lagged by 1
day. Depending on the condition, this is filled with observed discharge, predicted discharge,
a shuffled control, or is absent entirely (see §5). It is built by
`experiments/topology_ablation/build_upstream_discharge_feature.py` (observed) and
`build_predicted_upstream_q.py` (predicted).

**No leakage:** `upstream_q` is *upstream* basins' discharge, lagged ≥1 day, predicting the
*downstream* basin. The downstream basin's own discharge never enters its input. Same-day
upstream→downstream flow is physical routing (sub-daily travel times at daily resolution), not
target leakage.

---

## 4. The methodology (why it is a clean, valid ablation)

**Every condition is stock NeuralHydrology `cudalstm` with a byte-identical config.** Verified
by direct config diff (`analysis/COMPLIANCE.md`): same model, hidden_size 64, output_dropout
0.4, initial_forget_bias 3, Adam @ lr 1e-3, batch_size 256, 30 epochs, seq_length 30,
predict_last_n 1, clip_gradient_norm 1, Maurer forcings, 5 static attrs, one-hot on, same
train/val/test split, same random seed. **The ONLY difference between conditions is the
presence/content of the single `upstream_q` input column.**

This is the cleanest possible ablation: one variable changes, everything else is frozen. It is
what makes "the difference-maker is our addition" *literally true at the config level* — and
it is the specific fix for the architecture confound that invalidated the earlier
DirectedGraph-LSTM work.

**Metrics (all three, per our methodology):** NSE (primary), KGE, and log-NSE (downweights
high-flow outliers). Computed per-basin on the test period; reported as cross-basin medians and
paired per-basin deltas.

**Rigor practices used throughout:**
- **Multi-seed** (seeds 11, 13, 17) with mean ± std across seeds as the tracked invariant.
- **Pre-registration** — every experiment has a `preregistration_*.md` written *before* the run,
  stating hypothesis + success + falsification criteria. Falsified hypotheses are reported as
  such (see §6, the lag0 result).
- **Null control** — a shuffled version of the signal, to separate real content from mere
  added-input capacity.
- **Confound checks** — the key mechanistic result is checked against basin size.

---

## 5. The ablation framework — the conditions

The study has two parts. **Part A** (settled early) asked whether *static* topology features
help. **Part B** (the main contribution) asks whether *dynamic* upstream flow helps.

### Part A — the encoding × topology 2×2 (static features)

Stock cudalstm; the only change is adding 5 topology static features and/or toggling the
one-hot:

| Condition | one-hot | topology features |
|---|---|---|
| L | ✓ | — |
| L+T | ✓ | ✓ (graph_depth, n_upstream, total_upstream_area, in_degree, frac_upstream_area) |
| L_noID | ✗ | — |
| L_noID+T | ✗ | ✓ |

### Part B — the upstream-flow conditions (dynamic signal)

Stock cudalstm; the only change is the `upstream_q` dynamic input:

| Condition | `upstream_q` content | What it tests |
|---|---|---|
| **L** | (absent) | Baseline |
| **L+upQ (oracle)** | *observed* upstream discharge, lag 1 | Upper bound: does upstream flow carry signal at all? |
| **L+upQ_pred (realizable)** | *predicted* upstream discharge, lag 1 | The deployable version — no ground truth at inference |
| **L+upQshuf (null)** | observed upstream Q, shuffled in time | Capacity control — is it the signal, or just an extra input? |

Supporting conditions run for robustness: `L+upPrecip` (upstream precipitation instead of
discharge), `L+upQ` at lag 0 and lag 2 (travel-time sensitivity), and `L+upQ_pred` at lag 0.

**The realizable model is two-stage and fully deployable:** (1) train the L baseline; run it
over the full 1990–2008 span to *predict* every basin's discharge from its own forcings (the
`_Lfullspan_eval` step); (2) aggregate each basin's *predicted* upstream discharge as the
`upstream_q` input for a downstream model. At inference, no observed discharge is needed.

---

## 6. Experimentation and results

The arc, in order, with test-period numbers (Component 0, 183 basins).

### Result 1 — static topology features add nothing (Part A 2×2, single seed)

| Condition | median NSE |
|---|---|
| L | 0.653 |
| L+T | 0.654 |
| L_noID | 0.633 |
| L_noID+T | 0.625 |

- Topology benefit **with** one-hot (L+T − L): **−0.001** (nothing)
- Topology benefit **without** one-hot (L_noID+T − L_noID): **+0.003** (nothing)
- Per-basin distributions are symmetric noise (~⅓ up, ~⅓ down).

**Finding:** static "where am I in the network" features are inert — with *or* without the
one-hot. The redundancy hypothesis (one-hot masking topology) was falsified; the features are
simply weak. Minor real result: the one-hot itself is worth +0.012 NSE.

### Result 2 — dynamic upstream discharge helps (the oracle, single seed → then multi-seed)

| Condition | median NSE | Δ vs L |
|---|---|---|
| L | 0.653 | — |
| **L+upQ (observed, oracle)** | **0.703** | **+0.037** (67% of basins improve) |

**Finding — the pivotal result.** Static topology failed not because structure is
uninformative, but because a *constant scalar* can't carry the signal. Given the actual
*dynamic* upstream flow, the model gains real skill. This is an **upper bound** (uses observed
upstream discharge).

**Stress-tests (all passed):**
- **Shuffled-Q null control: −0.002** → the gain is real upstream content, not added capacity.
- **Upstream precipitation: +0.012** (⅓ of the discharge gain) → routed *flow* carries content
  beyond upstream *rain*; the signal isn't reducible to "add upstream precipitation."
- **Lag sweep: lag0 +0.087 / lag1 +0.037 / lag2 +0.036** → positive at all lags (no leakage
  signature); same-day upstream flow is most informative (short travel times).

### Result 3 — the realizable (predicted-Q) version works, 3 seeds (`analysis/MULTISEED.md`)

| Condition | mean ± std median NSE (seeds 11/13/17) |
|---|---|
| L | 0.653 ± 0.002 |
| L+upQ (oracle) | 0.691 ± 0.009 |
| **L+upQ_pred (realizable)** | **0.678 ± 0.008** |
| L+upQshuf (null) | 0.666 ± 0.008 |

Paired Δ vs L, per seed:
- Oracle: +0.037 / +0.047 / +0.021 → mean **+0.035**
- **Realizable: +0.027 / +0.026 / +0.013 → mean +0.022, all 3 seeds positive**
- Null: −0.006 / +0.009 / +0.004 → mean +0.003

**Finding:** predicted upstream Q (deployable, no ground truth) **recovers ~55–72% of the
oracle ceiling**, multi-seed confirmed, all seeds positive. Structure isn't just an upper
bound — it's a *working method*.

### Result 4 — the mechanism is routing, confound-checked (`analysis/CONFOUND.md`)

Realizable gain by graph depth (pooled 3 seeds):

| depth | n | median realizable Δ |
|---|---|---|
| 0 (headwater) | 99 | +0.002 |
| 1 | 243 | +0.020 |
| 2 | 153 | +0.031 |
| 3 | 48 | +0.044 |

The gain **rises monotonically with graph depth** — headwaters (no upstream) get ~zero, deep
downstream basins gain most. This is the **routing signature**: basins benefit in proportion to
how much upstream flow they receive.

**Confound check (depth correlates with basin size, r=0.38):**
- **T3 (partial control):** within *every* area tercile, depth≥2 beats headwaters (+0.021 /
  +0.036 / +0.050 for small/mid/large) — **3/3 terciles pass**; the effect is *strongest* in
  large basins, the opposite of a size confound.
- **T4:** corr(Δ, area) = **+0.015 (~0)** vs corr(Δ, depth) = +0.158. Area doesn't predict the
  gain; graph position does.

**Finding:** the depth gradient is genuinely *upstream routing*, not basin size. Mechanistically
grounded.

### Result 5 — metric robustness + not-a-straw-man (`analysis/COMPLIANCE.md`)

- **3-metric robustness:** realizable Δ = **NSE +0.022, log-NSE +0.027**, both all-seeds-positive;
  the null goes *negative* in log-NSE (−0.003), sharpening the contrast. **KGE +0.013 mean but
  one seed dips slightly negative** → honestly scoped as "robust in NSE and log-NSE;
  KGE-positive-on-average with seed sensitivity."
- **Not baseline-rescue:** the gain persists on already-well-predicted basins (L NSE > 0.6):
  +0.012 (n=342). Larger on bad basins (+0.24 on the worst) but positive everywhere → real
  structural signal, not patching a weak baseline.

### Result 6 — the predictability ceiling (lag0-realizable, falsified but informative)

We hypothesized that predicted-Q at **lag 0** (where the *observed* oracle is 2× stronger,
+0.087) would beat the lag1 realizable headline. **Falsified:** lag0-predicted Δ = +0.023 <
lag1-predicted +0.027. The reason is the finding: predicted-Q recovers only **26%** of the
lag0 oracle vs **72%** of the lag1 oracle. **Same-day upstream flow carries the most signal
when observed but is the hardest to *forecast* — lag 1 is the realizable sweet spot; the
deployable gain is capped by upstream *predictability*, not the downstream model.** lag1 stays
the headline.

---

## 6.5 How it all comes together — the paper as one clean narrative

Read end to end, the results form a single story with a clear beginning, turn, and payoff — the
shape of a publishable paper:

**Setup (the tension).** River basins form a physical network: water flows downstream, so a
basin's discharge *should* depend on its upstream neighbors. Yet strong multi-basin LSTMs treat
every basin independently and still match physics-based models (Kratzert 2019) — and when people
add the river network as graph structure, it mysteriously *doesn't help* (Kirschstein 2024). Is
the network genuinely uninformative, or are we adding it the wrong way?

**The controlled test (the method).** We answer this with the cleanest possible ablation: stock
NeuralHydrology `cudalstm`, byte-identical across all conditions, where the *only* thing that
changes is a single input carrying network information. This removes every confound (architecture,
training, encoding) that muddied prior attempts — so any difference is attributable to the network
signal alone.

**The turn (the key insight).** We add the network two ways. As **static topology features** (a
basin's fixed position in the graph) → **nothing** (~0 NSE, with or without the basin one-hot).
But as **dynamic upstream flow** (the actual water arriving from upstream) → a **real gain**
(+0.037 NSE oracle upper bound). *Same network, opposite outcomes.* The reason a static descriptor
fails is that it's a constant; the reason dynamic flow works is that it carries time-varying state.
**This one contrast explains Kirschstein's null in a single stroke: the information is in the
structure-as-flow, not the structure-as-label.**

**The payoff (it's real, deployable, and mechanistic).** The gain is not an artifact and not a
one-off:
- **Real** — beats a shuffled null control; survives to a third metric (log-NSE); persists even on
  already-well-predicted basins.
- **Deployable** — a two-stage model that *predicts* upstream flow from forcings (no ground truth
  at inference) recovers ~55–72% of the oracle ceiling, confirmed across 3 seeds, all positive.
- **Mechanistic** — the gain rises monotonically with graph depth and is zero for headwaters,
  confirmed independent of basin size. This is the *routing signature*: it works because water
  routes downstream, exactly as the physics says — realizing the physics-aware direction Jiang
  2025 pointed to.

**The honest boundary (the discussion).** We also map where it stops: the deployable gain is
capped not by the model but by how well upstream flow can be *forecast* (the lag-0 vs lag-1
result), and the scope is a regional eastern-US network, not a national benchmark.

**One-sentence thesis.** *River-network structure improves LSTM streamflow prediction when — and
only when — it is delivered as a dynamic upstream-flow signal rather than a static topological
descriptor; a deployable model captures most of this routing-driven gain, resolving why static
graph approaches have failed.*

**Paper section mapping.** Intro → the tension above (§1–2). Methods → the byte-identical stock
ablation + data (§3–5). Results → static null → dynamic gain → deployable → routing mechanism
(§6). Discussion → predictability ceiling, scope, and positioning vs. Kirschstein/Jiang (§7–8).
The controlled ablation framework is the methodological contribution that carries alongside the
empirical result (§9).

---

## 7. Honest caveats and scope

- **Regional, not national.** 183 basins in the eastern US, 6 HUC regions — adequate for a
  regional/workshop study, **not** a national CAMELS benchmark. The baseline (median NSE 0.653)
  is a legitimate stock `cudalstm`, *not* an EA-LSTM/531-basin SOTA (~0.74). **No SOTA claim.**
- **The oracle is an upper bound.** The +0.035 oracle gain uses observed upstream discharge; the
  *deployable* number is the realizable +0.022. Always framed against the ceiling.
- **KGE is the weakest of the three metrics** (positive on average, one seed slightly negative).
  Reported honestly.
- **Heuristic edges**, not NHDPlus ground-truth hydrography. Named as future work.
- **3 seeds, 1 run each** — enough for stable medians and sign agreement, not a large-N sweep.

---

## 8. Positioning vs. the cited literature

- **Kratzert et al. 2019 (HESS):** established the strong multi-basin LSTM baseline (with basin
  encoding + static attributes) that any structure-aware method must beat. Our L condition is
  this baseline, at regional scope.
- **Kirschstein & Sun 2024 (ICML):** found GNNs on river-network adjacency give **no** benefit —
  an unexplained null. **We explain it:** static topology/adjacency *is* inert (our Part A). The
  information isn't in the structure-as-label; it's in the structure-as-dynamic-flow.
- **Jiang et al. 2025 (ICML):** a physics-aware directional operator recovers gains where plain
  GNNs fail. **We execute that direction concretely:** dynamic upstream flow (routing) is what
  helps, and the benefit scales with graph depth exactly as physical routing predicts.

The study thus **resolves the Kirschstein null and operationalizes the Jiang direction** — it is
positioned in the literature, not floating.

---

## 9. The ablation framework as a contribution

Beyond the specific result, the **controlled ablation methodology itself** is a contribution:
a confound-free, pre-registered ladder on a stock trainer where structural information is added
as a single input, tested against a null control and an oracle upper bound. It is *why* the
+0.022 is credible, and it is reusable by others studying structure-aware hydrology models.

---

## 10. Current status and what's next

**Status: the core study is complete, clean, and publication-valid for a regional workshop
paper.** The full evidence chain:

> static topology null → dynamic upstream flow helps (multi-seed, all seeds positive) →
> deployable via *predicted* upstream Q at lag 1 (the predictability-optimal lag) → via
> *routing* (depth gradient, confound-checked against area) → robust in NSE and log-NSE
> (KGE positive-on-average) → not baseline-rescue → on a byte-identical stock-cudalstm ablation.

**Queued next steps (none are load-bearing for the core claim):**
1. **Paper skeleton** — the science is complete; the natural next move.
2. **Local-subgraph scale curve** — does the gain grow on small locally-coherent networks
   (subgraph + shortest-path-walker machinery already built). A strong secondary figure.
3. **531-basin scale-up / NHDPlus edges** — only needed to target a top-tier (vs workshop)
   venue; large compute.

**Where to read more:**
- Running decision log + every result with reasoning: `JOURNAL.md`
- Quick status brief: `updates.md`
- The study's own README + per-analysis pre-registrations and outputs:
  `experiments/topology_ablation/` (`README.md`, `preregistration_*.md`, `analysis/*.md`)
- Run outputs: `runs/topology_ablation/component0/` (`NOTES.md`)
