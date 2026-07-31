# Methodology Section — end-to-end plan + mathematical framework

Grounded against code this session: `build_upstream_discharge_feature.py`,
`build_predicted_upstream_q.py`, `discover_network.py`, `neuralhydrology/modelzoo/cudalstm.py`,
`inputlayer.py`, `head.py`, and the exact config. Every constant below is read from source, not
recalled. This is a PLAN (structure + the math to write), not final prose.

**Governing skills:** structure/prose = `ml-paper-writer`; the equations = `ml-math-rigor`.
**Honesty ceiling (from /crs):** the feature is a *fixed, directed, 1-hop precompute* — NOT message
passing, NOT a learned operator. The math must reflect exactly that depth and no more.

---

## PART 1 — Methodology section, subsection by subsection

Target ~1.5–2 pages. Order chosen so each subsection sets up the next; the "one input changes"
claim is made formally in 4.2 and everything else is standard-but-precise scaffolding around it.

### 4.1 Problem setup and notation
- One paragraph fixing notation used everywhere after. Define: basin index set, per-basin daily
  inputs, target, the prediction map, the sequence window. Introduce the study network as a DAG.
- **Purpose:** every later symbol is defined here first (ml-math-rigor: introduce-before-use).

### 4.2 The controlled-ablation design (the methodological contribution)
- State the design principle in words first: every condition is the *same* stock model with a
  *byte-identical* training configuration; the ONLY thing that differs across conditions is one
  scalar input channel. Then make it formal with the input-vector equation (Eq. 2 below), where the
  conditions are literally different instantiations of a single coordinate.
- Name what this buys: it removes the architecture / trainer / encoding confounds that muddied
  prior graph-LSTM attempts, so any measured difference is attributable to that one input.
- **Cite:** `kratzert2022joss` (the NeuralHydrology framework / model zoo we run on),
  `kratzert2019` (the multi-basin LSTM baseline this design instantiates).

### 4.3 The base model (what we build on — the NH model-zoo cudalstm)
- Describe the stock model precisely and cite it. `cudalstm` = InputLayer (concatenation of
  dynamic + static + basin-identity inputs; optional embedding, unused here) -> single-layer LSTM
  (Eq. 3, standard gates) -> linear regression head (Eq. 4). Trained with MSE loss, Adam.
- Exact hyperparameters in a config table (Table M1): hidden 64, seq_length 30, predict_last_n 1,
  output_dropout 0.4, initial_forget_bias 3, Adam lr 1e-3, batch 256, 30 epochs,
  clip_gradient_norm 1, MSE loss, basin one-hot on. All 3 conditions share this byte-for-byte.
- **Cite:** `kratzert2022joss` (software), and note this is the standard LSTM of `kratzert2019`.
- **ml-math-rigor note:** write the LSTM gate equations ONLY if they earn space; more likely cite
  them as standard (Hochreiter-Schmidhuber / kratzert2018) and give the *interface* (Eq. 3–4), not
  a re-derivation. Over-formalizing a stock LSTM reads as padding.

### 4.4 Data and study network
- CAMELS US attributes v2.0 (cite `addor2017` primary; `newman2015` forcings; `newman2014data` data
  product in Data availability). 5 Maurer forcings, 5 static attrs, 671-dim basin one-hot; target
  QObs(mm/d). Split train 1990–99 / val 2000–04 / test 2005–08.
- The study network: Component 0 = 183-basin connected eastern-US sub-network. Edge inference rule
  (Eq. 1): a directed edge j->i when child area >= 1.5 x parent area AND child lower in elevation
  AND within 150 km. State plainly that these are HEURISTIC edges, not NHDPlus hydrography — the
  robustness section shows the result does not depend on their density.
- Graph object: DAG G=(V,E), depth defined recursively (Eq. 5).

### 4.5 The structural signal: the upstream-flow feature (THE core equation)
- This is the one real piece of math. Define the area-weighted upstream-flow aggregate u_i(t)
  (Eq. 6): area-weighted MEAN of immediate-upstream parents' discharge, lagged tau=1 day. Every
  symbol from 4.1/4.4.
- State its nature explicitly (honesty ceiling): a FIXED, DIRECTED, SINGLE-HOP precompute over
  immediate graph predecessors — not a learned aggregation, not message passing, not iterated over
  the graph. This is the sentence that both scopes the claim and sharpens the contribution.
- The conditions are substitutions for the discharge term q_j in Eq. 6 (Table M2):
  observed (oracle) / predicted (realizable) / shuffled-in-time (null) / reversed-edges /
  random-rewire. This makes the whole experiment ONE equation with instantiations.
- No-leakage argument (Eq. 6 uses upstream, lagged >= 1 day; target basin's own flow never enters).

### 4.6 The deployable two-stage model (predicted-Q / realizable)
- Formalize the two stages (Eq. 7): Stage 1, the trained L baseline predicts each basin's discharge
  from its own forcings over the full span; Stage 2, substitute those predictions q_hat_j into
  Eq. 6 to form the realizable feature; train the downstream model on it. No ground-truth discharge
  at inference. State that this is what makes the method DEPLOYABLE.

### 4.7 Evaluation
- Metrics (Eq. 8–10): NSE (primary), KGE, log-NSE — define each exactly as computed in code
  (verify against COMPLIANCE.md/METRIC_HONESTY.md; NSE per-basin on test, cross-basin median).
- Paired per-basin delta Delta_i (Eq. 11) as the tracked contrast; cross-basin median + one-sided
  Wilcoxon; multi-seed (11/13/17) mean +/- std. Pre-registration mentioned (protocol, appendix).

---

## PART 2 — The mathematical framework (what ml-math-rigor will solidify)

The equations, in write order, each with its purpose. "Match formalism to claim strength" — the
only non-standard object is Eq. 6; everything else is precise scaffolding, not invented depth.

### Notation (fixed once, §4.1)
- V = set of basins (|V| = 183 for Component 0); i, j index basins.
- x_i(t) in R^d : per-timestep input vector for basin i at day t.
- m_i(t) in R^5 : dynamic forcings (PRCP, SRAD, Tmax, Tmin, Vp).
- s_i in R^5 : static catchment attributes.
- e_i in {0,1}^671 : basin one-hot identity encoding.
- q_i(t) in R_{>=0} : observed discharge (mm/d) of basin i on day t; y_i(t) = q_i(t) the target.
- q_hat_i(t) : model-predicted discharge.
- L = seq_length = 30 ; predict_last_n = 1.
- G = (V, E) : the inferred river DAG. P(i) = { j : (j->i) in E } = immediate upstream parents.
- A_i : drainage area (area_gages2) of basin i. tau = 1 day (lag).

### Eq. 1 — edge inference (defines E)
(j -> i) in E  <=>  A_i >= rho A_j  AND  elev(i) < elev(j)  AND  dist(i,j) <= D_max,
with rho = 1.5 (area ratio), D_max = 150 km. [grounded: discover_network.py lines 23-24]
- Purpose: makes the graph reproducible; flags edges as heuristic.

### Eq. 2 — the input vector (THE "one variable changes" formalization)
x_i(t) = [ m_i(t)  ||  s_i  ||  e_i  ||  u_i(t) ]  in R^{d},  d = 5 + 5 + 671 + 1 = 682.
- The four blocks are concatenated (NH InputLayer concatenates when no embedding is configured;
  ours isn't). Across ALL conditions, m, s, e are identical; only the last coordinate u_i(t)
  changes. This single equation IS the controlled-ablation contribution made formal.
- Baseline L: the u_i(t) coordinate is absent (d = 681); every +upQ condition adds exactly it.
- ml-math-rigor: state the concatenation explicitly; do NOT dress it as anything richer.

### Eq. 3-4 — the base model interface (cite as standard, keep minimal)
h_i(t) = LSTM(x_i(t-L+1 : t)) ;  q_hat_i(t) = W h_i(t) + b   (linear regression head).
- Cite the LSTM (kratzert2018 / Hochreiter-Schmidhuber) rather than re-deriving gates.
- Purpose: fixes the interface so Eq. 2's u_i(t) has a place to act; nothing more.

### Eq. 5 — graph depth (defines the routing-signature variable)
depth(i) = 0 if P(i) = empty ; else 1 + max_{j in P(i)} depth(j).
- Purpose: depth is the stratifier for the routing-signature result (Results). Edge case named:
  headwaters (P(i) empty) have depth 0 and u_i(t) = 0.

### Eq. 6 — the upstream-flow feature (THE core equation)
             sum_{j in P(i)} A_j * q_j(t - tau)
u_i(t)  =  -----------------------------------------      (0 if P(i) = empty)
                  sum_{j in P(i)} A_j
- Area-weighted MEAN (not sum) of immediate parents' lagged discharge. [grounded:
  build_upstream_discharge_feature.py — divide by wsum]. Mean keeps u in mm/d, O(target scale),
  so it survives NH per-feature standardization without one large basin dominating.
- Edge cases: headwaters -> 0 (empty parent set); a parent with missing discharge is dropped from
  both numerator and wsum (still a valid weighted mean over available parents).
- Dimensional check: A_j q_j has units km^2 * mm/d; dividing by sum A_j returns mm/d = units of the
  target. Consistent. [ml-math-rigor correctness audit passes]
- THE nature statement (honesty): fixed weights (area), fixed lag, computed ONCE before training,
  over IMMEDIATE predecessors only. A directed 1-hop gather. Not learned; not message passing.

### Eq. 6-conditions — the substitutions (Table M2)
u_i^{oracle}(t)     : q_j = observed q_j                          (upper bound)
u_i^{real}(t)       : q_j = q_hat_j from Stage 1 (Eq. 7)          (deployable)
u_i^{null}(t)       : q_j = observed q_j, permuted in time pi(t)  (capacity control)
u_i^{rev}(t)        : P(i) from REVERSED edges (downstream)       (directionality control)
u_i^{rand}(t)       : P(i) = degree-preserving random rewire      (topology-specificity control)
- All five are the SAME Eq. 6 with q_j or P(i) swapped. One equation, the whole experiment.

### Eq. 7 — the two-stage realizable model
Stage 1:  q_hat_j(t) = f_theta_L( m_j, s_j, e_j )  for all j, all t in [1990, 2008]
          (the trained L baseline evaluated over the full span).
Stage 2:  u_i^{real}(t) = Eq. 6 with q_j := q_hat_j ; train f_theta on x_i with this coordinate.
- Deployable: inference needs only forcings, never observed discharge. [grounded:
  build_predicted_upstream_q.py full_span_predictions]

### Eq. 8-10 — metrics (define exactly as computed)
NSE_i   = 1 - sum_t (q_i - q_hat_i)^2 / sum_t (q_i - mean(q_i))^2
KGE_i   = 1 - sqrt( (r-1)^2 + (beta-1)^2 + (gamma-1)^2 ),  r,beta,gamma the corr/bias/variability.
logNSE_i: NSE on log(q + eps), eps = 1e-3 * mean-flow. [grounded: METRIC_HONESTY.md; verify eps]
- Correctness: verify each against COMPLIANCE.md before writing; NSE in (-inf, 1], per-basin, test.

### Eq. 11 — the paired contrast (the tracked quantity)
Delta_i^{(c)} = NSE_i(condition c) - NSE_i(L) ;  report median_i Delta_i, one-sided Wilcoxon vs 0,
mean +/- std over seeds {11,13,17}.
- Purpose: every Results claim is a statement about median Delta and its p-value.

---

## PART 3 — Reviewer 2 (the math/methods a hostile reviewer attacks)

- *"Is this just message passing rebranded?"* No — Eq. 6 is a fixed, pre-computed, single-hop,
  area-weighted mean with no learned parameters and no iteration over the graph. Stated in §4.5 and
  visible in the equation (no learned weights, no depth recursion in the feature). This is the
  whole point: a non-learned feature recovers what learned GNNs did not.
- *"Why area-weighted MEAN not SUM?"* Grounded design choice: mean keeps u_i in mm/d at target
  scale so NH's per-feature standardization is well-behaved and no single large parent dominates.
  State it in one sentence in §4.5.
- *"Leakage via same-day upstream flow?"* tau >= 1 day; target basin's own discharge never enters
  its input; the realizable model uses predicted, not observed, upstream Q. §4.5 + §4.6.
- *"Byte-identical really?"* Yes — one config, one coordinate differs (Eq. 2). Config table M1 +
  COMPLIANCE.md diff. The design's credibility rests on this.
- *"cudalstm is not your model — what's novel?"* Correct, and stated: the novelty is the
  controlled-ablation *finding* (structure-as-flow vs structure-as-label) + the deployable
  two-stage feature, NOT a new architecture. Honest framing is the strength.

---

## PART 4 — Build order + what to cite where

1. Write §4.1 notation (fix symbols).  2. §4.4 data + Eq. 1/5 (graph).  3. §4.3 base model + Table
M1 + cite kratzert2022joss/kratzert2019/kratzert2018.  4. §4.2 design + Eq. 2 (the contribution).
5. §4.5 Eq. 6 + conditions (the core).  6. §4.6 Eq. 7 (deployable).  7. §4.7 Eq. 8-11 (eval).
Then run ml-math-rigor: notation audit, correctness audit (dimensional + metric-def check),
flow audit. Verify metric eps + KGE decomposition against METRIC_HONESTY.md before finalizing.

**Tables in Methodology:** M1 (config, byte-identical), M2 (the 5 feature conditions as
substitutions in Eq. 6). Results tables live in §Results, not here.

**[verify] items — ALL RESOLVED this session (verified against source):**
- [x] InputLayer CONCATENATES for our config (no `*_embedding` keys present; embedding net is
      identity → concatenation). Eq. 2's concatenation is literally correct.
- [x] loss = MSE (config `loss: MSE`), not NSE-loss. Eq. 3–4 interface uses MSE + Adam.
- [x] log-NSE eps = eps_frac × (per-basin mean observed flow), eps_frac ∈ {1e-2,1e-3,1e-4},
      headline at 1e-3. Eq. 10 is unit-aware. [analyze_metric_honesty.py:40-42]
- [x] 5 static attrs confirmed exactly: elev_mean, area_gages2, slope_mean, p_mean, pet_mean.

Nothing left unverified — drafting can proceed with zero unknowns.
