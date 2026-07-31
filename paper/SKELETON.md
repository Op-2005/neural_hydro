# Paper Skeleton — planning map (not prose)

Grounding: every number below is copied from `experiments/topology_ablation/analysis/{PAPER_TABLE,
MECHANISM_MULTISEED,ROUTING_BASELINE_3SEED}.md` and `current_implementation.md §3–5`. Sources
named per row. Nothing recalled.

**Target venue (working):** ML-for-science workshop OR domain journal (HESS/WRR). Regional scope is
normal there. Frame honestly; do NOT claim SOTA. (Top-tier main track = future, gated on national
scale-up — out of scope for this draft.)

---

## The ONE contribution sentence

> River-network structure improves LSTM streamflow prediction **only when delivered as a dynamic
> upstream-flow signal, not as a static topological descriptor** — and this dynamic signal is
> **specific to the real river network** (not any spatial flow aggregate), robust to graph density,
> and recoverable by a deployable predict-then-route model — which explains why static graph-neural
> approaches have not helped.

## The 1–3 claims (each with its evidence cell + strength)

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | Static topology features add ~0; dynamic upstream flow adds a real gain | Table 1: L 0.653 vs L+upQ_pred 0.678 (Δ+0.025, p=6.0e-19); static-feature 2×2 ~0 (RESULTS.md) | 3-seed, significant |
| C2 | The gain is **topology-specific** — the real river graph, not generic flow | MECHANISM_MULTISEED §1: forward−random +0.034, p=2.3e-14; random alone n.s. (p=0.09 vs L via 8.7e-4? see note) | 3-seed, significant |
| C3 | The gain is **deployable** (predicted-Q) and **robust** (graph pruning, no-ML baseline, metrics) | Table 1 realizable +0.025; Table 2 R1/R2; Table 4 + MECHANISM §2 k=2 +0.025≈full +0.026 | 3-seed (k=2 now 3-seed) |
| — | Directionality (mild preference, NOT a claim) | MECHANISM §1: forward−reversed +0.006, one-sided p≈0.03, seed-fragile | reported honestly, NOT headlined |

**Reviewer-risk note (from /crs):** C1's effect is modest (+0.025 NSE) on a non-SOTA baseline
(0.653, not the ~0.74 EA-LSTM national SOTA). Defense = the *contrast* (static null vs dynamic
gain) + topology-specificity + the Kirschstein explanation, NOT the number. Lead with insight.

---

## Section-by-section plan

### Title (draft options — pick later)
- "Structure as Flow, Not Label: When River-Network Topology Helps Streamflow LSTMs"
- "Topology-as-Flow: A Controlled Ablation of River-Network Structure in Streamflow Prediction"

### Abstract (5 moves — write last, after Results)
did → why matters → how → evidence → strongest result. Strongest result = the static-null /
dynamic-gain contrast + topology-specificity (p=2e-14). Modest number stated honestly.

### 1. Introduction (6 moves)
1. Multi-basin LSTMs match physics models yet treat basins independently [Kratzert 2019].
2. Adding the river network via GNNs has not helped [Kirschstein 2024] — an unexplained null.
3. Our idea: the network may be informative *as dynamic flow* even if inert *as a static label*.
4. Contributions (bulleted = C1/C2/C3 above).
5. Evidence one-liners (the pooled p-values).
6. Scope: regional (183 eastern-US basins), no SOTA claim, heuristic edges.

### 2. Related Work (context+difference, not a dump)
- Kratzert 2019 (HESS) — the strong per-basin LSTM baseline our L is. [CITE grounded]
- Kirschstein & Sun 2024 (ICML, PMLR 235, pp.24713–24725) — GNNs on river adjacency give no
  benefit; "does not explain why." We explain it. [CITE grounded]
- Jiang et al. 2025 (ICML, PMLR 267) — physics-aware directional operator recovers gains; we
  operationalize the direction as a plain feature, no message passing. [CITE grounded]
- One line each on GNN-theory (structure helps in can't-memorize regime) [CITE? Kipf-Welling].

### 3. Data and Setup (§3 grounded from current_implementation)
- CAMELS-US; Component 0 = 183 basins, 6 HUC regions, 624 heuristic directed edges (distance/area/
  elevation), depth ≤4. Edges heuristic, not NHDPlus — stated here.
- Inputs: 5 Maurer forcings, 5 static attrs, 671-dim basin one-hot. Target QObs(mm/d).
- Split: train 1990–99 / val 2000–04 / test 2005–08 (all numbers test).

### 4. Methods (MATH MODE — invoke ml-math-rigor when writing)
- The controlled ablation: stock cudalstm, byte-identical config (hidden 64, dropout 0.4,
  forget-bias 3, Adam 1e-3, batch 256, 30 epochs, seq 30, predict_last_n 1). ONLY the `upstream_q`
  input varies. [equation for the input vector — the "one variable changes" made formal]
- The upstream-flow feature (the ONE real equation):
  u_i(t) = [ Σ_{j∈P(i)} A_j q_j(t−τ) ] / [ Σ_{j∈P(i)} A_j ],  P(i)=immediate upstream parents,
  A_j=drainage area, τ=1 day. Conditions = substitutions for q_j (observed / predicted / shuffled /
  reversed-edges / random-rewire). State plainly: fixed, directed, 1-hop precompute — NOT message
  passing. [math-rigor: notation audit, "match formalism to claim strength"]
- No leakage: upstream, lagged ≥1 day. Deployable two-stage: predict q̂_j from forcings, then route.
- Metrics: NSE (primary), KGE, log-NSE — define each; verify against code (COMPLIANCE.md).

### 5. Experiments (each answers a question)
E1 static vs dynamic (C1) · E2 null control (shuffled) · E3 deployable predicted-Q · E4 no-ML
routing baseline · E5 depth gradient · E6 topology-specificity (reversed/random) · E7 graph-density
robustness (k=2). Pre-registered; multi-seed 11/13/17.

### 6. Results (tables already built — interpret, don't paste)
- **R-Table 1** = PAPER_TABLE Table 1 (conditions × metrics × Δ,p). Interpret: static null, dynamic
  gain, realizable recovers most of oracle, beats shuffled null.
- **R-Table 2** = routing baselines (use 3-SEED version from ROUTING_BASELINE_3SEED: R1 +0.324,
  R2 +0.664±0.008, realizable +0.683±0.008). Interpret: ML beats naive routing; honest R2 margin.
- **R-Table 3** = depth gradient (Table 3). Interpret: routing signature, per-stratum significant.
- **R-Table 4** = topology-specificity (MECHANISM §1): forward +0.046 / reversed +0.031 / random
  +0.012; forward−random +0.034 p=2e-14. Interpret: THE headline mechanism result.
- **R-Table 5** = graph robustness (MECHANISM §2): k=2 realizable +0.025 ≈ full +0.026, 3-seed.
- Directionality: one honest paragraph — mild preference, seed-fragile, not a claim.

### 7. Discussion
- Resolves Kirschstein null (topology-as-label inert; topology-as-flow specific). Operationalizes
  Jiang direction without message passing. The transferable principle (structure-as-dynamic-state
  vs static-label) — stated as a hypothesis this instantiates, NOT overclaimed to general domains.

### 8. Limitations (honest — a strength per the report)
Regional (183 basins, eastern US), not national SOTA. Heuristic edges (robust to density, but not
NHDPlus ground-truth). Modest effect size. Directionality only a weak aggregate preference. k=2 was
the last single-seed item, now 3-seed. 3 seeds, 1 run each.

### 9. Conclusion
One paragraph: the contribution sentence, restated, + the one-line forward pointer (national
scale-up / NHDPlus edges).

### Appendix
Full per-seed tables, confound checks (area, feature-magnitude), eps-sensitivity, KGE
decomposition, pre-registrations, reproducibility statement + compute note.

---

## Figure/Table plan
- **Fig 1** (core idea): the two-panel contrast — static topology feature → ~0; dynamic upstream
  flow → gain. The "same structure, opposite outcome" diagram. Communicates the whole paper.
- **Fig 2**: depth gradient (Δ vs graph depth) — the routing signature. Data: Table 3.
- **Fig 3**: topology-specificity bar (forward vs reversed vs random). Data: MECHANISM §1.
- **Table 1**: conditions × metrics (PAPER_TABLE Table 1).
- **Table 2**: no-ML routing baselines (3-seed).
- **Table 3**: graph-robustness (k=2, 3-seed).

## Open TODOs before full draft (flagged, not faked)
- [ ] Confirm the static-feature 2×2 numbers from RESULTS.md for C1 (read before writing E1).
- [ ] Reconcile PAPER_TABLE Table 4 (single-seed k=2) → replace with MECHANISM_MULTISEED §2
      3-seed numbers when writing R-Table 5.
- [ ] Routing Table 2: use the 3-SEED numbers, not the single-seed Table-2-in-PAPER_TABLE.
- [ ] Author list, affiliations, funding — [TBD by user].
- [ ] Bibliography: 4 grounded cites confirmed (Kratzert19, Kirschstein24, Jiang25); Kipf-Welling
      [CITE? verify] ; any others [CITE?].
