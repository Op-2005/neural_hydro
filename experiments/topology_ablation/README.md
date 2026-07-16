# Topology Ablation — The Controlled Restart

**Created 2026-06-20.** This is the methodologically-clean restart of the graph-LSTM
program, built after diagnosing that every prior result was confounded.

## Why we restarted

The 5cond factorial and the local-subgraph runs all showed graph features (topology,
message passing) *hurting* relative to a plain LSTM — including the impossible-looking
**G+T+M < G**. Adding inputs to a model should, at worst, be ignorable. For added
features to make a model *worse*, something is broken. We found three compounding
confounds:

1. **Architecture/trainer confound.** The custom `DirectedGraphLSTM` (Python timestep
   loop, its own training loop) is undertrained (loss still falling at 30 epochs) and
   not GPU-accelerated. Every comparison against NH's well-tuned `cudalstm` was
   apples-to-oranges.
2. **Encoding redundancy.** NH's strong baseline uses a 671-dim basin one-hot
   (`use_basin_id_encoding: True`). That lets the LSTM memorize each basin's behavior.
   The 5 topology scalars are <1% of the static input and informationally redundant
   with the one-hot — so topology features *cannot* help, by construction.
3. **From-scratch noise injection.** With `--no-warm-start`, the topology feature
   columns of the input matrix start at random init and an undertrained model never
   suppresses them → added features actively hurt.

None of our negative results were clean. We don't actually know what a correctly-trained,
fairly-compared graph model does.

## The fix: test topology features on STOCK NeuralHydrology

The key realization: **NH auto-loads any `camels_attributes_v2.0/camels_*.txt` file as
static attributes, and the basin one-hot is a single config flag.** So the entire
topology-feature question runs on stock `cudalstm` — NH's well-tuned, GPU-native, fully
trained pipeline — with *zero custom model code*. All three confounds above vanish for
the topology question (message passing, which genuinely needs custom code, is deferred
to a later phase, gated on this one).

## The foundational experiment: encoding × topology 2×2

|  | topology OFF | topology ON |
|---|---|---|
| **one-hot ON** | L | L+T |
| **one-hot OFF** | L_noID | L_noID+T |

| Contrast | Question | Prediction |
|---|---|---|
| `(L+T) − L` | Does topology help the standard model? | ≈ 0 (redundant with one-hot) |
| `(L_noID+T) − L_noID` | Does topology help when the model can't memorize identity? | **> 0 — the headline** |
| interaction | Is the topology benefit modulated by the encoding? | < 0 (topology helps more w/o one-hot) |
| `(L − L_noID)` | What does the one-hot itself buy? | > 0 (memorization is powerful) |

This is grounded in GNN theory: Kipf & Welling (2017) and the broader GCN literature show
graph structure helps most in the **can't-memorize / low-label regime**. The basin
one-hot is the *can-memorize* regime where structure is theoretically least useful. We're
testing that prediction in hydrology — a real, falsifiable, theory-connected contribution.

## Ordered program

| Phase | What | Custom code? | Compute |
|---|---|---|---|
| **1** | This 2×2 on component0 + a few subgraphs, single seed | none (stock cudalstm) | ~20 min T4 |
| **2** | *(gated on Phase 1 showing topology helps w/o one-hot)* redesigned message passing in the one-hot-off regime, built correctly | yes — but only if justified | TBD |
| **3** | Publication scale-up: multi-seed, more networks, NHDPlus edges, robustness | — | larger |

Single seed until we see a clear signal; multi-seed only for the publication run.

## Experiment phases (results: see `JOURNAL.md`; consolidated table: `analysis/PAPER_TABLE.md`)

1. **Encoding × topology 2×2** — topology static features add ~0 NSE (with or without one-hot). Static network position is not the lever.
2. **Upstream-signal chain** — feeding *upstream discharge* helps: observed-Q oracle **+0.037 NSE** (lag1; +0.087 lag0), survives a shuffled-Q null control (−0.002), beats upstream-precip 3× (+0.012), lag-robust. Structure helps as a **dynamic** signal.
3. **Realizability** (PASS) — the gain survives with *predicted* (not observed) upstream Q: realizable **+0.022 NSE** (3 seeds, all positive), recovering ~55–72% of the oracle ceiling. Deployable, no ground truth at inference.
4. **Rigor hardening** (all CPU re-analysis) — realizable-vs-null significant (**p=2.3e-12**, bootstrap CI excludes 0); depth gradient is routing not size/feature-magnitude (partial corr survives); per-depth Wilcoxon significant at depth 1–3; robust in NSE + log-NSE; beats a no-ML routing baseline (3-seed, margin +0.019).
5. **Graph robustness** — the routing gain does **not** depend on the heuristic graph's over-connectivity (in-degree mean 4.16/max 15). Pruning to hydrography-realistic in-degree≤2 (266 edges vs 624) retains the R1 signal proxy (100%) **and** the trained-LSTM gain (realizable **+0.021 NSE / +0.034 log-NSE**, p=4e-4; oracle *strengthens* to +0.049). Threat closed at both proxy and model level.

## Files

**Phase 1 — topology features (2×2):**
| File | Purpose |
|---|---|
| `generate_topology_attributes.py` | Computes network-position features (depth, n_upstream, total_upstream_area, in_degree, frac_upstream_area); writes `camels_topology.txt` that NH auto-loads. Run once. |
| `make_configs.py` | Generates the 4 stock-cudalstm 2×2 configs for a network/seed. |
| `run_2x2.py` / `analyze_2x2.py` | Train the 4 conditions / produce the 2×2 table + contrasts + RESULTS.md. |
| `notebooks/colab_topology_2x2.ipynb` | Colab GPU runner for the 2×2. |

**Phase 2 — upstream-signal chain:**
| File | Purpose |
|---|---|
| `build_upstream_discharge_feature.py` | The ORACLE feature: area-weighted mean of upstream basins' lagged **observed** discharge (mm/d). Upper bound on structural signal. |
| `build_upstream_variants.py` | Null/content variants: shuffled-Q (null control) and upstream-precip. |
| `run_upstream_feature.py` | Generic runner: train stock cudalstm + any upstream feature pickle, evaluate. |
| `run_oracle.py` | Convenience runner for the L vs L+upstream_q oracle pair. |
| `preregistration_upstream_signal.md` | Pre-reg + results for the chain (null/precip/lag). |

**Phase 3 — realizability (predicted, not observed):**
| File | Purpose |
|---|---|
| `build_predicted_upstream_q.py` | Stage 1: re-evaluate trained L over full span 1990–2008 → predicted Q per basin (cached in `runs/.../_Lfullspan_eval/`). Stage 2: aggregate upstream **predicted** Q into a feature. Deployable, no target leakage. |
| `notebooks/colab_realizability.ipynb` | Colab runner for the realizability test (needs full CAMELS dataset). |
| `preregistration_realizability.md` | Pre-reg: success ≥ +0.015 (≥40% of the +0.037 ceiling). |
| `notebooks/colab_multiseed.ipynb` | Colab runner for the multi-seed confirmation (seeds 11/13/17). |

**Phase 4 — multi-seed, robustness, compliance (all CPU re-analysis of committed runs):**
| File | Purpose |
|---|---|
| `analyze_multiseed.py` | Cross-seed mean±std + paired Δ + realizable-vs-null (Step A) + depth-stratified gain (Step B). Writes `analysis/MULTISEED.md`. |
| `analyze_confound.py` | Depth-gradient confound check: routing (n_upstream) vs basin size (area), partial control within area terciles. Writes `analysis/CONFOUND.md`. |
| `analyze_compliance.py` | Methodology audit: all-3-metric contrasts (incl. log-NSE) + baseline-strength stratification. Writes `analysis/COMPLIANCE.md`. |
| `analyze_significance.py` | Paired Wilcoxon + bootstrap CI for the realizable gain vs L and vs the null. Writes `analysis/SIGNIFICANCE.md`. |
| `analyze_feature_magnitude_confound.py` | Second confound check: is the depth gradient really feature-magnitude? (It runs opposite.) Writes `analysis/FEATURE_MAGNITUDE_CONFOUND.md`. |
| `analyze_metric_honesty.py` | log-NSE eps-sensitivity sweep + KGE (r/β/γ) decomposition of the seed-13 dip. Writes `analysis/METRIC_HONESTY.md`. |
| `analyze_depth_significance.py` | Per-depth Wilcoxon of the realizable gain (significant at depth 1–3, absent at headwaters). Writes `analysis/DEPTH_SIGNIFICANCE.md`. |
| `analyze_routing_baseline.py` / `analyze_routing_baseline_3seed.py` | No-ML lstsq routing baselines (R1 pure, R2 routing+local) vs the LSTM; single- and 3-seed. Writes `analysis/ROUTING_BASELINE{,_3SEED}.md`. |
| `preregistration_{multiseed,robustness_chain,confound_check,compliance,hardening_chain}.md` | Pre-regs + results for the above. |

**Phase 5 — graph robustness (is the routing gain a heuristic-edge artifact?):**
| File | Purpose |
|---|---|
| `analyze_graph_robustness.py` | Rebuilds `upstream_q` on pruned graphs (in-degree≤k, random dropout) and scores signal strength via the R1 lstsq proxy — ZERO training. Writes `analysis/GRAPH_ROBUSTNESS.md`. |
| `analyze_k2_graph_check.py` | The **model-level** confirmation: analyzes the GPU-trained k=2 (in-degree≤2) oracle + realizable runs vs L. Writes `analysis/K2_GRAPH_CHECK.md`. |
| `build_paper_table.py` | Consolidates all conditions × metrics × significance + routing baselines + depth table into `analysis/PAPER_TABLE.md` (the paper's Results spine). |
| `notebooks/colab_oracle_completion_and_k2.ipynb` | Idempotent Colab runner: oracle seed-11 restore + k=2 graph-check re-trains. |
| `preregistration_{graph_robustness_chain,baseline_completion_and_k2,routing_baseline_chain}.md` | Pre-regs for the above. |
| `component0_edges_k2.csv` (in `topology_analysis/.../outputs/`) | The k=2 nearest-parent pruned edge set (266 edges). |

**Generated:** `configs/` (per-run YAMLs), `features/` (upstream-Q pickles, gitignored), `analysis/` (all `*.md` + CSV outputs). Run outputs land in `runs/topology_ablation/component0/` (see its `NOTES.md`).

**Analysis-file index (`analysis/`):** `PAPER_TABLE.md` (consolidated Results — start here) · `RESULTS.md` (2×2) · `MULTISEED.md` · `SIGNIFICANCE.md` · `CONFOUND.md` + `FEATURE_MAGNITUDE_CONFOUND.md` (routing vs size/magnitude) · `DEPTH_SIGNIFICANCE.md` · `COMPLIANCE.md` + `METRIC_HONESTY.md` (3-metric rigor) · `ROUTING_BASELINE.md` + `ROUTING_BASELINE_3SEED.md` (no-ML baseline) · `GRAPH_ROBUSTNESS.md` + `K2_GRAPH_CHECK.md` (edge-density robustness).

## How to run

**Colab (recommended, GPU):** open `notebooks/colab_topology_2x2.ipynb` → T4 → Run all.

**Local (CPU, fine for small subgraphs):**
```bash
python experiments/topology_ablation/generate_topology_attributes.py   # once
python experiments/topology_ablation/run_2x2.py --networks sg_northeast --seed 11 --device cpu
python experiments/topology_ablation/analyze_2x2.py
```

## Honest framing note

We commit to the *framework and the rigor*, not to the *sign of the result*. If topology
helps without the one-hot → a positive, theory-grounded finding. If it doesn't even then →
the features are genuinely weak, also a clean finding. The 2×2 is informative either way.
Legacy work (`../5cond_factorial/`, `../local_subgraphs/`) is preserved as the
"confounded-measurement" prior that motivated this restart.
