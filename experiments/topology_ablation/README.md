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

## Experiment phases (results: see `JOURNAL.md`)

1. **Encoding × topology 2×2** — topology static features add ~0 NSE (with or without one-hot). Static network position is not the lever.
2. **Upstream-signal chain** — feeding *upstream discharge* helps: observed-Q oracle **+0.037 NSE** (lag1; +0.087 lag0), survives a shuffled-Q null control (−0.002), beats upstream-precip 3× (+0.012), lag-robust. Structure helps as a **dynamic** signal.
3. **Realizability** (in progress) — does the gain survive with *predicted* (not observed) upstream Q? Decides deployable-method vs upper-bound.

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

**Generated:** `configs/` (per-run YAMLs), `features/` (upstream-Q pickles, gitignored), `analysis/` (2×2 outputs). Run outputs land in `runs/topology_ablation/component0/` (see its `NOTES.md`).

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
