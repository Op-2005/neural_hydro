# topology_ablation / component0 — run index

All runs on stock NH `cudalstm`, component0 (183 basins), seed 11, 30 epochs,
one-hot ON, train 1990-1999 / test 2005-2008. Results = `test/model_epoch030/test_metrics.csv`.
Heavy checkpoints/train_data are gitignored (regenerable); only metrics+config tracked.

## The encoding × topology 2×2 (Phase 1 — topology features null)
| Run | What | median NSE |
|---|---|---|
| `L_component0_seed11` | baseline (one-hot, no topo) | 0.653 |
| `L_T_component0_seed11` | + 5 topology static features | 0.654 |
| `L_noID_component0_seed11` | one-hot OFF | 0.633 |
| `L_noID_T_component0_seed11` | one-hot OFF + topo | 0.625 |

Topology features add ~0 NSE either way → static network position is not the lever.

## Upstream-signal chain (the positive result + stress tests)
| Run | What | median NSE | Δ vs L |
|---|---|---|---|
| `L_upQ_component0_seed11` | + upstream OBSERVED discharge (oracle, lag1) | 0.703 | **+0.037** |
| `L_upQshuf_component0_seed11` | shuffled-Q null control | 0.658 | −0.002 (gate PASS) |
| `L_upPrecip_component0_seed11` | + upstream precipitation | 0.674 | +0.012 |
| `L_upQ_lag0_component0_seed11` | upstream Q, lag 0 | 0.749 | +0.087 |
| `L_upQ_lag2_component0_seed11` | upstream Q, lag 2 | 0.699 | +0.036 |
| `L_upQpred_component0_seed11` | + upstream PREDICTED Q (realizability) | 0.683 | **+0.027** |

Oracle (observed upstream Q) is an UPPER BOUND. Realizability PASSED — predicted Q is
deployable. Pre-reg: `../../../experiments/topology_ablation/preregistration_*.md`.

## Multi-seed (seeds 13/17 added; headline invariant)
`L`, `L_upQ`, `L_upQpred`, `L_upQshuf` also present at `_seed13` / `_seed17`. Cross-seed
realizable Δ = **+0.022 ± 0.006**, all 3 positive (`analysis/MULTISEED.md`, `SIGNIFICANCE.md`).

## k=2 pruned-graph check (graph-robustness, seed 11)
| Run | What | median NSE | Δ vs L (connected) |
|---|---|---|---|
| `L_upQ_k2_component0_seed11` | oracle on in-degree≤2 pruned graph (266 edges) | 0.710 | **+0.049** (p=2e-12) |
| `L_upQpred_k2_component0_seed11` | realizable on k=2 pruned graph | 0.675 | **+0.021** (p=4e-4) |

Pruning the over-connected heuristic graph to hydrography-realistic connectivity does NOT
kill the gain (realizable holds; oracle strengthens). Over-connectivity threat closed at the
model level. See `analysis/K2_GRAPH_CHECK.md`, `GRAPH_ROBUSTNESS.md`.

## `_Lfullspan_eval/` (intermediate artifact)
The trained L baseline re-evaluated over the FULL span 1990-2008 (not just test) to
produce predicted Q for every basin. Stage 1 of the realizability test; its
`test/model_epoch030/test_results.p` feeds `build_predicted_upstream_q.py`. Generated on
Colab (needs full CAMELS dataset); kept here as the cached input so Stage 2 reruns locally.
Gitignored (large pickle), regenerable via the realizability Colab notebook.

## Distance-preserving control (3 seeds) — the topology-specificity FALSIFICATION
| Run | What | median Δ vs L (connected) |
|---|---|---|
| `L_upQdistctrl_component0_seed{11,13,17}` | oracle Q aggregated over a distance-matched, NON-upstream rewire | **+0.049** (per-seed +0.046/+0.060/+0.041) |

Each true edge is replaced by the nearest-distance non-parent basin (in-degree preserved,
0% edge overlap, 101 km vs true 92 km, vs 511 km for the random rewire). Pre-registered
prediction was that destroying the true topology while holding proximity fixed would drop the
gain to the random level. It did NOT: distance-control +0.049 ≈ forward +0.046 ≫ random +0.012.
**This falsifies topology-specificity — the gain is proximity/regional, not the true upstream
topology.** Builder `../../../experiments/topology_ablation/build_distance_control.py`; pre-reg
`preregistration_distance_control.md`; notebook `notebooks/colab_distance_control.ipynb`;
consequences in `analysis/MECHANISM_MULTISEED.md` and JOURNAL 2026-08-10. Relocated here from
the repo root (Colab/Drive download landing spot) on 2026-08-10.
