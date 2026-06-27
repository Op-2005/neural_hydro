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
| `L_upQpred_component0_seed11` | + upstream PREDICTED Q (realizability) | (running) | TBD |

Oracle (observed upstream Q) is an UPPER BOUND. Realizability run tests the deployable
version (predicted Q). Pre-reg: `../../../experiments/topology_ablation/preregistration_*.md`.

## `_Lfullspan_eval/` (intermediate artifact)
The trained L baseline re-evaluated over the FULL span 1990-2008 (not just test) to
produce predicted Q for every basin. Stage 1 of the realizability test; its
`test/model_epoch030/test_results.p` feeds `build_predicted_upstream_q.py`. Generated on
Colab (needs full CAMELS dataset); kept here as the cached input so Stage 2 reruns locally.
Gitignored (large pickle), regenerable via the realizability Colab notebook.
