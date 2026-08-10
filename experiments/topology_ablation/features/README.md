# features/ — upstream-signal input pickles (gitignored, regenerable)

The additional-feature files NH concatenates onto each basin's dynamic inputs. One `upstream_q`
column per basin, indexed by a DatetimeIndex **named `date`** (NH requirement; an unnamed index is
the recurring `KeyError: 'date'` bug). All are **gitignored** (large) and rebuilt on Colab from
committed inputs (edge lists in `topology_analysis/phase1_network_discovery/outputs/`, CAMELS on
Drive).

| Pickle | Content | Built by |
|---|---|---|
| `upstream_q_component0_lag{0,1,2}.p` | observed upstream Q (oracle), area-weighted, lag 0/1/2 | `build_upstream_discharge_feature.py` |
| `upstream_q_pred_component0_lag1.p` | predicted upstream Q (realizable, two-stage) | `build_predicted_upstream_q.py` |
| `upstream_q_pred_component0_seed{N}_lag{0,1}.p` | per-seed predicted upstream Q | same |
| `upstream_q_shuffled_component0_lag1.p` | time-shuffled null control | `build_upstream_variants.py` |
| `upstream_precip_component0_lag1.p` | upstream precipitation variant | `build_upstream_variants.py` |
| `upstream_q_{obs,pred}_component0_k2_lag1.p` | oracle / realizable on the in-degree≤2 pruned graph | `build_upstream_variants.py` |
| `upstream_q_{reversed,random}_component0_lag1.p` | directionality controls (reversed / random rewire) | `build_directionality_variants.py` |
| `upstream_q_distctrl_component0_lag1.p` | **distance-preserving control** (nearest-distance non-parents) | `build_distance_control.py` |

Regenerate, e.g.: `python experiments/topology_ablation/build_distance_control.py --network component0 --lag-days 1`
