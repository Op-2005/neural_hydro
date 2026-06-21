# Topology-Ablation 2x2 — Does network position help, and when?

Controlled experiment on **stock NH cudalstm** (identical trainer; only `use_basin_id_encoding` and `static_attributes` differ). Tests whether the basin one-hot encoding makes topology features redundant.

## Per-network median NSE

| Network | L | L+T | L_noID | L_noID+T |
|---|---|---|---|---|
| component0 | +0.653 | +0.654 | +0.633 | +0.625 |

## Key contrasts (paired per-basin median ΔNSE)

| Network | topo benefit WITH one-hot | topo benefit WITHOUT one-hot | interaction | encoding cost |
|---|---|---|---|---|
| component0 | -0.001 | +0.003 | -0.004 | +0.012 |

**Pre-registered prediction:** `topo benefit WITHOUT one-hot` > 0 while `topo benefit WITH one-hot` ≈ 0 → the basin one-hot subsumes topology features. If confirmed, the paper's framing is *'network structure helps streamflow LSTMs only when the model cannot memorize per-basin identity'* — a controlled, theory-grounded contribution (cf. Kipf-Welling: graph structure helps most in the can't-memorize regime).
