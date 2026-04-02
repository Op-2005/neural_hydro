# Phase 0 — Topology Scaffold

Early exploration of graph-based information flow on 10 HUC-01 headwater basins
(Maine / New Hampshire). Self-contained — no NeuralHydrology imports.

## Key finding

The 10 basins are separate headwater catchments with no upstream-downstream
relationships. Heuristic edge inference correctly returns **E = empty**.

Signal decay experiments on synthetic graphs show:
- Perturbation decays exponentially with hop distance
- Tree branching creates over-squashing at branch points
- 2-layer MPNN extends information reach by ~1 hop vs 1-layer

## Files

```
basin_graph.py          BasinGraph + NodeStateMatrix data structures
infer_topology.py       Heuristic edge inference (50 km, area ratio 1.1x)
graph_analysis.py       Component analysis, betweenness, bottleneck detection
hop_distance.py         BFS shortest path computation
mpnn.py                 Minimal MPNN (message / aggregate / update)
signal_decay.py         Perturbation experiment class
run_signal_decay.py     Experiment runner + plotting
synthetic_graphs.py     Chain and tree graph generators
outputs/                Signal decay results (metrics.csv, plot.png, config, summary)
```
