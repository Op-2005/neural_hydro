# Topology Analysis

Research on topology-aware message passing for watershed prediction.

```
topology_analysis/
│
├── phase0_scaffold/                  Early exploration (Jan-Feb 2026)
│   ├── basin_graph.py                  Graph + node state data structures
│   ├── infer_topology.py               Heuristic edge inference (10 basins, E=empty)
│   ├── graph_analysis.py               Structural analysis (components, betweenness)
│   ├── hop_distance.py                 BFS shortest paths
│   ├── mpnn.py                         Message Passing Neural Network
│   ├── signal_decay.py                 Perturbation / over-squashing experiments
│   ├── run_signal_decay.py             Signal decay runner
│   ├── synthetic_graphs.py             Chain + tree test graphs
│   └── outputs/                        Signal decay results (CSV, plot, config)
│
├── phase1_network_discovery/         Full CAMELS-US topology (April 2026)
│   ├── discover_network.py             Infer 1298 edges on 671 basins, select study network
│   ├── plot_study_network.py           Study network geographic + DAG plots
│   ├── plot_full_network.py            National map, component stats, edge diagnostics
│   ├── DEBRIEF.md                      Detailed analysis, caveats, next steps
│   └── outputs/
│       ├── full_network_edges.csv        1298 edges across all CAMELS basins
│       ├── study_network_edges.csv       34 edges in the 23-basin study network
│       ├── study_network_basins.txt      Basin IDs (one per line)
│       ├── study_network_summary.txt     Inventory: area, elevation, depth, role
│       ├── national_network_map.png      671 basins, top 5 components colored
│       ├── component_distribution.png    Component size bar chart + histogram
│       ├── edge_diagnostics.png          Distance vs area ratio scatter
│       ├── study_network_map.png         Geographic map (blue/grey/red by role)
│       ├── study_network_dag.png         Topological DAG layout
│       └── study_network_depth.png       Depth colorbar (0=headwater, 3=outlet)
│
└── README.md                         This file
```

## Phase 0 — Scaffold

Explored graph-based information flow on 10 HUC-01 headwater basins. These basins
have no upstream-downstream relationships, so the heuristic inference correctly returns
an empty edge set. Built a minimal MPNN and demonstrated signal decay on synthetic
chain/tree graphs. All code is self-contained (no NeuralHydrology imports).

## Phase 1 — Network Discovery

Ran topology inference on all 671 CAMELS-US basins with wider parameters (150 km,
area ratio 1.5x). Found 1298 directed edges forming 46 connected components.
Selected a **23-basin study network in HUC-12 (south-central Texas)** with max depth 3,
a single HUC region, and complete data coverage. This network is the target for the
Graph-LSTM experiment.

## Next

- **Phase 2** — Train baseline LSTM on the 23-basin study network
- **Phase 3** — Build and evaluate Directed Graph-LSTM with lagged upstream hidden states
