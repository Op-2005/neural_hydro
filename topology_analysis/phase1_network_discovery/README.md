# Phase 1 — River Network Discovery

Heuristic topology inference on all 671 CAMELS-US basins. Identifies connected
subnetworks and selects a 23-basin study network for the Graph-LSTM experiment.

## Run

```bash
python topology_analysis/phase1_network_discovery/discover_network.py
python topology_analysis/phase1_network_discovery/plot_study_network.py
python topology_analysis/phase1_network_discovery/plot_full_network.py
```

## Results

```
Basins loaded:         671
Directed edges found:  1298
Connected components:  46

Selected study network (Component 3):
  Location:    HUC-12, south-central Texas
  Basins:      23  (7 headwater, 7 interior, 9 outlet)
  Edges:       34
  Max depth:   3
  Data:        All basins have maurer forcings + streamflow
```

## Outputs

```
outputs/
├── full_network_edges.csv          1298 edges across all CAMELS basins
├── study_network_edges.csv         34 edges in the selected network
├── study_network_basins.txt        23 basin IDs (one per line)
├── study_network_summary.txt       Per-basin inventory (area, elev, depth, role)
├── national_network_map.png        All basins, top 5 components colored
├── component_distribution.png      Component sizes (bar + histogram)
├── edge_diagnostics.png            Distance vs area ratio for all edges
├── study_network_map.png           Geographic layout, colored by role
├── study_network_dag.png           Topological DAG layout
└── study_network_depth.png         Geographic layout, colored by depth (0-3)
```

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_distance_km` | 150 | Wider than Phase 0 (50 km) for national scale |
| `area_ratio_threshold` | 1.5 | Conservative — child must be 50% larger |
| `elevation_must_decrease` | True | Water flows downhill |

## Caveats

See [DEBRIEF.md](DEBRIEF.md) for full analysis. Key points:

- Edges are heuristic, not ground truth (NHDPlus flowlines would be better)
- Some false positives likely exist (34 edges for 23 nodes is denser than a tree)
- Maurer forcings end at 2008; effective test period is 2005-2008
