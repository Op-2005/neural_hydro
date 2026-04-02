# Phase 1 Debrief: River Network Discovery

## What was done

Ran heuristic topology inference on all 671 CAMELS-US basins to discover upstream-downstream
relationships, identified connected river subnetworks, and selected a study network for the
directed Graph-LSTM experiment.

## Scripts created

| File | Purpose |
|------|---------|
| `topology_analysis/discover_network.py` | Full pipeline: load all basins, infer edges, find components, select study network |
| `topology_analysis/plot_study_network.py` | Geographic map + DAG layout visualizations |

## Outputs produced

| File | Contents |
|------|----------|
| `topology_analysis/outputs/full_network_edges.csv` | 1298 directed edges across 584 CAMELS basins |
| `topology_analysis/outputs/study_network_edges.csv` | 34 edges within the selected 23-basin network |
| `topology_analysis/outputs/study_network_basins.txt` | 23 basin IDs, one per line |
| `topology_analysis/outputs/study_network_summary.txt` | Full inventory with areas, elevations, depths, roles |
| `topology_analysis/outputs/study_network_map.png` | Geographic map of the study network |
| `topology_analysis/outputs/study_network_dag.png` | Topological DAG layout |

## Inference parameters

```
max_distance_km = 150
area_ratio_threshold = 1.5      # child must be >= 50% larger than parent
elevation_must_decrease = True   # downstream is lower
```

These are wider than the original 50km/1.1x parameters (which found zero edges for the 10
HUC-01 headwater basins). The 150km distance and 1.5x area ratio are appropriate for the full
national dataset where basins can be far apart but still connected through the same river system.

## Full CAMELS-US network overview

```
Total basins loaded:              671
Total directed edges found:       1298
Nodes involved in at least 1 edge: 584
Connected components (>= 2 nodes): 46
```

Top 5 components by size:

| Component | Nodes | Edges | Depth | HUC regions |
|-----------|-------|-------|-------|-------------|
| 0 | 183 | 624 | 4 | 01, 02, 03, 04, 05, 06 |
| 1 | 72 | 239 | 3 | 17, 18 |
| 2 | 33 | 43 | 2 | 07, 08, 10, 11 |
| 3 | 23 | 34 | 3 | 12 |
| 4 | 19 | 20 | 3 | 03, 08 |

## Selected study network

**Component 3** — 23 basins in HUC-12 (south-central Texas)

Selected because it meets all criteria from the plan:
- **Size**: 23 basins (within the preferred 20-40 range)
- **Depth**: max path length = 3 (meets minimum of 3)
- **Geographic coherence**: single HUC region (HUC-12), avoiding cross-climate confounds
- **Data completeness**: all 23 basins have maurer forcings (1980-2008) and streamflow (1980-2014)

Why not the larger components:
- Component 0 (183 nodes) is too large and spans 6 HUC regions
- Component 1 (72 nodes) spans 2 HUC regions in the Pacific Northwest
- Component 2 (33 nodes) only has depth 2

### Network structure

```
Basins:     23
Edges:      34
Roots:      7  (headwater basins with no upstream CAMELS gauge)
Interior:   7  (basins with both upstream and downstream CAMELS gauges)
Outlets:    9  (basins with no downstream CAMELS gauge)
Max depth:  3
```

Area range: 31.8 - 2124.0 km²
Elevation range: 67.3 - 669.2 m
Geographic extent: (28.29°N, 100.24°W) to (31.28°N, 96.69°W)

### Basin inventory

| Basin | Area (km²) | Elev (m) | Depth | Role |
|-------|-----------|----------|-------|------|
| 08103900 | 86.0 | 383.0 | 0 | headwater |
| 08150800 | 557.5 | 563.6 | 0 | headwater |
| 08158810 | 31.8 | 308.1 | 0 | headwater |
| 08165300 | 436.0 | 669.2 | 0 | headwater |
| 08176900 | 925.4 | 88.5 | 0 | headwater |
| 08196000 | 327.0 | 536.8 | 0 | headwater |
| 08200000 | 249.1 | 464.3 | 0 | headwater |
| 08104900 | 342.6 | 317.4 | 1 | interior |
| 08109700 | 610.1 | 143.1 | 1 | interior |
| 08155200 | 232.7 | 337.3 | 1 | interior |
| 08158700 | 320.3 | 368.0 | 1 | interior |
| 08195000 | 1028.3 | 578.2 | 1 | interior |
| 08202700 | 434.8 | 399.0 | 1 | interior |
| 08101000 | 1177.1 | 388.2 | 1 | outlet |
| 08171300 | 1067.5 | 379.2 | 1 | outlet |
| 08178880 | 850.6 | 542.6 | 1 | outlet |
| 08190000 | 1961.4 | 567.8 | 1 | outlet |
| 08190500 | 1799.1 | 590.4 | 1 | outlet |
| 08198500 | 624.4 | 439.4 | 1 | outlet |
| 08164300 | 861.5 | 102.7 | 2 | interior |
| 08175000 | 1421.7 | 113.3 | 2 | outlet |
| 08164000 | 2124.0 | 73.1 | 3 | outlet |
| 08189500 | 1808.3 | 67.3 | 3 | outlet |

## Important caveats

### Heuristic edges are not ground truth

The edges are inferred from three heuristics (area, elevation, proximity), not from actual
river network data (e.g., NHDPlus flowlines). This means:

- **False positives exist**: Some edges connect basins that are geographically close and have
  the right area/elevation gradient but are not actually in the same river system. For example,
  headwater basin 08103900 (86 km², central Texas) is connected to 5 downstream basins at
  distances up to 125 km — some of these are likely spurious.

- **False negatives exist**: Real upstream-downstream pairs may be missed if they don't satisfy
  all three heuristics simultaneously (e.g., distance > 150 km, or unusual elevation profiles).

- **The network is a DAG, not a tree**: Several basins have multiple "parents" and multiple
  "children," which wouldn't happen in a true river network (where each point has exactly one
  downstream). The 34 edges for 23 nodes (ratio 1.48) is higher than a tree (which would have
  22 edges for 23 nodes).

For the Graph-LSTM experiment, this is acceptable as a first pass — the model should learn to
ignore spurious edges (the zero-initialized residual connection means upstream messages start
with no influence). But for a publication-quality result, consider replacing these heuristics
with NHDPlus-derived flowlines.

### Maurer forcings end at 2008

The maurer forcing data covers 1980-2008. The existing experiment config uses test_end_date=2010.
NeuralHydrology handles this gracefully — forcing values beyond 2008 are NaN and evaluation
skips those dates. No config change is needed, but effective test period for maurer is 2005-2008
(4 years instead of 6).

## What's next (Phase 2 and 3)

1. **Phase 2**: Create `experiments/lstm_study_network.yaml` pointing to the 23-basin list.
   Train and evaluate the baseline LSTM to establish per-basin NSE for this network.

2. **Phase 3**: Build `DirectedGraphLSTM` that augments each basin's LSTM with lagged upstream
   hidden states. The scientific prediction: downstream basins (depth >= 2) should show improved
   NSE compared to the independent baseline, while headwater basins (depth = 0) should be
   unchanged.

## How to reproduce

```bash
# Run network discovery
python topology_analysis/discover_network.py

# Generate visualizations
python topology_analysis/plot_study_network.py
```
