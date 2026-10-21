# topology_analysis — network construction

**This folder is live infrastructure, not a research phase.** It builds the basin sets and edge
lists that the ablation conditions consume, and `experiments/topology_ablation/` reads its
`outputs/` directly. **Relocating this folder breaks those paths**, so check what still points
here before moving anything:

```bash
grep -rn "topology_analysis" experiments/ --include="*.py"
```

An earlier Phase 0 (a 10-basin MPNN and signal-decay exploration, Jan–Feb 2026) was retired in
October 2026. Those 10 basins are all headwaters, so the edge inference correctly returned an
empty set and there was no graph to model.

```
phase1_network_discovery/
├── discover_network.py      infer_edges() over all CAMELS-US basins
├── extract_component_0.py   isolate the eastern-US study network used by the study
├── plot_full_network.py     national map, component stats, edge diagnostics
├── plot_study_network.py    geographic + DAG layouts
├── DEBRIEF.md               the April 2026 discovery write-up and its caveats
└── outputs/                 basin lists, edge lists, summary stats, figures
```

## The edge rule

For every ordered pair (A, B), draw A → B ("A is upstream of B") when all three hold:

| Condition | CAMELS field | Intuition |
|---|---|---|
| B's drainage area exceeds A's by a set ratio | `area_gages2` | a downstream gauge drains everything above it |
| B's mean elevation is below A's | `elev_mean` | water flows downhill |
| gauges within a distance cap | haversine on `gauge_lat/lon` | excludes merely-bigger-and-lower basins |

The thresholds are set in `discover_network.py`; read them there rather than from this file.
Summary statistics for the graph it produces — node and edge counts, component sizes, degree
and edge-length distributions — are written to `outputs/` by the plotting scripts.

**The rule is a heuristic, never validated against surveyed hydrography.** No flow lines, no
catchment shapefiles, no check that A actually sits inside B's catchment. It infers connectivity
from three scalars because that is all CAMELS ships per basin. This is stated as a limitation
wherever the graph is used.

Properties worth knowing before quoting the graph:

- **Over-connected.** A basin links to *every* qualifying basin, so its upstream set mixes direct
  parents with grandparents, and mean in-degree runs well above the 2–3 of real confluences.
  Pruning each basin to its nearest parents leaves the measured gain unchanged
  (`analysis/K2_GRAPH_CHECK.md`).
- **Neither transitively closed nor reduced.** The distance cap severs some long ancestor links,
  so an upstream set holds many indirect ancestors but not all of them.
- **Edges are long**, and the **distance cap** sets that length rather than the area ratio —
  sweeping the ratio barely moves it (`analysis/EDGE_RULE_SENSITIVITY.md`). This matters because
  it is why selecting the nearest gauges outperforms the network: the rule picks neighbours
  roughly twice as far away.
- **Acyclic** by construction, since every edge strictly descends.

## outputs/

Generated; treat the directory listing as the inventory rather than trusting a table here.
By naming convention:

| Pattern | What it is |
|---|---|
| `component0_basins.txt` | the study network's basin list — every condition's basin set |
| `component0_edges.csv` | the network selector's edge list |
| `component0_edges_k2.csv` | pruned to the nearest parents |
| `component0_edges_distctrl.csv` | **the distance-matched control** that falsified the topology claim |
| `component0_edges_dist*km.csv` | the distance sweep, one file per target separation |
| `component0_edges_knn*.csv` | nearest-gauge selectors, including minimum-separation floors |
| `component0_edges_randomclean.csv` | the distance-ignoring random rewire |
| `component0_depth*.csv` | graph depth — used by analyses whose interpretation was later retired |
| `full_network_edges.csv` | the national graph before component extraction |
| `study_network_*` | the abandoned 23-basin Texas pilot network, kept for provenance |

**The study used Component 0**, not the Texas network; no reported result rests on
`study_network_*`.

## Regenerating

```bash
python topology_analysis/phase1_network_discovery/discover_network.py
python topology_analysis/phase1_network_discovery/extract_component_0.py
```

The control, sweep and kNN variants are built by
`experiments/topology_ablation/build_upstream_variants.py`, which writes back into `outputs/`.
Regeneration is deterministic, but there is no reason to run it: every consumed file is on disk,
and every reported number traces to a run that used these exact files.
