# Basin Lists

USGS gauge-ID files referenced by NH configs in `../configs/` and by training
scripts in `../training/`. One gauge ID per line, 8 characters.

| File | Basins | Used by |
|---|---|---|
| `study_network_basins.txt` | 23 | `lstm_study_network.yaml`, `lstm_study_network_strong.yaml`, `train_graph_lstm.py`, `train_graph_ungauged.py` (as the all-basins graph reference) |
| `ungauged_train_basins.txt` | 20 | `lstm_ungauged_train.yaml`, `train_graph_ungauged.py` |
| `ungauged_test_basins.txt` | 3 (08158700, 08164300, 08189500) | `lstm_ungauged_train.yaml` (test split), `train_graph_ungauged.py` (held-out mask) |

The **Component 0** basin list (183 basins, scaled experiment) is **not**
duplicated here — it lives alongside the topology artifacts that defined it:
`topology_analysis/phase1_network_discovery/outputs/component0_basins.txt`.

## How the study network was selected

The 23 basins in `study_network_basins.txt` are Component 3 from the full
CAMELS-US topology inference (see
`topology_analysis/phase1_network_discovery/discover_network.py`). Selection
criteria: ≥ 15 nodes, graph depth ≥ 3, single HUC region for geographic
coherence. Full inventory (area, elevation, graph depth, role) is in
`topology_analysis/phase1_network_discovery/outputs/study_network_summary.txt`.

## How the ungauged split was chosen

08158700, 08164300, 08189500 were picked to cover three structurally different
PUB situations:
- **08158700** — leaf basin with a single training-set parent (clean case)
- **08164300** — middle node with a held-out parent *and* serving as parent to
  another held-out basin (worst case: chain contamination)
- **08189500** — leaf basin with one held-out parent + one training parent
  (partial contamination)

The contrast between these three cases is what `INSIGHTS.md` Finding #6 rests on.
