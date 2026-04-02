# Neural Hydrology — Baker Lab @ UCLA

Topology-aware deep learning for watershed prediction, adapted from the
[NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) framework
(Kratzert et al., JKU Linz).

## Current State (April 2026)

**Baseline established, study network selected, Graph-LSTM experiment ready to begin.**

- Baseline LSTM trained on 10 HUC-01 basins (Maine/NH) — median NSE **0.73**
- Topology inference run on all 671 CAMELS-US basins — **1298 directed edges** found
- **23-basin study network** selected in HUC-12 (Texas) with max depth 3
- Next: train baseline on study network, then build Directed Graph-LSTM

## Repository Structure

```
neural_hydrology/
│
├── neuralhydrology/                  Upstream NeuralHydrology framework (unmodified)
│   ├── nh_run.py                       Entry point: train / evaluate / finetune
│   ├── datasetzoo/                     Dataset loaders (CAMELS-US, GB, BR, etc.)
│   ├── modelzoo/                       Models (CudaLSTM, EALSTM, Transformer, etc.)
│   ├── training/                       Training loop, loss, early stopping
│   ├── evaluation/                     Tester, metrics (NSE, RMSE, KGE)
│   └── utils/                          Config parser, logging
│
├── experiments/                      Experiment configs and basin lists
│   ├── lstm_camels_us.yaml             10-basin LSTM config (maurer forcings)
│   └── 1_basin.txt                     10 HUC-01 basin IDs
│
├── datasets/camels_us/               CAMELS-US data (671 basins)
│   ├── usgs_streamflow/{HUC}/          Daily discharge by HUC-02 region
│   ├── basin_mean_forcing/             maurer / daymet / nldas (HUC subfolders)
│   └── camels_attributes_v2.0/         Static attributes (topo, soil, climate)
│
├── topology_analysis/                Research: graph-based extensions
│   ├── phase0_scaffold/                Early exploration (10 basins, MPNN, signal decay)
│   └── phase1_network_discovery/       Full CAMELS-US topology + study network selection
│
├── runs/                             Saved experiment runs
│   ├── lstm_camels_us_1901_235614/     Jan 19 — 10 basins, batch_size=32
│   ├── lstm_camels_us_2001_000652/     Jan 20 — 10 basins, batch_size=128 (best)
│   └── _archive/                       Early failed/debug runs
│
├── next_steps.md                     Three-phase plan (network discovery → Graph-LSTM)
├── examples/                         Upstream tutorial notebooks
├── test/                             Upstream test suite
└── docs/                             Upstream documentation
```

## Quick Start

```bash
# Train baseline LSTM
python neuralhydrology/nh_run.py train --config-file experiments/lstm_camels_us.yaml

# Evaluate a run
python neuralhydrology/nh_run.py evaluate --run-dir runs/lstm_camels_us_2001_000652 --epoch 5

# Run network discovery (Phase 1)
python topology_analysis/phase1_network_discovery/discover_network.py
python topology_analysis/phase1_network_discovery/plot_study_network.py
python topology_analysis/phase1_network_discovery/plot_full_network.py
```

## Baseline Results (10 HUC-01 Basins)

| Basin | Location | Test NSE |
|-------|----------|----------|
| 01013500 | Fish River, Fort Kent ME | 0.844 |
| 01030500 | Mattawamkeag River ME | 0.794 |
| 01047000 | Carrabassett River ME | 0.777 |
| 01022500 | Narraguagus River ME | 0.766 |
| 01057000 | Little Androscoggin ME | 0.765 |
| 01052500 | Diamond River NH | 0.748 |
| 01031500 | Piscataquis River ME | 0.746 |
| 01055000 | Swift River ME | 0.671 |
| 01073000 | Oyster River NH | 0.662 |
| 01054200 | Wild River ME | 0.613 |

*Run `lstm_camels_us_2001_000652` — 5 epochs, batch_size=128, maurer forcings*

## Network Discovery Results (Phase 1)

Heuristic topology inference on all 671 CAMELS-US basins (150 km, area ratio 1.5x):

- **1298 directed edges** across 584 basins, forming **46 connected components**
- Selected **23-basin study network** in HUC-12 (south-central Texas)
  - 7 headwaters, 7 interior basins, 9 outlets
  - Max depth: 3 hops from headwater to outlet
  - Single HUC region (avoids cross-climate confounds)
  - All basins have complete maurer forcings and streamflow data

See `topology_analysis/phase1_network_discovery/` for full outputs and visualizations.

## Research Direction

The scientific hypothesis: a downstream basin's discharge depends on upstream conditions
with a time lag. A per-basin LSTM cannot learn this. A **Directed Graph-LSTM** that
receives lagged hidden states from upstream neighbors should improve predictions at
downstream basins (depth >= 2) while leaving headwater predictions unchanged.

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | Done | MPNN scaffold, signal decay on synthetic graphs |
| 1 | Done | Full CAMELS topology inference, study network selection |
| 2 | Next | Baseline LSTM on 23-basin study network |
| 3 | Planned | Directed Graph-LSTM with upstream message passing |

## Data Notes

CAMELS-US data is organized by **HUC-02 region** (not by gauge ID prefix). Basin
`01435000` is in HUC-02 (Catskills, NY), not HUC-01. The loading code uses recursive
glob (`**/`) so folder structure doesn't matter for file discovery. Maurer forcings
cover 1980-2008; effective test period is 2005-2008.

## Upstream Framework

Built on [NeuralHydrology](https://neuralhydrology.readthedocs.io) by Kratzert, Gauch,
Nearing, and Klotz (JKU Linz).

```bibtex
@article{kratzert2022joss,
  title = {NeuralHydrology --- A Python library for Deep Learning research in hydrology},
  author = {Frederik Kratzert and Martin Gauch and Grey Nearing and Daniel Klotz},
  journal = {Journal of Open Source Software},
  year = {2022},
  volume = {7},
  number = {71},
  pages = {4050},
  doi = {10.21105/joss.04050},
}
```
