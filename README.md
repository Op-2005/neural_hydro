# Neural Hydrology — Graph-LSTM Experiments (Baker Lab @ UCLA)

Graph-based extensions to the [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
framework (Kratzert et al., JKU Linz). This fork tests whether river-network
topology and inter-basin message passing improve streamflow prediction over a
strong multi-basin LSTM baseline.

## What's here

- `neuralhydrology/` — upstream NH framework, unmodified.
- `datasets/camels_us/` — CAMELS-US data (674 basins, 3 forcing products).
- `topology_analysis/` — river-network inference from CAMELS gauges (1298 edges
  across 584 basins in 46 connected components).
- `experiments/` — training scripts, configs, basin lists, analysis tools. See
  `experiments/README.md` for per-file descriptions.
- `runs/` — every experiment run, numbered chronologically. See
  `runs/README.md` for the index.
- `idea1.md` — master file for the current active research direction.
- `idea2/` — alternative direction (temporal-lag spectral framing), set aside.
- `INSIGHTS.md` — research findings from the 23-basin pilot.
- `CURRENT_STATE.md` — chronological log of experiments and evolving thinking.

## Quick start

```bash
conda activate nh

# Train the 23-basin strong baseline
python neuralhydrology/nh_run.py train --config-file experiments/configs/lstm_study_network_strong.yaml

# Train a graph-LSTM variant (edit flags at top of script first)
python experiments/training/train_graph_lstm.py

# Compare results
python experiments/analysis/compare_results.py \
    --baseline runs/05_lstm_23basin_strong_baseline/test/model_epoch030/test_metrics.csv \
    --baseline-label "Strong LSTM" \
    --graph runs/06_graph_edge_warm_full/test_metrics.csv:Graph+Edge
```

## Credits

Built on NeuralHydrology:

```bibtex
@article{kratzert2022joss,
  title = {NeuralHydrology --- A Python library for Deep Learning research in hydrology},
  author = {Frederik Kratzert and Martin Gauch and Grey Nearing and Daniel Klotz},
  journal = {Journal of Open Source Software},
  year = {2022},
  volume = {7}, number = {71}, pages = {4050},
  doi = {10.21105/joss.04050},
}
```
