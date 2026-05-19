# Neural Hydrology — Graph-LSTM Experiments (Baker Lab @ UCLA)

Graph-based extensions to the [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) framework (Kratzert et al., JKU Linz). This fork investigates whether river-network topology and inter-basin message passing improve streamflow prediction over a strong multi-basin LSTM baseline on CAMELS-US.

## What this is

A research codebase, not a finished product. The work is ongoing; current results, open questions, and the next experiments are tracked elsewhere (see "Where to go next" below).

## Where to go next

| If you want… | Go to |
|---|---|
| The current research direction | `idea1.md` |
| Latest experiment results and what they mean | `experiments/5cond_factorial/analysis/` |
| The running log of decisions, pivots, and PI feedback | `JOURNAL.md` |
| Chronological history of all sessions | `CURRENT_STATE.md` |
| Training scripts, configs, analysis tools | `experiments/` (each subfolder has its own README) |
| Per-experiment run outputs | `runs/` |
| River-network inference (Phase 1) | `topology_analysis/` |
| Literature review and positioning | `research_papers.md` |
| The upstream NH framework (unmodified) | `neuralhydrology/` |
| Alternative direction set aside in April 2026 | `idea2/` |

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

CAMELS-US dataset: Newman et al. (2015), Addor et al. (2017).

Research conducted at the Baker Lab, UCLA.
