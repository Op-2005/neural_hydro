# Neural Hydrology — Graph-LSTM Experiments (Baker Lab @ UCLA)

Extensions to the [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) framework (Kratzert et al., JKU Linz) that ask **when and how river-network structure improves LSTM streamflow prediction** over a strong multi-basin LSTM baseline on CAMELS-US. The study is a controlled ablation on stock NeuralHydrology: every condition shares an identical model and training setup, and the only thing that varies is the structural signal added as an input — first static topology features, then dynamic upstream flow. This directly resolves the "does graph structure help streamflow LSTMs?" question left open by prior work (Kirschstein 2024's null; Jiang 2025's physics-aware direction).

## What this is

A research codebase, not a finished product. The work is ongoing; current results, open questions, and the next experiments are tracked elsewhere (see "Where to go next" below).

## Where to go next

| If you want… | Go to |
|---|---|
| The current study (controlled ablation, upstream-signal) | `experiments/topology_ablation/` (has its own README) |
| The running log of decisions, results, and pivots | `JOURNAL.md` |
| A quick plain-language status brief | `updates.md` |
| The original research direction + protocol | `idea1.md` |
| Earlier confounded work (kept for provenance) | `experiments/5cond_factorial/`, `experiments/local_subgraphs/` |
| Training scripts, configs, analysis tools | `experiments/` (each subfolder has its own README) |
| Per-experiment run outputs | `runs/` |
| River-network inference (Phase 1) | `topology_analysis/` |
| Literature review and positioning | `research_papers.md` |
| Chronological session history | `CURRENT_STATE.md` |
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
