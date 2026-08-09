# Neural Hydrology — Graph-LSTM Experiments (Baker Lab @ UCLA)

Extensions to the [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology) framework (Kratzert et al., JKU Linz) that ask **when and how river-network structure improves LSTM streamflow prediction** over a strong multi-basin LSTM baseline on CAMELS-US. The study is a controlled ablation on stock NeuralHydrology: every condition shares an identical model and training setup, and the only thing that varies is the structural signal added as an input — first static topology features, then dynamic upstream flow. This directly resolves the "does graph structure help streamflow LSTMs?" question left open by prior work (Kirschstein 2024's null; Jiang 2025's physics-aware direction).

## What this is

A research codebase, not a finished product. It holds the experiment code, configurations, and
per-experiment result tables for an ongoing study. The manuscript and the internal research log are
kept as private working documents and are not published here, so some files reference planning docs
that are not part of this repository.

## Where to go next

| If you want… | Go to |
|---|---|
| The current study (controlled ablation, upstream-signal) | `experiments/topology_ablation/` (has its own README) |
| The consolidated results tables | `experiments/topology_ablation/analysis/PAPER_TABLE.md` |
| Pre-registrations for each experiment | `experiments/topology_ablation/preregistration_*.md` |
| Training scripts, configs, analysis tools | `experiments/` (each subfolder has its own README) |
| Per-experiment run outputs and notes | `runs/` |
| River-network inference (Phase 1) | `topology_analysis/` |
| Earlier confounded work (kept for provenance) | `experiments/5cond_factorial/`, `experiments/local_subgraphs/` |
| Alternative direction set aside in April 2026 | `archive/idea2/` |
| The upstream NH framework (unmodified) | `neuralhydrology/` |

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
