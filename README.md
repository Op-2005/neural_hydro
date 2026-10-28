# Does a streamflow LSTM actually use the river network?

An experimental study built on [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology),
using the CAMELS-US basins.

## Research question

Models of physical systems are often built on top of a known network: bonded atoms, connected
power lines, rivers draining into one another. When such a model gets better, people usually take
that as evidence the model used the physical relation the network encodes. That is rarely tested
directly, because it means varying the graph while holding the model and its training fixed.

This repository runs that test for daily streamflow prediction. **When a rainfall–runoff LSTM
improves after being given a river network, what about the network is it actually using?**

## Approach

Every condition trains the **stock `cudalstm`** with the same configuration, apart from one change:
the same hyperparameters, data split and seeds. Only the spatial information appended to the input
varies, so any change in accuracy comes from that information.

- **One variable at a time.** Configs are diffed against the baseline to confirm the only
  difference is the added input.
- **Pre-registered.** Each experiment's hypothesis and falsification criterion are written down
  before its runs exist. Predictions that failed are reported as failed.
- **Three seeds, judged by the weakest.** Significance is never pooled across seeds.
- **Generated numbers.** Every reported value is regenerated from the raw run metrics by a script
  rather than transcribed by hand.

## Experiments

| Experiment | What it asks |
|---|---|
| Static vs. dynamic encoding | Does the network help as a fixed per-basin label (depth, upstream count, upstream area), or only as a daily signal (neighbours' discharge from the previous day)? |
| Date-shuffled control | Is a gain from neighbour discharge real timing information, or just extra model capacity? |
| Distance-matched substitution | Replace every upstream neighbour with a basin that is *not* upstream but lies at about the same distance. Does the gain survive? |
| Distance sweep and nearest gauges | How does the gain change as the substitutes move further away, and what happens if neighbours are picked by distance alone? |
| Predicted neighbour flow | Do the conclusions hold when neighbour discharge must be forecast rather than observed? |
| Flow bands and precipitation | Where in the flow regime does the gain sit, and does upstream rainfall reproduce it? |
| Own lagged discharge | How much of the neighbour signal does the basin's own recent discharge already carry? |
| Hydrography check | How many of the inferred network's edges are real upstream relations in NHDPlus (via USGS NLDI)? |

The study also includes an oracle upper bound, a random rewiring control and a routing baseline
that uses no machine learning.

## Findings in brief

- **Encoding matters.** As a static label the network adds nothing, because a shared LSTM can
  already learn any per-basin constant. As daily neighbour discharge it helps, and shuffling the
  dates removes the gain.
- **Proximity, not connectivity.** Non-upstream basins at a matched distance give the same gain,
  and the gain falls off as they move further away. Simply taking the two nearest gauges does
  better than the river network.
- **Scope.** The nearest-gauge advantage disappears when neighbour discharge has to be predicted.
  Most of the neighbour gain is already carried by the basin's own previous-day discharge. Few of
  the inferred edges turn out to be real hydrographic connections.

The network is **inferred heuristically** from drainage area, elevation and distance, not taken
from surveyed hydrography. The basins are a regional subset of CAMELS-US in the eastern United
States. All effect sizes and tests, with both tails, are in
[`experiments/topology_ablation/analysis/VERIFIED_NUMBERS.md`](experiments/topology_ablation/analysis/VERIFIED_NUMBERS.md).

This is a research codebase kept as a chronological record. Some earlier result files carry
interpretations that later controls overturned. Each analysis README marks which results still
stand.

## Repository layout

| Path | Contents |
|---|---|
| [`experiments/topology_ablation/`](experiments/topology_ablation/) | The current study: feature builders, configs, pre-registrations, notebooks, analysis scripts and figures. Start with its README. |
| [`experiments/`](experiments/) | Earlier experiments, kept to show how the project got to its current design (see [`experiments/README.md`](experiments/README.md)). |
| [`topology_analysis/`](topology_analysis/) | Builds the inferred river network and the basin sets the experiments use. |
| `runs/` | Raw run outputs: per-run configs, metrics and notes. |
| `datasets/camels_us/` | The CAMELS-US forcings, attributes and discharge. |
| `neuralhydrology/` | The upstream NeuralHydrology framework, unmodified. |

## Reproducing

```bash
conda env create -f environments/environment_cpu.yml   # or environment_cuda11_8.yml
conda activate neuralhydrology
pip install -e .

# train one condition
python neuralhydrology/nh_run.py train --config-file experiments/topology_ablation/configs/L_component0_seed11.yaml

# regenerate every reported number from the runs on disk
python experiments/topology_ablation/verify_all_numbers.py
```

Each run writes its metrics to
`runs/topology_ablation/component0/<condition>_component0_seed<s>/test/model_epoch030/test_metrics.csv`.
Every table and figure is computed from these files. The full set of conditions was trained in the
Colab notebooks under `experiments/topology_ablation/notebooks/`.

## Citation

The accompanying paper is *A Streamflow Forecasting Model Learns Proximity, Not Connectivity, from an
Inferred River Network*, by Om Phadke and Justin Baker (UCLA), XAI4Science Workshop at NeurIPS 2026.

## Acknowledgements

- **NeuralHydrology** by Kratzert, Gauch, Nearing and Klotz is the modelling framework. Please cite
  it if you use this code:

  ```bibtex
  @article{kratzert2022joss,
    title   = {NeuralHydrology --- A Python library for Deep Learning research in hydrology},
    author  = {Frederik Kratzert and Martin Gauch and Grey Nearing and Daniel Klotz},
    journal = {Journal of Open Source Software},
    year    = {2022},
    volume  = {7}, number = {71}, pages = {4050},
    doi     = {10.21105/joss.04050},
  }
  ```

- **CAMELS-US** by Newman et al. (2015) and Addor et al. (2017) provides the basin forcings,
  attributes and streamflow.
- **USGS NLDI / NHDPlus** provides the hydrography used to check the inferred network.

## License

BSD 3-Clause, inherited from NeuralHydrology. See [`LICENSE`](LICENSE).
