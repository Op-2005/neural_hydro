# Does a streamflow LSTM actually use the river network?

An experimental study built on [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
using the CAMELS-US basins. When a rainfall–runoff LSTM improves after being given a river network,
**what about that network is it actually using?**

Every condition trains the stock `cudalstm` with the same configuration; only the spatial
information added to the input changes. Each experiment is pre-registered and run with three seeds.

## Repository layout

| Path | Contents |
|---|---|
| [`topology_analysis/`](topology_analysis/) | Infers the river network from CAMELS-US basin attributes and builds the basin and edge lists. |
| [`experiments/topology_ablation/`](experiments/topology_ablation/) | The current study: input builders, training runners, pre-registrations and analysis. Start with its README. |
| [`experiments/`](experiments/) | Earlier experiments that led to the current design. |
| `runs/` | Run outputs: configs, metrics and notes. |
| `datasets/camels_us/` | CAMELS-US forcings, attributes and discharge. |
| `neuralhydrology/` | The upstream framework, unmodified. |

## Getting started

```bash
conda env create -f environments/environment_cpu.yml   # or environment_cuda11_8.yml
conda activate neuralhydrology
pip install -e .
```

### See the inferred river network

```bash
# infer upstream-downstream edges across all CAMELS-US basins
python topology_analysis/phase1_network_discovery/discover_network.py
# extract the 183-basin eastern-US network the experiments use
python topology_analysis/phase1_network_discovery/extract_component_0.py
# map the network and its connected components
python topology_analysis/phase1_network_discovery/plot_full_network.py
```

The edge lists and maps are written to `topology_analysis/phase1_network_discovery/outputs/`
(for example `national_network_map.png` and `component0_edges.csv`).

### Train the baseline

```bash
python neuralhydrology/nh_run.py train --config-file experiments/topology_ablation/configs/L_component0_seed11.yaml
python neuralhydrology/nh_run.py evaluate --run-dir runs/<run-folder>
```

### Add neighbour discharge from the network

```bash
# build the lagged upstream-discharge input
python experiments/topology_ablation/build_upstream_discharge_feature.py --network component0 --lag-days 1

# train and evaluate the baseline with that input appended
python experiments/topology_ablation/run_upstream_feature.py --network component0 --seed 11 --device cpu \
    --feature-file experiments/topology_ablation/features/upstream_q_component0_lag1.p \
    --cond-name L_upQ
```

### Swap in a different neighbour set

Any edge list in `outputs/` can drive the same input, for example the two nearest gauges instead of
the river network:

```bash
python experiments/topology_ablation/build_own_lag_inputs.py \
    --edge-file topology_analysis/phase1_network_discovery/outputs/component0_edges_knn2.csv --tag knn2

python experiments/topology_ablation/run_upstream_feature.py --network component0 --seed 11 --device cpu \
    --feature-file experiments/topology_ablation/features/upstream_q_knn2_edgefile_component0_lag1.p \
    --cond-name L_upQknn2
```

Each run writes per-basin metrics to
`runs/topology_ablation/component0/<condition>_component0_seed<s>/test/model_epoch030/test_metrics.csv`.
The other `build_*` and `run_*` scripts in `experiments/topology_ablation/` follow the same pattern
for the remaining conditions, and its `notebooks/` folder has the Colab notebooks used to train
the full set on a GPU.

## Acknowledgements

- [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology): Kratzert, Gauch, Nearing
  and Klotz (2022), *JOSS* 7(71), 4050, [doi:10.21105/joss.04050](https://doi.org/10.21105/joss.04050).
- CAMELS-US: Newman et al. (2015); Addor et al. (2017).
- USGS NLDI / NHDPlus, used to check the inferred network.

## License

BSD 3-Clause, inherited from NeuralHydrology. See [`LICENSE`](LICENSE).
