# Does a streamflow LSTM actually use the river network?

An experimental study built on [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
using the CAMELS-US basins. When a rainfall–runoff LSTM improves after being given a river network,
**what about that network is it actually using?**

Every condition trains the stock `cudalstm` with the same configuration; only the spatial
information added to the input changes. Each experiment is pre-registered, run with three seeds,
and evaluated by scripts that regenerate every number from the raw run metrics.

## Repository layout

| Path | Contents |
|---|---|
| [`experiments/topology_ablation/`](experiments/topology_ablation/) | The current study. Start with its README; [`analysis/VERIFIED_NUMBERS.md`](experiments/topology_ablation/analysis/VERIFIED_NUMBERS.md) holds every reported result. |
| [`experiments/`](experiments/) | Earlier experiments that led to the current design. |
| [`topology_analysis/`](topology_analysis/) | Builds the inferred river network and basin sets. |
| `runs/` | Raw run outputs: configs, metrics and notes. |
| `datasets/camels_us/` | CAMELS-US forcings, attributes and discharge. |
| `neuralhydrology/` | The upstream framework, unmodified. |

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

## Citation

*A Streamflow Forecasting Model Learns Proximity, Not Connectivity, from an Inferred River Network.*
Om Phadke and Justin Baker (UCLA). XAI4Science Workshop at NeurIPS 2026.

## Acknowledgements

- [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology): Kratzert, Gauch, Nearing
  and Klotz (2022), *JOSS* 7(71), 4050, [doi:10.21105/joss.04050](https://doi.org/10.21105/joss.04050).
- CAMELS-US: Newman et al. (2015); Addor et al. (2017).
- USGS NLDI / NHDPlus, used to check the inferred network.

## License

BSD 3-Clause, inherited from NeuralHydrology. See [`LICENSE`](LICENSE).
