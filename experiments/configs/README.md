# Configs

NH-format YAML files. Each config is the input to `neuralhydrology/nh_run.py train`
and produces exactly one run under `runs/`. The basin lists referenced inside
each config live in `../basin_lists/`.

| File | Basin set | Basin ID encoding | Produced run | Purpose |
|---|---|---|---|---|
| `lstm_study_network.yaml` | 23 HUC-12 Texas basins | No | `runs/03_lstm_23basin_baseline/` | **Weak baseline** for the pilot. Uses only 5 static attributes. |
| `lstm_study_network_strong.yaml` | 23 HUC-12 Texas basins | **Yes** | `runs/05_lstm_23basin_strong_baseline/` | **Strong Kratzert-style baseline.** Reference line for every graph-variant comparison on the pilot. |
| `lstm_ungauged_train.yaml` | 20 train / 3 held-out (Texas subset) | No | `runs/12_lstm_ungauged_baseline/` | PUB baseline (held-out basins have no learnable basin embedding). |
| `lstm_component0_baseline.yaml` | 183 basins (Component 0, eastern US) | Yes | (not yet run) | Scaled baseline for the Idea-1 A/B/C ablation. Needs compute. |

## Train a config

From the repo root:

```bash
python neuralhydrology/nh_run.py train --config-file experiments/configs/lstm_study_network_strong.yaml
```

## Re-evaluate a trained run

```bash
python neuralhydrology/nh_run.py evaluate --run-dir runs/05_lstm_23basin_strong_baseline --epoch 30
```

## Hyperparameters (shared across pilot configs)

| Param | Value |
|---|---|
| hidden_size | 64 |
| dropout | 0.4 |
| seq_length | 30 |
| batch_size | 256 |
| epochs | 30 |
| learning_rate | 1e-3 (constant) |
| loss | MSE |
| optimizer | Adam |
| predict_last_n | 1 |
| forcings | maurer |
| dynamic_inputs | PRCP, SRAD, Tmax, Tmin, Vp |
| target | QObs(mm/d), clipped to zero |
| static_attributes | elev_mean, area_gages2, slope_mean, p_mean, pet_mean |
| train window | 1990-01-01 to 1999-12-31 |
| val window | 2000-01-01 to 2004-12-31 |
| test window | 2005-01-01 to 2008-12-31 |

These are the Kratzert-2019-aligned defaults. Do not change without a note in
`../../idea1.md` or a run's `NOTES.md`.
