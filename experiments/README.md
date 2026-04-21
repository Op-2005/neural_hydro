# Experiments — Index

This folder holds everything needed to **run** or **audit** any experiment in
this fork. Nothing here produces *new* claims — it produces the artifacts
(runs, figures, CSVs) that the markdown docs in the repo root argue from.

If you want to find something, start here:

| I want to… | Go to |
|---|---|
| See *what experiments exist* and *what each one showed* | `runs/README.md` (run-by-run table) + each `runs/<N>_*/NOTES.md` (brief per-run writeup) |
| See the *current research plan* | `../idea1.md` |
| See the *alternative plan set aside* | `../idea2/README.md` |
| Train the **LSTM baseline** for a given setting | [`configs/`](configs/) — pick a YAML + `python neuralhydrology/nh_run.py train --config-file <path>` |
| Train the **graph-LSTM** (or its ablation variants) | [`training/`](training/) — `train_graph_lstm.py`, `train_graph_ungauged.py`, `train_graph_component0.py` |
| **Compare** a baseline run with one or more graph-variant runs | [`analysis/compare_results.py`](analysis/compare_results.py) |
| Generate **hydrographs** / weight-inspection plots / per-basin deltas | [`analysis/analyze_results.py`](analysis/analyze_results.py) |
| Re-run any **post-hoc analysis** described in `INSIGHTS.md` | [`analysis/`](analysis/) |
| Inspect the **basin IDs** used by a run | [`basin_lists/`](basin_lists/) |
| Inspect the **figures / CSVs already produced** | [`analysis_outputs/`](analysis_outputs/) |

## Subfolder layout

```
experiments/
├── configs/           NH-format YAML configs. Each produces one named run.
├── basin_lists/       USGS gauge-ID lists referenced by configs & training scripts.
├── training/          Training scripts (LSTM baselines + graph-LSTM variants).
├── analysis/          Post-hoc analysis scripts run against completed runs.
├── analysis_outputs/  Results produced by analysis scripts (CSVs, PNGs, JSON).
└── README.md          (this file)
```

## Running anything from here

All commands below assume you are in the repo root and have
`conda activate nh` (or equivalent) done.

```bash
# 1. Train the 23-basin strong baseline (produces a run under runs/lstm_study_network_strong_*)
python neuralhydrology/nh_run.py train --config-file experiments/configs/lstm_study_network_strong.yaml

# 2. Train the headline graph-LSTM (edit flags at top of script to change variant)
python experiments/training/train_graph_lstm.py

# 3. Compare strong baseline vs any graph variant
python experiments/analysis/compare_results.py \
    --baseline runs/05_lstm_23basin_strong_baseline/test/model_epoch030/test_metrics.csv \
    --baseline-label "Strong LSTM" \
    --graph runs/06_graph_edge_warm_full/test_metrics.csv:Edge+Warm \
            runs/11_graph_edge_pruned_edges/test_metrics.csv:Pruned

# 4. Train the scaled Component-0 baseline (not yet run — waiting on compute decision)
python neuralhydrology/nh_run.py train --config-file experiments/configs/lstm_component0_baseline.yaml
python experiments/training/train_graph_component0.py --variant warm --seed 42 --epochs 15
```

## Audit trail — pilot

Follow this chain to audit any pilot claim from the 23-basin experiment:

1. Find the *claim* in [`../INSIGHTS.md`](../INSIGHTS.md) (e.g. "Aggregation-
   family variants all converge").
2. Identify which runs it rests on via [`../runs/README.md`](../runs/README.md)
   (for this example: runs 06, 08, 09, 10).
3. Open each run's `NOTES.md` for a brief per-run writeup.
4. Open `test_metrics.csv` in each run for per-basin numbers.
5. Reproduce the aggregated view with
   `experiments/analysis/compare_results.py` passing those metric files.

## Audit trail — scaling

For the Component-0 scale-up plan (not yet run):

1. Plan lives in [`../idea1.md`](../idea1.md).
2. Basin list: `topology_analysis/phase1_network_discovery/outputs/component0_basins.txt`.
3. Edge list: `topology_analysis/phase1_network_discovery/outputs/component0_edges.csv`.
4. Baseline config: `configs/lstm_component0_baseline.yaml`.
5. Graph trainer: `training/train_graph_component0.py --variant {warm,frozen,gcn_lowpass} --seed <n>`.
