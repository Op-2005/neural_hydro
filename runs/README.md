# Experiment Runs

## Index

| # | Folder | Model | Basins | Epochs | Median NSE | Notes |
|---|--------|-------|--------|--------|-----------|-------|
| 01 | `01_lstm_10basin_huc01/` | CudaLSTM | 10 (Maine/NH) | 5 | 0.730 | First successful run |
| 02 | `02_lstm_10basin_huc01_best/` | CudaLSTM | 10 (Maine/NH) | 5 | 0.730 | Best HUC-01 run |
| 03 | `03_lstm_23basin_baseline/` | CudaLSTM | 23 (Texas HUC-12) | 30 | 0.407 | Study network baseline |
| 04 | `04_graph_lstm_23basin/` | DirectedGraphLSTM | 23 (Texas HUC-12) | 10 | 0.329 | First graph run (undertrained) |

## Run Details

### 01 & 02 — LSTM on 10 HUC-01 Basins

```
Model:           CudaLSTM (nn.LSTM)
Basins:          10 headwater basins in Maine/New Hampshire
Forcings:        maurer (PRCP, SRAD, Tmax, Tmin, Vp)
Static attrs:    elev_mean, area_gages2, slope_mean, p_mean, pet_mean
Hidden size:     64
Epochs:          5
Batch size:      32
Train period:    1990-1999
Test period:     2005-2008
Parameters:      ~19k
Median test NSE: 0.730
Best basin:      01013500 (NSE 0.844)
Worst basin:     01054200 (NSE 0.613)
```

### 03 — Baseline LSTM on 23-Basin Study Network

```
Model:           CudaLSTM (nn.LSTM)
Basins:          23 basins in south-central Texas (HUC-12)
Forcings:        maurer (PRCP, SRAD, Tmax, Tmin, Vp)
Static attrs:    elev_mean, area_gages2, slope_mean, p_mean, pet_mean
Hidden size:     64
Epochs:          30
Batch size:      256
Train period:    1990-1999
Test period:     2005-2008
Parameters:      ~19k
Final train loss: 0.11
Median test NSE: 0.407
Best basin:      08189500 (NSE 0.757, depth 3)
Worst basin:     08150800 (NSE -0.847, depth 0)
```

This is the baseline that the Graph-LSTM must beat.

### 04 — Directed Graph-LSTM on 23-Basin Study Network

```
Model:           DirectedGraphLSTM (LSTMCell + upstream message passing)
Basins:          23 basins in south-central Texas (HUC-12)
Graph:           34 directed edges, max depth 3
Forcings:        maurer (PRCP, SRAD, Tmax, Tmin, Vp)
Static attrs:    elev_mean, area_gages2, slope_mean, p_mean, pet_mean
Hidden size:     64
Epochs:          10 (undertrained — loss still dropping)
Batch size:      256
Train period:    1990-1999
Test period:     2005-2008
Parameters:      ~28k (19k LSTM + 8k message passing + 1k head)
Final train loss: 0.52 (vs baseline's converged 0.11)
Median test NSE: 0.329
Best basin:      08189500 (NSE 0.739, depth 3)
Worst basin:     08190500 (NSE -1.426, depth 1)
```

Undertrained — 10 epochs at ~5 min/epoch vs baseline's 30 epochs at ~20 sec/epoch.
Loss was still dropping. See CURRENT_STATE.md for full analysis.

## What each file is

```
config.yml               NeuralHydrology experiment config (frozen at run time)
model_epoch*.pt          PyTorch model checkpoint (final epoch only, intermediates deleted)
model_best.pt            Best checkpoint by test NSE (graph run only)
output.log               Full training + evaluation log
test_metrics.csv         Per-basin NSE on test period
run_config.json          Hyperparameters and metadata (graph run only)
train_data_scaler.yml    Z-score normalization parameters (mean/std per feature)
img_log/                 Training curves (NH runs only)
```

## How to evaluate a run

```bash
# NH runs (01, 02, 03)
/Applications/anaconda3/envs/nh/bin/python neuralhydrology/nh_run.py evaluate \
    --run-dir runs/03_lstm_23basin_baseline --epoch 30

# Compare baseline vs graph
/Applications/anaconda3/envs/nh/bin/python experiments/compare_results.py \
    --baseline runs/03_lstm_23basin_baseline/test/model_epoch030/test_metrics.csv \
    --graph runs/04_graph_lstm_23basin/test_metrics.csv
```
