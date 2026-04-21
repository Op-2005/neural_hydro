# Training Scripts

All training scripts for the graph-LSTM and its ablations. The plain LSTM
baselines use the NH framework directly via `neuralhydrology/nh_run.py train`
with configs from `../configs/` — no script in this folder is needed for them.

| File | Produces which runs | What it does |
|---|---|---|
| `train_graph_lstm.py` | `runs/06_graph_edge_warm_full/` through `runs/11_graph_edge_pruned_edges/` | **Main graph-LSTM trainer on the 23-basin pilot network.** Uses the DirectedGraphLSTM class (also defined here) with a custom timestep loop over an `nn.LSTMCell` + directed upstream-parent message passing. 7 flags at the top of the file switch ablation variants (edge features, Jiang-diff term, attention, sigmoid gate, frozen-LSTM, warm-start, basin encoding). Edit flags → run → creates a new `runs/graph_lstm_<tags>_<timestamp>/` folder. |
| `train_graph_ungauged.py` | `runs/13_graph_ungauged/` | **PUB version.** Same DirectedGraphLSTM, but the training loss masks out the 3 held-out basins while the forward pass still produces predictions for them via message passing from their (training-set) parents. |
| `train_graph_component0.py` | (not yet launched) | **Scaled version for Component 0 (183 basins).** CLI-parameterized (`--variant {warm,frozen,gcn_lowpass}`, `--seed`, `--epochs`, `--smoke-test`) — no editing flags. Needs a trained NH baseline at `runs/lstm_component0_baseline_*/` to warm-start from. For the Idea-1 ablation we will add a fourth `--variant topology_features` once the plan is green-lit. |

## The DirectedGraphLSTM in one sentence

At each timestep `t`, each basin `v` is advanced by an LSTMCell with its own
forcings + static attributes, then adds `tanh(W_out · msg_v)` as a residual,
where `msg_v = mean_u∈parents(v) W_msg · [h_u, e_uv]` — i.e., the mean of
transformed messages from each upstream parent, with the parent's *previous
timestep* hidden state `h_u(t-1)` (so the lag encodes physical travel time).
`W_out` is zero-initialized so the model starts identical to the warm-started
LSTM; the residual only grows with training.

## Running these

From the repo root:

```bash
# 23-basin graph-LSTM (edit flags at top of script first)
python experiments/training/train_graph_lstm.py

# Ungauged / PUB
python experiments/training/train_graph_ungauged.py

# Component 0 scale-up — quick timing estimate first
python experiments/training/train_graph_component0.py --variant warm --seed 42 --smoke-test
```

## Where to see what each run produced

Each produced run has its own `NOTES.md` inside the run folder summarizing the
numbers and the role of that run in the narrative. See
`../../runs/README.md` for the full index.

## Key cross-folder references

| Script | Reads |
|---|---|
| `train_graph_lstm.py` | `../basin_lists/study_network_basins.txt`, `topology_analysis/phase1_network_discovery/outputs/study_network_edges.csv`, `runs/05_lstm_23basin_strong_baseline/` (for warm-start) |
| `train_graph_ungauged.py` | `../basin_lists/{study_network,ungauged_train,ungauged_test}_basins.txt`, same edge file, `runs/12_lstm_ungauged_baseline/` |
| `train_graph_component0.py` | `topology_analysis/phase1_network_discovery/outputs/component0_{basins.txt,edges.csv}`, most-recent `runs/lstm_component0_baseline_*/` |
