# Claude Code Prompt: River Network Discovery and Directed Graph-LSTM for Topology-Aware Streamflow Prediction

## The core situation

This project predicts streamflow (discharge) using NeuralHydrology on CAMELS-US data. A baseline LSTM
is trained on 10 HUC-01 Maine/NH basins and achieves median NSE ~0.73. Those 10 basins are all separate
headwater catchments — no upstream-downstream relationships exist between them — so the topology inference
code in `topology_analysis/infer_topology.py` correctly returns an empty edge set.

This is not a bug. It is a basin selection problem.

The full CAMELS-US dataset (674 basins, all data already downloaded) contains many nested gauge pairs —
places where one USGS gauge is genuinely upstream of another within the same river system. The topology
inference code already handles this correctly. It just needs to run on the full dataset.

This task has three phases, in order of priority:

1. **Network discovery**: Run topology inference on all 674 CAMELS basins. Find every upstream-downstream
   basin pair. Identify connected subnetworks. Select the best study network for the graph experiment.

2. **Baseline retraining**: Retrain the LSTM on the selected network basins to establish a fair NSE
   baseline for the graph comparison.

3. **Directed Graph-LSTM**: Build a model where each basin's LSTM receives lagged hidden states from its
   upstream neighbors as additional input. Train it on the same network basins and compare NSE.

**Phase 1 alone is a complete deliverable.** Phases 2 and 3 follow naturally but may extend beyond a
single session. Prioritize correctness and clarity over speed.

---

## Background: what already exists

**`topology_analysis/infer_topology.py`** — already implements the three heuristics for inferring
upstream-downstream edges:
  - Area accumulation: `area(child) >= area(parent) * threshold` (child basin must be larger)
  - Elevation gradient: `elev(child) < elev(parent)` (downstream is lower)
  - Spatial proximity: Haversine distance <= `max_distance_km`

This code works. It was only run on 10 isolated headwater basins, which is why edges were empty.

**`datasets/camels_us/camels_attributes_v2.0/camels_topo.txt`** — contains topographic attributes
(gauge_id, gauge_lat, gauge_lon, elev_mean, area_gages2) for all 674 basins. This is the input to
topology inference.

**`neuralhydrology/datasetzoo/camelsus.py`** — already loads CAMELS attributes and forcings for any
basin list. The `load_camels_us_attributes()` function accepts a list of basin IDs.

**`neuralhydrology/modelzoo/`** — contains `cudalstm.py` (baseline), `basemodel.py`, `inputlayer.py`,
`head.py`, and `template.py`. A new model must inherit from `BaseModel` and implement `forward()`.

**Baseline NSE** (run `lstm_camels_us_2001_000652`, 5 epochs, 10 HUC-01 basins):
```
01013500: 0.844  |  01030500: 0.794  |  01047000: 0.777  |  01022500: 0.766
01057000: 0.765  |  01052500: 0.748  |  01031500: 0.746  |  01055000: 0.671
01073000: 0.662  |  01054200: 0.613  |  Median: 0.730
```
These basins will NOT be part of the new graph experiment (they have no connectivity). They serve only
as a reference point for what "good LSTM performance" looks like on this dataset.

---

## Phase 1: Full network discovery (priority — do this first)

### 1.1 — Run topology inference on all 674 basins

Create `topology_analysis/discover_network.py`.

Load `camels_topo.txt` for all 674 basins. Run the existing upstream-downstream heuristics with these
parameters (wider than the original 50km default, appropriate for the full national dataset):

```
max_distance_km = 150
area_ratio_threshold = 1.5   # stronger signal: child must be 50% larger
elevation_must_decrease = True
```

The area_ratio_threshold of 1.5 is intentionally conservative. A child basin with 50% more drainage
area than its candidate parent is much more likely to genuinely contain that parent's watershed. This
reduces false positives from basins in different river systems that happen to be geographically close.

For each ordered pair (parent_candidate, child_candidate) across all 674 basins, apply the three
checks. Collect all valid (parent, child) edges into a directed edge list.

Save the full edge list to `topology_analysis/outputs/full_network_edges.csv`:
```
parent_id, child_id, parent_area_km2, child_area_km2, area_ratio, elev_diff_m, distance_km
```

### 1.2 — Identify connected components and study network

Build a `networkx.DiGraph` from the edge list. Find all weakly connected components (ignoring edge
direction). Report:

```
=== CAMELS-US River Network Discovery ===
Total basins: 674
Total directed edges found: N
Connected components with >= 2 nodes: K

Top 10 components by size:
  Component  Nodes  Edges  Max_depth  HUC regions
  1          N      E      D          01, 02, ...
  2          ...
  ...
```

For each component, also report:
- Number of root nodes (no upstream neighbors within CAMELS)
- Number of leaf nodes (no downstream neighbors within CAMELS)
- Maximum path length (root to leaf, measured in hops)
- Geographic bounding box (min/max lat/lon)

### 1.3 — Select the study network

Select the single best component for the graph experiment using these criteria, in order of importance:

1. **Size**: at least 15 nodes, preferably 20-40 (large enough for graph effects to emerge, small
   enough to train quickly)
2. **Depth**: maximum path length >= 3 (need enough hops for message passing to matter)
3. **Geographic coherence**: all basins in the same broad region (same climate regime reduces
   confounding factors)
4. **Data completeness**: all basins have maurer forcings and discharge data in the standard period

If multiple components qualify, pick the largest. If none has depth >= 3, pick the one with the
greatest depth.

Save the selected network to:
```
topology_analysis/outputs/study_network_edges.csv      # directed edge list
topology_analysis/outputs/study_network_basins.txt     # one basin ID per line
topology_analysis/outputs/study_network_summary.txt    # human-readable summary
```

### 1.4 — Visualize the study network

Create `topology_analysis/plot_study_network.py`.

Produce a geographic map saved to `topology_analysis/outputs/study_network_map.png` at 150 DPI.

- Plot basin centroids as nodes at their (lon, lat) coordinates
- Draw directed edges as arrows (upstream → downstream)
- Color nodes by: root (blue), interior (grey), leaf/outlet (red)
- Size nodes proportional to drainage area
- Add basin ID labels (last 5 digits only to avoid clutter)
- Title: "CAMELS-US Study Network — [N] basins, [E] edges, max depth [D]"
- Use matplotlib only (no basemap/cartopy required; plain scatter + annotate is fine)

Also produce a topological layout (ignoring geography) showing the tree/DAG structure:
- Use `networkx.drawing.nx_agraph.graphviz_layout(G, prog='dot')` if graphviz is available,
  otherwise use `networkx.spring_layout`
- Same color scheme as geographic plot
- Save to `topology_analysis/outputs/study_network_dag.png`

---

## Phase 2: Baseline LSTM on the study network

### 2.1 — Create experiment config

Create `experiments/lstm_study_network.yaml` by copying `experiments/lstm_camels_us.yaml` and
changing only:
- `basin_file` pointing to `topology_analysis/outputs/study_network_basins.txt`
- Keep all other settings identical (same forcings, hidden size, epochs, train/val/test periods)

### 2.2 — Train and evaluate

Run:
```bash
python neuralhydrology/nh_run.py train --config-file experiments/lstm_study_network.yaml
python neuralhydrology/nh_run.py evaluate --run-dir runs/lstm_study_network_*/ --epoch 5
```

This establishes the per-basin independent LSTM baseline NSE for the study network. These are the
numbers the Graph-LSTM must beat on downstream basins.

---

## Phase 3: Directed Graph-LSTM

### 3.1 — The scientific hypothesis being tested

A downstream basin's discharge at time t depends not only on its own forcing history, but also on what
happened in upstream basins at times t-1, t-2, ... (water travel time). A per-basin LSTM cannot learn
this because it sees only local forcing. A Graph-LSTM that aggregates upstream hidden states with a
1-day lag can learn the effective propagation kernel of the river network.

The prediction improvement (if any) should be concentrated in **downstream basins** (leaves/outlets),
not in headwater basins — because headwaters have no upstream neighbors to benefit from. This is a
concrete, falsifiable prediction that can be verified from the NSE comparison table.

### 3.2 — Architecture

Create `neuralhydrology/modelzoo/directed_graph_lstm.py`.

```
Class: DirectedGraphLSTM(BaseModel)

At each timestep t, for each basin v:

  1. STANDARD LSTM INPUT (same as baseline):
     x_v^t = embed([forcing_v^t, static_v])
     h_v^t, c_v^t = LSTMCell(x_v^t, h_v^{t-1}, c_v^{t-1})

  2. UPSTREAM MESSAGE (directed, lagged by 1 timestep):
     parents(v) = upstream neighbors of v in the directed graph
     If parents(v) is empty (headwater basin): m_v^t = 0
     If parents(v) is non-empty:
       m_v^t = mean_{u in parents(v)} [ W_upstream * h_u^{t-1} ]
               (W_upstream: hidden_size → hidden_size, learnable)

  3. HIDDEN STATE UPDATE with upstream message (residual):
     h_v^t = h_v^t + tanh(W_msg * m_v^t)
             (W_msg: hidden_size → hidden_size, learnable, init to zeros)

  4. READOUT (same head as baseline):
     y_hat_v = head(dropout(h_v^T))
```

The residual connection in step 3 and zero initialization of W_msg ensures the model starts as a
standard per-basin LSTM and can learn to use upstream signals only when beneficial. This is critical:
it means the Graph-LSTM cannot be worse than the baseline at initialization.

**Key implementation detail**: The directed graph structure (parent-child adjacency) is loaded from
`topology_analysis/outputs/study_network_edges.csv` at model init. It is fixed (not learned). Store
as a dict: `{basin_id: [parent_basin_ids]}`, indexed consistently with the batch ordering.

Register this model in the NeuralHydrology model registry under the name `'directed_graph_lstm'`.

### 3.3 — Training loop adaptation

The upstream message passing requires that at each timestep, the hidden states of ALL basins in the
network are available simultaneously (so a downstream basin can read its upstream neighbor's state).

NeuralHydrology's default training loop processes basins independently in a batch. For the directed
Graph-LSTM, we need to process all study network basins jointly at each timestep.

Add config key `graph_mode: true` to the YAML. When this is set, the training loop should:
- Load sequences for ALL study network basins aligned on the same dates
- Process them jointly through the directed Graph-LSTM
- Compute NSE loss averaged across all basins

If modifying the training loop is too invasive, implement a **standalone training script**
`experiments/train_directed_graph_lstm.py` that:
- Uses NeuralHydrology's data loaders to fetch basin data
- Implements its own training loop with joint multi-basin batching
- Saves checkpoints in the same format as NeuralHydrology runs
- Evaluates and reports per-basin test NSE

### 3.4 — Evaluation and comparison

Create `experiments/compare_results.py` that produces:

```
=== Directed Graph-LSTM vs. Baseline LSTM: Study Network Comparison ===

Basin     Role       Depth  Baseline NSE  Graph-LSTM NSE  Delta   Upstream_count
XXXXXXXX  headwater  0      X.XXX         X.XXX           +X.XXX  0
XXXXXXXX  interior   1      X.XXX         X.XXX           +X.XXX  1
XXXXXXXX  interior   2      X.XXX         X.XXX           +X.XXX  2
XXXXXXXX  outlet     3      X.XXX         X.XXX           +X.XXX  3
...
----------------------------------------------------------------
Headwater basins (depth=0):  median delta = X.XXX  [expected: ~0, graph adds nothing]
Interior basins (depth>=1):  median delta = X.XXX  [expected: positive]
Outlet basins (max depth):   median delta = X.XXX  [expected: largest positive]
Overall median:              X.XXX → X.XXX (delta = X.XXX)
```

The "Depth" column is the basin's maximum distance from any upstream root. This is the key scientific
column: if the graph is working correctly, improvement should increase with depth.

---

## File structure after this task

```
neural_hydrology/
├── topology_analysis/
│   ├── discover_network.py            ← NEW (Phase 1.1-1.3)
│   ├── plot_study_network.py          ← NEW (Phase 1.4)
│   └── outputs/
│       ├── full_network_edges.csv     ← NEW
│       ├── study_network_edges.csv    ← NEW
│       ├── study_network_basins.txt   ← NEW
│       ├── study_network_summary.txt  ← NEW
│       ├── study_network_map.png      ← NEW
│       └── study_network_dag.png      ← NEW
│
├── neuralhydrology/modelzoo/
│   └── directed_graph_lstm.py         ← NEW (Phase 3)
│
└── experiments/
    ├── lstm_study_network.yaml        ← NEW (Phase 2)
    ├── directed_graph_lstm.yaml       ← NEW (Phase 3)
    ├── train_directed_graph_lstm.py   ← NEW (Phase 3, if standalone)
    └── compare_results.py             ← NEW (Phase 3)
```

All existing files are unchanged.

---

## Implementation notes

**Topology inference parameters**: The original code used 50km and area_ratio=1.1. For the full
national dataset, use 150km and 1.5. The larger distance accommodates bigger basins in the western US.
The stricter area ratio reduces false positives. If the full 674-basin run produces fewer than 20
edges total, relax to 100km / 1.3 and re-run.

**CAMELS topo file format**: separator is `;`, columns include `gauge_id` (string, zero-padded to 8
digits), `gauge_lat`, `gauge_lon`, `elev_mean` (meters), `area_gages2` (km²). Load with
`pd.read_csv(..., sep=';', dtype={'gauge_id': str})`.

**Directed graph consistency**: In the directed graph, edges point FROM upstream TO downstream
(parent → child). When aggregating upstream messages for basin v, aggregate over `G.predecessors(v)`,
not `G.successors(v)`.

**Basin ordering in joint training**: Fix a canonical ordering of basins (sorted basin IDs). All
tensors use this ordering consistently. Store as `basin_id_to_idx` dict.

**Reproducibility**: `torch.manual_seed(42)` and `numpy.random.seed(42)` at script tops.

---

## What success looks like

After Phase 1:
- Running `python topology_analysis/discover_network.py` prints the network summary and saves files
- Running `python topology_analysis/plot_study_network.py` produces both map and DAG figures
- The study network has >= 15 basins, >= 1 directed edge, >= 3 max depth

After Phase 2:
- The LSTM trains on the study network basins without errors
- Per-basin NSE is reported and comparable to the original 10-basin baseline

After Phase 3:
- `compare_results.py` produces the comparison table
- The key scientific test: downstream basins (depth >= 2) show positive delta NSE

---
---

# Updated Next Steps (April 14, 2026)

## What has been completed

| Phase | Status | Key result |
|-------|--------|------------|
| 1 — Network discovery | Done | 1,298 edges across 671 basins; 23-basin HUC-12 study network selected |
| 2 — Baseline LSTM | Done | Median NSE 0.407 on study network (30 epochs, CudaLSTM) |
| 3 — Graph-LSTM (first run) | Done | Median NSE 0.329 (10 epochs, undertrained) |

All infrastructure is in place: data loading, topology, model, training script, evaluation,
and comparison table. The Graph-LSTM's first run underperformed because it had 1/3 the
training epochs of the baseline and its loss was still dropping. The architecture is functional
and the experiment pipeline works end-to-end.

## Immediate next steps (priority order)

### 1. Train the Graph-LSTM for more epochs

The single highest-impact action. The current 10-epoch run had loss 0.52 vs the baseline's
converged 0.11. The model was clearly still learning.

**Option A — Long run on CPU (~2.5 hours):**
Change `EPOCHS = 30` in `experiments/train_graph_lstm.py` and rerun. This gives equal epoch
count to the baseline. At ~5 min/epoch on CPU, expect ~2.5 hours. Run overnight or during a
meeting.

```bash
/Applications/anaconda3/envs/nh/bin/python experiments/train_graph_lstm.py
```

**Option B — GPU acceleration:**
If a CUDA GPU is available (lab machine, cloud instance), change `DEVICE = torch.device("cuda:0")`
in the script. The LSTMCell loop is Python-bound so the speedup may be modest (~2x), but it
helps. The real bottleneck is the sequential timestep loop, not tensor operations.

**Option C — Warm-start from the baseline:**
Initialize the Graph-LSTM's LSTMCell weights from the trained baseline CudaLSTM checkpoint
(`runs/lstm_study_network_1304_222043/model_epoch030.pt`). This way the LSTM part starts
already converged and only the message passing weights (W_upstream, W_msg) need to be learned.
This would dramatically reduce required training time and make the comparison fairer, since
the baseline LSTM component is identical. This is the recommended approach.

### 2. Fair comparison design

The current comparison is confounded by different training effort. For a scientifically valid
result, one of these must be true:
- Same number of effective gradient steps (epochs x batches_per_epoch)
- Same final training loss (train until both converge)
- Warm-start the Graph-LSTM from the baseline (so only the graph component needs training)

The warm-start approach (Option C above) is the cleanest experimental design because it
isolates the effect of the graph structure from the effect of different optimization
trajectories.

### 3. Analyze per-basin results more deeply

Even with the undertrained model, some patterns are worth investigating:
- Basin 08150800 improved from -0.85 to +0.42 — why? Is it the batching difference or
  something about the graph structure?
- The depth-3 basins (08189500, 08164000) barely degraded despite undertraining — is the
  graph helping them resist degradation?
- Generate per-basin hydrographs (observed vs predicted time series) for 3-4 interesting
  basins to visually inspect where the models differ.

### 4. Speed up the training loop

The ~5 min/epoch bottleneck comes from the Python-level timestep loop (30 LSTMCell calls per
window, 3652 windows per epoch). Potential optimizations:
- **Subsample training windows**: Use 1/3 of windows per epoch (random subset), train for 3x
  more epochs. Same total gradient steps, but each epoch is faster and the model sees more
  data diversity.
- **Batch the LSTMCell calls**: Instead of processing one window at a time, process multiple
  windows simultaneously (batch dimension). The current code already batches windows but
  processes them in a Python loop — vectorizing this would help.
- **Use nn.LSTM for non-message timesteps**: Process the first 29 timesteps with nn.LSTM
  (fast) and only apply message passing at the final timestep. This loses the full inter-
  timestep message passing but keeps 96% of the speed.

### 5. Ablation experiments

Once the Graph-LSTM trains to convergence, run ablations to isolate what helps:
- **Random graph**: Shuffle the edges randomly (same number, different connections). If this
  performs as well as the real graph, the specific topology doesn't matter.
- **No-lag variant**: Use `h_u^t` instead of `h_u^{t-1}` (same-timestep message). If this
  hurts, the lag is important (consistent with physical water travel time).
- **Deeper message passing**: Apply 2 rounds of message passing per timestep (extending the
  receptive field). The Phase 0 signal decay experiments predicted this extends reach by ~1
  hop.

### 6. Longer-term directions

- **Replace heuristic edges with NHDPlus flowlines** for ground-truth river connectivity.
  The current 34-edge network likely has false positives.
- **Scale to a larger component** (Component 0: 183 basins, depth 4) once the approach is
  validated on the 23-basin network.
- **Multi-timescale message passing**: Lag by more than 1 day for distant upstream-downstream
  pairs (water travel time increases with distance).
- **Attention-based message aggregation**: Replace mean-pooling with learned attention weights
  over upstream neighbors, allowing the model to weight nearby vs distant upstream basins
  differently.

## Files created during Phase 3

```
experiments/train_graph_lstm.py           Standalone Graph-LSTM training script
experiments/compare_results.py            Depth-stratified comparison table
runs/graph_lstm_study_network_1404_*/     First Graph-LSTM run (10 epochs)
    test_metrics.csv                        Per-basin NSE results
    model_epoch*.pt                         Checkpoints
    model_best.pt                           Best checkpoint (by test NSE)
    run_config.json                         Hyperparameters and metadata
```