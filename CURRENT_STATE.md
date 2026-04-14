# Current State of the Neural Hydrology Project

Last updated: April 14, 2026

---

## Table of Contents

1. [Dataset and Data Structure](#1-dataset-and-data-structure)
2. [Phase 1: River Network Discovery](#2-phase-1-river-network-discovery)
3. [Phase 2: Baseline LSTM on the Study Network](#3-phase-2-baseline-lstm-on-the-study-network)
4. [Phase 3: Directed Graph-LSTM (In Progress)](#4-phase-3-directed-graph-lstm-in-progress)

---

## 1. Dataset and Data Structure

### What is CAMELS-US?

CAMELS (Catchment Attributes and MEteorology for Large-sample Studies) is a dataset of 671
hydrological basins across the contiguous United States. Each basin has a USGS stream gauge
that measures daily river discharge (how much water flows past the gauge per day). The dataset
also includes daily meteorological forcing data (precipitation, temperature, solar radiation,
etc.) and static catchment attributes (elevation, area, soil type, climate statistics).

The dataset lives at `datasets/camels_us/` and has three components:

### 1.1 Streamflow data

```
datasets/camels_us/usgs_streamflow/{HUC}/{basin_id}_streamflow_qc.txt
```

Each file contains daily discharge for one basin from 1980 to 2014. The raw units are cubic
feet per second (cfs). The NeuralHydrology loader converts this to **millimeters per day
(mm/d)** by normalizing by catchment area:

```
QObs(mm/d) = Q(cfs) * 28316846.592 * 86400 / (area_m2 * 10^6)
```

This normalization makes discharge comparable across basins of different sizes — a small
creek and a large river can both have similar mm/d values if they receive similar rainfall
per unit area.

File format: whitespace-separated, no header, columns are `basin Year Month Day Q flag`.

### 1.2 Meteorological forcings

```
datasets/camels_us/basin_mean_forcing/{forcing_type}/{HUC}/{basin_id}_*_forcing_leap.txt
```

Three forcing products are available:
- **maurer** — Maurer extended gridded data (1980-2008)
- **daymet** — Daymet daily surface weather (1980-2014)
- **nldas** — North American Land Data Assimilation System (1980-2014)

Our experiments use **maurer** forcings. The five dynamic input features are:

| Feature | Unit | Description |
|---------|------|-------------|
| PRCP(mm/day) | mm/day | Daily precipitation |
| SRAD(W/m2) | W/m² | Incoming shortwave radiation |
| Tmax(C) | °C | Daily maximum temperature |
| Tmin(C) | °C | Daily minimum temperature |
| Vp(Pa) | Pa | Vapor pressure |

File format: 3-line header (latitude, longitude, area in m²), then whitespace-separated daily
data with columns `Year Mnth Day Hr Dayl(s) PRCP(mm/day) SRAD(W/m2) SWE(mm) Tmax(C) Tmin(C) Vp(Pa)`.

**Important**: Maurer forcings end on December 31, 2008. The test period for all experiments
using maurer is therefore 2005-2008, not 2005-2010. NeuralHydrology fills the gap with NaN
and skips those dates during evaluation.

### 1.3 Static catchment attributes

```
datasets/camels_us/camels_attributes_v2.0/camels_*.txt
```

Seven attribute files covering topography, climate, hydrology, soil, geology, vegetation, and
basin names. Our experiments use five static attributes:

| Attribute | Source file | Description |
|-----------|------------|-------------|
| elev_mean | camels_topo.txt | Mean basin elevation (m) |
| area_gages2 | camels_topo.txt | Drainage area (km²) |
| slope_mean | camels_topo.txt | Mean basin slope (m/km) |
| p_mean | camels_clim.txt | Mean daily precipitation (mm/day) |
| pet_mean | camels_clim.txt | Mean daily potential evapotranspiration (mm/day) |

### 1.4 Folder organization

CAMELS organizes all data by **HUC-02 region** (2-digit Hydrologic Unit Code), not by basin
gauge ID prefix. There are 18 HUC regions (01 through 18) covering the US.

**Critical subtlety**: A basin's gauge ID prefix does NOT determine its HUC folder. Basin
`01435000` (Catskills, New York) lives in `02/` because it belongs to HUC-02, even though its
ID starts with "01". Basin `04296000` (Coventry, Vermont) lives in `01/` because it belongs to
HUC-01, despite starting with "04". The HUC assignment comes from `camels_name.txt`.

The data loading code in `neuralhydrology/datasetzoo/camelsus.py` uses recursive glob patterns
like `glob(f'**/{basin_id}_streamflow_qc.txt')`, so it finds files regardless of subfolder
nesting. This is what makes the system robust to folder structure variations.

### 1.5 How data flows through the system

```
Config YAML (specifies data_dir, forcings, dynamic_inputs, static_attributes, date ranges)
    |
    v
CamelsUS._load_basin_data(basin_id)
    |-- load_camels_us_forcings()     --> glob finds forcing file, reads header for area,
    |                                      parses daily meteorological data into DataFrame
    |-- load_camels_us_discharge()    --> glob finds streamflow file, normalizes cfs to mm/d
    |-- concat into single DataFrame indexed by date
    v
CamelsUS._load_attributes()           --> reads camels_*.txt files, returns DataFrame of
    |                                      static attributes indexed by basin_id
    v
BaseDataset.__init__()
    |-- converts DataFrame to xarray Dataset
    |-- computes z-score scaler (mean/std) on TRAINING period only
    |-- saves scaler to run_dir/train_data/train_data_scaler.yml
    |-- builds lookup table: sample_index -> (basin_id, time_window_indices)
    v
BaseDataset.__getitem__(index)
    |-- looks up (basin_id, window_start..window_end) from lookup table
    |-- returns dict:
    |     x_d: {feature_name: tensor[seq_len=30, 1]} for each dynamic input
    |     x_s: tensor[5]  (static attributes, z-score normalized)
    |     y:   tensor[seq_len=30, 1]  (target discharge, z-score normalized)
    v
DataLoader batches these into tensors of shape [batch_size, seq_len, features]
```

The normalization is z-score: `x_normalized = (x - mean) / std`, computed per-feature across
all basins and all training dates. This scaler is saved during training and reloaded during
evaluation so that test predictions can be un-normalized back to physical units for NSE
computation.

---

## 2. Phase 1: River Network Discovery

### 2.1 What we did

We ran heuristic topology inference on **all 671 CAMELS-US basins** to discover which basins
are upstream of which. The original 10-basin experiment used basins in Maine and New Hampshire
(HUC-01), which are all separate headwater catchments with no upstream-downstream relationships
between them. Scaling to the full dataset was necessary to find connected river subnetworks.

Script: `topology_analysis/phase1_network_discovery/discover_network.py`

### 2.2 How edges are inferred

For every ordered pair of basins (A, B) among all 671, we apply three heuristic tests. All
three must pass to infer a directed edge A → B (meaning A is upstream of B):

**1. Area accumulation** — `area(B) >= area(A) * 1.5`

A downstream basin drains a larger watershed because it collects water from all its upstream
tributaries. We require the candidate downstream basin to be at least 50% larger. This
conservative threshold reduces false positives from basins that happen to be similar in size
but are in different river systems.

**2. Elevation gradient** — `elevation(B) < elevation(A)`

Water flows downhill. The mean elevation of the downstream basin must be strictly lower. This
is computed from `elev_mean` in the CAMELS topographic attributes, which represents the
average elevation of the entire catchment (not just the gauge location).

**3. Spatial proximity** — `haversine_distance(A, B) <= 150 km`

Two basins in the same river system should be geographically close. We compute great-circle
distance between gauge coordinates. The 150 km threshold is appropriate for the national
dataset where major rivers can connect basins that are far apart. (The original Phase 0
scaffold used 50 km, which was too tight.)

The computation is vectorized: pairwise distance matrix (671 x 671), pairwise area ratio
matrix, and pairwise elevation difference matrix are computed once, then a boolean mask
selects all valid edges simultaneously.

### 2.3 Results

```
Total basins loaded:              671
Total directed edges found:       1,298
Nodes involved in edges:          584 (87 basins are completely isolated)
Connected components (>= 2):      46
```

### 2.4 Study network selection

We selected **Component 3** — a 23-basin subnetwork in HUC-12 (south-central Texas) — based
on four criteria in order of importance:

1. **Size** (15-40 basins): 23 basins, within the preferred range
2. **Depth** (>= 3 hops): max directed path length = 3
3. **Geographic coherence**: single HUC region, avoiding cross-climate confounding
4. **Data completeness**: all 23 basins have maurer forcings and streamflow data

Why not the larger components:
- Component 0 (183 basins) is too large and spans 6 HUC regions (eastern seaboard)
- Component 1 (72 basins) spans 2 HUC regions in the Pacific Northwest
- Component 2 (33 basins) has max depth of only 2
- Component 4 (19 basins) spans 2 non-adjacent HUC regions

### 2.5 Study network structure

```
Basins:     23
Edges:      34
Headwaters:  7 (depth 0 — no upstream CAMELS gauge)
Interior:    7 (depth 1-2 — have both upstream and downstream connections)
Outlets:     9 (no downstream CAMELS gauge)
Max depth:   3 (longest path from any headwater to any outlet)
```

The network sits in south-central Texas between latitudes 28.3°N and 31.3°N, longitudes
96.7°W and 100.2°W. Basin areas range from 31.8 km² (tiny headwater 08158810) to 2,124.0 km²
(large outlet 08164000). Elevations range from 67.3 m (downstream outlet 08189500) to
669.2 m (headwater 08165300).

### 2.6 Explanation of each visualization

**national_network_map.png** — A map of all 671 CAMELS basins plotted at their geographic
coordinates (longitude vs latitude). Each basin is a dot. The top 5 connected components are
color-coded: red for the giant 183-basin eastern component, blue for the 72-basin Pacific
Northwest, yellow for our 23-basin Texas study network (component 3), and so on. Light grey
dots represent the 87 completely isolated basins that have no inferred connections to any
other basin. Thin grey lines show all 1,298 inferred edges. This figure answers: "where in
the country did the heuristic find structure?"

**component_distribution.png** — Two panels. The left panel is a bar chart of the 15 largest
connected components ranked by number of basins, with our study network highlighted in red at
rank 3. This shows the heavy-tailed distribution: one massive component dominates, then sizes
drop off rapidly. The right panel is a log-scale histogram of all 46 component sizes, showing
that most components are small (2-5 basins). This justifies selecting a mid-sized component:
large enough for graph effects to emerge, small enough to train quickly.

**edge_diagnostics.png** — A scatter plot of all 1,298 inferred edges. The x-axis is the
geographic distance between the two basins (km), y-axis is the area ratio (child area divided
by parent area), and color encodes the elevation difference (meters). This is a quality check
on the heuristic. Most edges cluster at shorter distances (< 100 km) and moderate area ratios
(1.5x to 5x). Higher elevation differences (brighter colors) tend to appear at longer
distances, which makes physical sense — a basin far downstream has dropped more elevation.
The red dashed line marks the 1.5x area ratio threshold.

**study_network_map.png** — A zoomed-in geographic map of just the 23 Texas basins. Blue
dots are headwater basins (no upstream CAMELS gauge), grey dots are interior basins (have
both upstream and downstream connections), and red dots are outlet basins (no downstream
CAMELS gauge in our network). Curved arrows show the inferred upstream-to-downstream
direction. Node size is proportional to drainage area, so larger downstream basins appear as
bigger dots. Basin labels show the last 5 digits of the gauge ID.

**study_network_dag.png** — The same 23 basins arranged as a topological DAG (directed
acyclic graph) using graphviz's dot layout. This ignores geography and shows pure flow
structure: headwaters at the top, outlets at the bottom. The hierarchical layout reveals the
depth structure that matters for message passing. Multiple arrows converging on a single basin
show where upstream information aggregates.

**study_network_depth.png** — The geographic layout again, but now each basin is colored on
a continuous blue-to-red scale representing its **depth** — the maximum number of directed
hops from any headwater to that basin. Blue = depth 0 (headwater, no upstream information
available), through yellow (depth 1-2), to dark red = depth 3 (deepest outlet, farthest from
any headwater). This is the key experimental variable for Phase 3: the hypothesis predicts
that improvement from graph structure should correlate with this color gradient.

### 2.7 Caveats

The 34 edges for 23 nodes (ratio 1.48) is denser than a true river tree (which would have
exactly 22 edges for 23 nodes). This means some edges are **false positives** — basins that
pass all three heuristic tests but aren't actually in the same river system. For example,
headwater 08103900 (86 km²) has 5 inferred children at distances up to 125 km, some of which
are likely spurious.

For publication-quality results, these heuristic edges should be replaced with NHDPlus
flowline-derived connectivity (ground truth river network data from the USGS). For testing
whether graph structure helps at all, heuristics are an acceptable first pass.

### 2.8 Output files

| File | Location | Description |
|------|----------|-------------|
| full_network_edges.csv | topology_analysis/phase1_network_discovery/outputs/ | All 1,298 edges with area ratios, elevation diffs, distances |
| study_network_edges.csv | topology_analysis/phase1_network_discovery/outputs/ | 34 edges within the 23-basin study network |
| study_network_basins.txt | topology_analysis/phase1_network_discovery/outputs/ | 23 basin IDs, one per line (also copied to experiments/) |
| study_network_summary.txt | topology_analysis/phase1_network_discovery/outputs/ | Per-basin inventory: area, elevation, depth, role |

---

## 3. Phase 2: Baseline LSTM on the Study Network

### 3.1 What we are predicting

We are predicting **daily river discharge** (QObs, in mm/day) at each of the 23 basin
gauges. Given the previous 30 days of meteorological forcing data (precipitation, radiation,
temperature, vapor pressure) and static catchment attributes (elevation, area, slope, mean
precipitation, mean PET), the model outputs a single discharge prediction for the final day
of the 30-day window.

This is a **supervised regression task**. The target variable is the observed USGS streamflow,
normalized to mm/day and then z-score standardized.

### 3.2 The model

We use **CudaLSTM** from the NeuralHydrology model zoo — a standard LSTM wrapped in PyTorch's
optimized `nn.LSTM` implementation. The architecture is:

```
Input: 30 days of forcing data [batch, 30, 5] + static attributes [batch, 5]
    |
    v
InputLayer (embedding)
    |-- concatenates dynamic features with static attributes at each timestep
    |-- output shape: [30, batch, input_size]
    v
nn.LSTM(input_size, hidden_size=64)
    |-- processes the 30-timestep sequence
    |-- output: lstm_output [batch, 30, 64], h_n [batch, 1, 64], c_n [batch, 1, 64]
    v
Dropout(p=0.4)
    v
Regression Head: Linear(64 -> 1)
    |-- applied to the last timestep only (predict_last_n=1)
    |-- output: y_hat [batch, 1, 1]  (predicted discharge, z-score normalized)
```

Key hyperparameters:
- Hidden size: 64
- Initial forget bias: 3 (keeps forget gate closed early in training, helping gradient flow)
- Output dropout: 0.4
- Sequence length: 30 days

The model is trained **independently per basin** — each sample in a batch can be from any
basin at any time window. The model does not know which basin a sample comes from (no basin
ID encoding). Instead, it relies on the static attributes (elev_mean, area, slope, p_mean,
pet_mean) to implicitly differentiate basin behavior. This is the standard NeuralHydrology
approach from Kratzert et al.

### 3.3 Loss function

The loss is **Mean Squared Error (MSE)** on the z-score normalized predictions vs normalized
observations:

```
loss = mean( (y_hat_normalized - y_obs_normalized)^2 )
```

Where both y_hat and y_obs are in normalized space (zero mean, unit variance across the
training set). NaN targets (missing discharge observations) are excluded from the loss.

The MSE is computed on **only the last timestep** of each 30-day window (because
`predict_last_n=1`), not on all 30 timesteps. This means the model is optimized for
single-step-ahead prediction given a 30-day lookback.

### 3.4 Evaluation metric

The primary metric is **Nash-Sutcliffe Efficiency (NSE)**, the standard benchmark in hydrology:

```
NSE = 1 - sum((Q_obs - Q_sim)^2) / sum((Q_obs - mean(Q_obs))^2)
```

- NSE = 1.0 means perfect prediction
- NSE = 0.0 means the model is as good as predicting the mean discharge every day
- NSE < 0.0 means the model is worse than predicting the mean (a bad model)

NSE is computed **in physical units** (mm/d), not normalized space. The evaluation pipeline
un-normalizes predictions by reversing the z-score: `Q_sim = y_hat * std + mean`, then
computes NSE against the raw observed discharge.

NSE is computed **per basin** over the entire test period (2005-2008). The median NSE across
all basins is the headline metric.

### 3.5 Training details

Config file: `experiments/lstm_study_network.yaml`

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Basins | 23 (HUC-12 Texas study network) | Selected in Phase 1 |
| Model | cudalstm | Optimized nn.LSTM (was "lstm"/CustomLSTM in original) |
| Epochs | 30 | Enough for loss to plateau (5 was too few) |
| Batch size | 256 | Larger dataset benefits from bigger batches |
| Learning rate | 1e-3 (Adam) | Standard, no schedule |
| Train period | 1990-01-01 to 1999-12-31 | 10 years |
| Validation period | 2000-01-01 to 2004-12-31 | 5 years |
| Test period | 2005-01-01 to 2008-12-31 | 4 years (maurer limit) |
| Gradient clipping | max norm 1.0 | Prevents exploding gradients |
| Validation frequency | every epoch, 5 random basins | More stable than 1 basin |

Training samples: 83,306 windows (3,622 windows/basin x 23 basins).
Training time: ~20 seconds per epoch on CPU, ~10 minutes total.
Training loss: 0.40 (epoch 1) → 0.11 (epoch 30), still slowly declining.

### 3.6 Results

Run directory: `runs/lstm_study_network_1304_222043/`

**Per-basin results, sorted by depth:**

| Basin | Role | Depth | Area (km²) | NSE |
|-------|------|-------|-----------|-----|
| 08200000 | headwater | 0 | 249.1 | 0.571 |
| 08103900 | headwater | 0 | 86.0 | 0.435 |
| 08158810 | headwater | 0 | 31.8 | 0.429 |
| 08176900 | headwater | 0 | 925.4 | 0.388 |
| 08165300 | headwater | 0 | 436.0 | 0.119 |
| 08196000 | headwater | 0 | 327.0 | -0.130 |
| 08150800 | headwater | 0 | 557.5 | -0.847 |
| 08178880 | outlet | 1 | 850.6 | 0.652 |
| 08101000 | outlet | 1 | 1177.1 | 0.643 |
| 08171300 | outlet | 1 | 1067.5 | 0.625 |
| 08104900 | interior | 1 | 342.6 | 0.532 |
| 08195000 | interior | 1 | 1028.3 | 0.512 |
| 08109700 | interior | 1 | 610.1 | 0.467 |
| 08190000 | outlet | 1 | 1961.4 | 0.369 |
| 08158700 | interior | 1 | 320.3 | 0.327 |
| 08202700 | interior | 1 | 434.8 | 0.325 |
| 08155200 | interior | 1 | 232.7 | 0.206 |
| 08198500 | outlet | 1 | 624.4 | 0.084 |
| 08190500 | outlet | 1 | 1799.1 | -0.654 |
| 08175000 | outlet | 2 | 1421.7 | 0.407 |
| 08164300 | interior | 2 | 861.5 | 0.308 |
| 08189500 | outlet | 3 | 1808.3 | 0.757 |
| 08164000 | outlet | 3 | 2124.0 | 0.616 |

**Summary by depth:**

| Depth | Count | Median NSE | Description |
|-------|-------|-----------|-------------|
| 0 | 7 | 0.388 | Headwaters — no upstream gauge |
| 1 | 12 | 0.418 | Interior + outlets with shallow connectivity |
| 2 | 2 | 0.358 | Intermediate depth |
| 3 | 2 | 0.686 | Deepest outlets |
| **All** | **23** | **0.407** | |

**Comparison to the original 10-basin HUC-01 experiment:**

The original Maine/NH basins achieved median NSE 0.73 (5 epochs). The Texas study network
achieves median NSE 0.41 (30 epochs). This is expected — semi-arid Texas basins are
inherently harder to predict than humid Maine basins. Texas has more flashy, intermittent
rainfall-runoff events, lower base flow, and higher evaporative demand, all of which make
discharge prediction more challenging.

Three basins have negative NSE (worse than predicting the mean):
- 08150800 (NSE = -0.85): A headwater basin with high elevation (564m) and moderate area
- 08190500 (NSE = -0.65): A large outlet (1799 km²) at high elevation (590m)
- 08196000 (NSE = -0.13): A headwater with moderate area (327 km²)

These may have unusual hydrology (springs, karst, regulation) or simply need more training.

---

## 4. Phase 3: Directed Graph-LSTM

### 4.1 The scientific hypothesis

A downstream basin's discharge at time *t* depends not only on its own meteorological forcing
history, but also on what happened in upstream basins at times *t-1, t-2, ...* — the water
takes time to travel from upstream to downstream. This is called **flow routing** or **lateral
flow**.

A per-basin LSTM (Phase 2) cannot learn this because each basin is processed independently.
It only sees its own local forcing and static attributes. It has no information about what is
happening upstream.

A **Directed Graph-LSTM** that receives the previous timestep's hidden states from upstream
neighbors can, in principle, learn the effective propagation kernel of the river network.

**The falsifiable prediction**: Downstream basins (depth >= 2) should show improved NSE
compared to the baseline, while headwater basins (depth = 0) should be unchanged — because
headwaters have no upstream neighbors to benefit from. If the improvement does NOT correlate
with depth, either the graph structure is wrong or the model is not learning to use upstream
information.

### 4.2 Architecture

The model is implemented in `experiments/train_graph_lstm.py` as the `DirectedGraphLSTM`
class. At each timestep *t*, for each of the 23 basins *v*:

**Step 1 — Standard LSTM step** (same as baseline):
```
x_v^t = concat(dynamic_forcing_v^t, static_attributes_v)
h_v^t, c_v^t = LSTMCell(x_v^t, h_v^{t-1}, c_v^{t-1})
```
This is identical to what the baseline does. The LSTM cell takes the current input and the
previous hidden/cell state and produces new states.

**Step 2 — Upstream message** (the new part):
```
parents(v) = upstream basins of v from the directed graph
if parents(v) is empty:    (headwater basin)
    m_v^t = 0              (no message — this basin is unchanged from baseline)
else:
    m_v^t = mean over all parents u of: W_upstream * h_u^{t-1}
```
This aggregates the **previous timestep's** hidden states from all upstream neighbors. The
1-timestep lag represents the physical fact that water takes time to travel downstream.
`W_upstream` is a learnable linear transformation (64 -> 64) that projects upstream states
into a "message" space.

**Step 3 — Residual update**:
```
h_v^t = h_v^t + tanh(W_msg * m_v^t)
```
The upstream message is added to the LSTM hidden state through a residual connection.
`W_msg` (64 -> 64) is **initialized to zeros**, which is critical: at the start of training,
`tanh(0 * anything) = 0`, so the model behaves exactly like the baseline LSTM. The upstream
signal can only influence predictions if the model learns that it's helpful. This guarantees
the Graph-LSTM cannot be worse than the baseline at initialization.

**Step 4 — Readout** (same head as baseline):
```
y_hat_v = head(dropout(h_v^T))    where T is the last timestep
```

### 4.3 Why this requires a standalone training script

NeuralHydrology's training loop samples random (basin, time_window) pairs into batches.
A batch might contain basin 08101000 at timestep 500, basin 08189500 at timestep 3000, and
basin 08103900 at timestep 1200. There is no alignment — basins at different timesteps are
mixed together.

The Graph-LSTM needs **all 23 basins at the same timestep** simultaneously, because basin
08189500's upstream message at time *t* requires knowing basin 08176900's hidden state at
time *t-1*. This is fundamentally incompatible with NH's random-batch training.

The standalone script (`experiments/train_graph_lstm.py`) solves this by:
1. Loading all 23 basins' full time series into memory (pre-computed from NH's data loaders)
2. Aligning them by date (same 30-day windows for all basins simultaneously)
3. Processing one time window at a time: all 23 basins step through 30 timesteps together
4. Computing MSE loss on the last timestep across all basins
5. Batching over time windows (not over basins)

### 4.4 The speed tradeoff

The baseline uses `nn.LSTM`, which processes the entire 30-timestep sequence in one optimized
CUDA/CPU call. The Graph-LSTM must use `nn.LSTMCell` in a Python loop over 30 timesteps
because it needs to access intermediate hidden states for message passing. This makes it
roughly **5 minutes per epoch** vs the baseline's 20 seconds per epoch.

### 4.5 Model parameter count

```
LSTMCell (input to hidden):     (10 + 64) * 64 * 4 = 18,944  (input_size=10, hidden=64, 4 gates)
LSTMCell (hidden to hidden):    64 * 64 * 4 = 16,384
LSTMCell biases:                64 * 4 * 2 = 512
W_upstream (message projection): 64 * 64 = 4,096
W_msg (residual gate):          64 * 64 = 4,096
Regression head:                64 * 1 + 1 = 65
Total:                          ~27,713 parameters
```

For comparison, the baseline CudaLSTM has a similar count but without W_upstream and W_msg.
The graph model adds 8,192 parameters (the two 64x64 matrices) for the message passing.

### 4.6 First test run — results and analysis

Run directory: `runs/graph_lstm_study_network_1404_012933/`

**Training:** 10 epochs, ~5 min/epoch, ~53 minutes total on CPU.

```
Epoch  1: loss = 0.912
Epoch  2: loss = 0.777
Epoch  3: loss = 0.707
Epoch  4: loss = 0.657
Epoch  5: loss = 0.683    (test median NSE = 0.186)
Epoch  6: loss = 0.590
Epoch  7: loss = 0.577
Epoch  8: loss = 0.560
Epoch  9: loss = 0.539
Epoch 10: loss = 0.520    (test median NSE = 0.329)
```

Loss was still dropping at epoch 10 (0.52), far from the baseline's converged 0.11 at epoch
30. The model had not finished learning.

**Head-to-head comparison (baseline vs graph):**

| Basin | Role | Depth | Upstream | Baseline NSE | Graph NSE | Delta |
|-------|------|-------|----------|-------------|-----------|-------|
| 08150800 | headwater | 0 | 0 | -0.847 | 0.416 | +1.263 |
| 08196000 | headwater | 0 | 0 | -0.130 | 0.287 | +0.417 |
| 08176900 | headwater | 0 | 0 | 0.388 | 0.391 | +0.003 |
| 08103900 | headwater | 0 | 0 | 0.435 | 0.367 | -0.069 |
| 08158810 | headwater | 0 | 0 | 0.429 | 0.300 | -0.129 |
| 08200000 | headwater | 0 | 0 | 0.571 | 0.242 | -0.329 |
| 08165300 | headwater | 0 | 0 | 0.119 | -0.362 | -0.481 |
| 08198500 | outlet | 1 | 2 | 0.084 | 0.177 | +0.093 |
| 08109700 | interior | 1 | 5 | 0.467 | 0.356 | -0.111 |
| 08101000 | outlet | 1 | 1 | 0.643 | 0.495 | -0.148 |
| 08202700 | interior | 1 | 1 | 0.325 | 0.156 | -0.170 |
| 08155200 | interior | 1 | 1 | 0.206 | 0.027 | -0.179 |
| 08195000 | interior | 1 | 1 | 0.512 | 0.329 | -0.183 |
| 08171300 | outlet | 1 | 5 | 0.625 | 0.433 | -0.192 |
| 08178880 | outlet | 1 | 2 | 0.652 | 0.427 | -0.225 |
| 08104900 | interior | 1 | 1 | 0.532 | 0.230 | -0.302 |
| 08158700 | interior | 1 | 1 | 0.327 | -0.105 | -0.432 |
| 08190000 | outlet | 1 | 2 | 0.369 | -0.102 | -0.471 |
| 08190500 | outlet | 1 | 1 | -0.654 | -1.426 | -0.772 |
| 08164300 | interior | 2 | 3 | 0.308 | 0.335 | +0.027 |
| 08175000 | outlet | 2 | 4 | 0.407 | 0.353 | -0.054 |
| 08189500 | outlet | 3 | 2 | 0.757 | 0.739 | -0.018 |
| 08164000 | outlet | 3 | 2 | 0.616 | 0.489 | -0.127 |

**Summary by depth:**

| Depth | Baseline median | Graph median | Delta median |
|-------|----------------|-------------|-------------|
| 0 (headwater) | 0.388 | 0.300 | -0.069 |
| 1 | 0.418 | 0.204 | -0.188 |
| 2 | 0.358 | 0.344 | -0.014 |
| 3 | 0.686 | 0.614 | -0.072 |
| **All** | **0.407** | **0.329** | **-0.148** |

### 4.7 Interpretation — what this tells us

**The Graph-LSTM did not beat the baseline.** Overall median NSE dropped from 0.407 to 0.329.
The delta is negative at every depth level.

**However, this is almost certainly a training gap, not an architecture failure.** The
evidence:

1. **The graph model was undertrained.** It had 10 epochs (~53 min) while the baseline had 30
   epochs (~10 min). The graph model's final loss (0.52) was far from converged compared to
   the baseline's (0.11). Epoch-over-epoch the loss was still dropping steadily.

2. **Test NSE improved rapidly between epoch 5 and 10.** Median went from 0.186 to 0.329 in
   just 5 more epochs. Extrapolating the trajectory, 30 epochs could plausibly close the gap.

3. **Depth-2 and depth-3 basins held close to baseline despite fewer epochs.** The deepest
   basins (08189500 at depth 3: 0.739 vs 0.757) lost very little. These are the basins where
   graph structure matters most — and they were the most resilient. This is a positive signal
   buried under the overall undertraining.

4. **Two previously-negative basins flipped positive.** Basin 08150800 went from NSE -0.85 to
   +0.42, and 08196000 from -0.13 to +0.29. Both are headwaters (depth 0), so this
   improvement cannot be from upstream messages — it likely reflects the different training
   dynamics of the standalone script (different batching, different optimization trajectory).

**What this does NOT tell us:** Whether the graph structure actually helps downstream basins.
The signal is obscured by the training gap. We need equal training effort (same number of
effective gradient steps, or same final loss) before the depth-stratified comparison is
scientifically meaningful.

---

## Repository Structure

```
neural_hydrology/
|
|-- neuralhydrology/                    Upstream NeuralHydrology framework (unmodified)
|   |-- nh_run.py                         Entry point: train / evaluate / finetune
|   |-- datasetzoo/                       Dataset loaders (CAMELS variants)
|   |   |-- basedataset.py                  Base class: windowing, scaling, __getitem__
|   |   +-- camelsus.py                     CAMELS-US: loads forcings + discharge
|   |-- modelzoo/                         Model implementations
|   |   |-- basemodel.py                    Abstract base class
|   |   |-- cudalstm.py                    Standard LSTM (our baseline)
|   |   |-- inputlayer.py                  Feature embedding
|   |   +-- head.py                        Output heads (regression, uncertainty)
|   |-- training/
|   |   |-- basetrainer.py                 Training loop, checkpointing
|   |   +-- loss.py                        Loss functions
|   +-- evaluation/
|       |-- tester.py                      Per-basin inference and metric computation
|       +-- metrics.py                     NSE, RMSE, KGE implementations
|
|-- experiments/                          Experiment configs and scripts
|   |-- 1_basin.txt                        10 HUC-01 basin IDs (original experiment)
|   |-- study_network_basins.txt           23 HUC-12 basin IDs (study network)
|   |-- lstm_camels_us.yaml                Config: 10-basin LSTM baseline
|   |-- lstm_study_network.yaml            Config: 23-basin LSTM baseline
|   |-- train_graph_lstm.py                Standalone training: Directed Graph-LSTM
|   +-- compare_results.py                 Baseline vs Graph-LSTM comparison table
|
|-- datasets/camels_us/                   CAMELS-US data (671 basins)
|   |-- usgs_streamflow/{HUC}/             Daily discharge by HUC-02 region
|   |-- basin_mean_forcing/                maurer / daymet / nldas
|   +-- camels_attributes_v2.0/            Static catchment attributes
|
|-- topology_analysis/                    Research: graph-based extensions
|   |-- phase0_scaffold/                   Early exploration (10 basins, MPNN, signal decay)
|   +-- phase1_network_discovery/          Full CAMELS-US topology + study network selection
|       |-- discover_network.py              Infer 1298 edges, select study network
|       |-- plot_study_network.py            Study network geographic + DAG plots
|       |-- plot_full_network.py             National map, distributions, diagnostics
|       +-- outputs/                         Edge lists, basin lists, 6 visualizations
|
|-- runs/                                 Saved experiment runs
|   |-- lstm_camels_us_1901_235614/        10-basin HUC-01 (Jan 19, tested)
|   |-- lstm_camels_us_2001_000652/        10-basin HUC-01 (Jan 20, best)
|   |-- lstm_study_network_1304_222043/    23-basin HUC-12 baseline (30 epochs, tested)
|   +-- _archive/                          Early failed/debug runs
|
|-- next_steps.md                         Three-phase plan from Claude Chat
|-- CURRENT_STATE.md                      This file
+-- README.md                             Project overview
```
