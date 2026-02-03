# Topology Analysis for NeuralHydrology

This folder contains a self‑contained, readable implementation of the **topology analysis pipeline** you built on top of NeuralHydrology. It extracts a small CAMELS basin subset, defines a graph over basins, equips that graph with node states, and then studies how information would flow (or fail to flow) through this structure under a simple Message Passing Neural Network (MPNN).

The code is deliberately **geometry‑free** (no curvature, distances, or rewiring algorithms) and focuses purely on **topology + message passing** as an information‑flow constraint.

---

## 1. Basin graph and experimental object

**Goal**: Define the minimal object you analyze everywhere else: a directed basin graph with time‑indexed node states.

- **Node set**: 10 CAMELS basins in Maine / New Hampshire (HUC 01), taken from `experiments/1_basin.txt`.
- **Node IDs**: USGS gauge IDs
  - `01013500, 01022500, 01030500, 01031500, 01047000,
     01052500, 01054200, 01055000, 01057000, 01073000`
- **Node state space**:
  - Each node carries a state vector \(h_v^{(t)} \in \mathbb{R}^{64}\) at each daily time step \(t\).
  - The dimension 64 mirrors the LSTM hidden size in your NeuralHydrology baseline.
  - For the analysis here, states are either abstract feature vectors or random latent vectors; the **structure**, not the origin, matters.
- **Time indexing**:
  - Daily time steps, shared across all basins.
  - In practice, the MPNN experiments fix a **single snapshot** \(H \in \mathbb{R}^{N \times 64}\) for all basins and study how a small perturbation at one node is transported.

**Key file**:
- `basin_graph.py`
  - `BasinGraph`: directed graph over basins (`nodes`, `edges`, `get_parents`, `get_children`, etc.).
  - `NodeStateMatrix`: convenience container for time‑indexed node states (used conceptually; Phase 4 uses simple tensors instead).

Reference document:
- `experimental_object_definition.md` — the original, more formal write‑up of this object.

---

## 2. Inferring edges from CAMELS attributes

**Goal**: Use CAMELS metadata to infer potential upstream → downstream relationships between basins.

**Script**: `infer_topology.py`

- Loads **topographic attributes** from `datasets/camels_us/camels_attributes_v2.0/camels_topo.txt` for the 10 basins.
- For each basin pair (parent, child), an edge `parent → child` is added only if all three heuristics hold:
  1. **Area accumulation**: `area(child) ≥ area(parent) * area_ratio_threshold`
     - Default: `area_ratio_threshold = 1.1` (child at least 10% larger).
  2. **Elevation gradient**: `elev(child) < elev(parent)` (downstream should be lower).
  3. **Spatial proximity**: Haversine distance between centroids ≤ `max_distance_km`
     - Default: `max_distance_km = 50 km`.

For the Maine/NH subset, **no pair of basins passes all three tests**, so the inferred edge set is empty:

- Output: `basin_topology.txt`
  - Contains a header but **no edges** (`E = ∅`).
  - Represents 10 isolated nodes — a fully disconnected graph.

This is important: it means **the real CAMELS subset you’re using has no topological connectivity** under these heuristics. Everything downstream treats this as the “no‑edges” baseline.

---

## 3. Structural analysis of the basin graph

**Goal**: Characterize the graph structure itself — connected components, path lengths, and bottlenecks — *before* adding any neural model.

**Script**: `graph_analysis.py`

- Consumes a `BasinGraph`:
  - Either built from `basin_topology.txt` (if present)
  - Or, by default, an empty‑edge graph over the 10 basins.
- Computes:
  - **Weakly connected components** (ignoring direction).
  - **Roots & leaves** in each component (`is_root`, `is_leaf`).
  - **Path metrics**:
    - Maximum path length in each component.
    - Average path length (over all root→leaf paths).
  - **Node degrees**: in‑degree, out‑degree for each basin.
  - **Qualitative betweenness**: how many shortest paths pass through each node.
- Identifies **bottleneck candidates** as nodes with high in/out degree or high path count.

For the Maine/NH graph:
- Nodes: 10
- Edges: 0
- Components: 10 (each basin isolated)
- Max depth: 0
- In‑degree = out‑degree = 0 for all nodes
- No structural bottlenecks (no paths exist).

This baseline tells you: **if the graph is truly disconnected, no message‑passing model can move information between basins**, regardless of architecture.

---

## 4. Minimal MPNN for information flow

**Goal**: Implement a very simple, invariant MPNN to study how information would propagate *if* edges existed.

**File**: `mpnn.py`

- `MPNNLayer` (single message‑passing step):
  - **Message**: \(m_{ij} = \phi_m(h_i, h_j)\)
    - Input: concatenated states `[h_i, h_j]`.
    - MLP: `Linear(2d → hidden) → ReLU → Linear(hidden → d_msg)`.
  - **Aggregation**: \(m_i = \sum_{j ∈ N(i)} m_{ij}\)
    - For isolated nodes `N(i) = ∅`: `m_i = 0`.
  - **Update**: \(h'_i = \phi_h(h_i, m_i)\)
    - MLP on `[h_i, m_i]` returning a new state in \(\mathbb{R}^{64}\).
- `MPNNModel` (1–2 layers):
  - Simply stacks 1 or 2 `MPNNLayer`s.
  - No equivariance, geometry, or rewiring; pure sum‑aggregation MPNN.

**Verification**: `mpnn_verification.py`

- **Disconnected graph test**:
  - 10 nodes, 0 edges.
  - Checks shape stability and that outputs are finite and non‑zero.
- **Synthetic edge test**:
  - Adds a couple of synthetic upstream→downstream edges.
  - Confirms that downstream nodes change when edges are present.
- **1‑ vs 2‑layer model**:
  - Verifies both configurations run and preserve shapes.

This gives you a **trusted propagation operator** to plug into later experiments.

---

## 5. Hop distance utilities and synthetic graphs

**Files**:
- `hop_distance.py`
  - `compute_hop_distances(graph, source)` — BFS over directed edges, returning hop distance \(d(s, i)\) for all nodes.
  - `group_nodes_by_hop_distance(distances)` — groups node IDs by hop distance.
- `synthetic_graphs.py`
  - `create_chain_graph(node_ids, length)`
    - Builds a simple chain: `v0 → v1 → … → vL`.
    - Ideal for clean hop‑distance experiments.
  - `create_tree_graph(node_ids, branching_factor=2, depth=2)`
    - Builds a small tree using the same node IDs.
    - Useful for seeing aggregation/compression at merge points.

These utilities are used only for **diagnostic synthetic topologies**, never to modify the real CAMELS topology on disk.

---

## 6. Signal‑decay / bottleneck experiment

**Goal**: Empirically show how a small perturbation at one basin is attenuated as it propagates through the graph under the MPNN — the basic symptom of **over‑squashing**.

**Core logic**: `signal_decay.py`

1. **Setup**
   - Fix random seed (PyTorch + NumPy) and perturbation size `epsilon` (e.g. 0.05).
   - Create initial node states `H ∈ ℝ^{N×64}` (randomly, but reproducible).
2. **Perturbation protocol**
   - Choose a **source node** `s` (first basin ID).
   - Construct `H⁺` by adding `ε u` to `h_s`, where `u` is a fixed unit vector.
   - Run the same `MPNNModel` (1‑ or 2‑layer) on both:
     - `Z = f_MPNN(H)`
     - `Z⁺ = f_MPNN(H⁺)`
   - Define per‑node signal metric: \(\Delta_i = \lVert Z_i^{(+)} - Z_i \rVert_2\).
3. **Hop‑distance analysis**
   - For each node \(i\), compute hop distance \(d(s, i)\) in the graph (shortest directed path length from the source) using `compute_hop_distances`.
   - Group nodes by hop distance and summarize Δ per hop (mean/median/max).

**Runner**: `run_signal_decay.py`

- Creates a timestamped output directory under `signal_decay_outputs/`.
- Runs two regimes for each of 1‑ and 2‑layer MPNNs:
  1. **Real graph baseline** (`E = ∅`):
     - Graph: `BasinGraph` with 10 nodes and no edges.
     - Expected: only the source node has non‑zero Δ; all others ≈ 0.
  2. **Synthetic diagnostic graphs** (chain and tree):
     - Chain graph built from the same node IDs.
     - Small tree graph (branching factor 2, depth 2) if enough nodes.
     - Expected: Δ decays with hop distance; 2‑layer extends reach slightly compared to 1‑layer.
- Saves:
  - `metrics.csv` — row per `(regime, graph_type, layers, source_node, target_node)` with `hop` and `delta_norm`.
  - `run_config.json` — seed, epsilon, dimensions, timestamp.
  - `summary.md` — small tables of mean Δ per hop and textual interpretation.
  - `plot.png` — mean Δ vs hop (chain + tree, 1‑ vs 2‑layer).

Existing runs live in:
- `signal_decay_outputs/YYYYMMDD_HHMMSS_phase4/`

These runs show exactly what you expect from sum‑aggregation MPNNs:
- On the **real (disconnected) graph**: perturbations do **not** propagate beyond the source.
- On **synthetic chains/trees**: Δ decays monotonically with hop distance; the tree shows additional compression due to aggregation at branching points.

---

## 7. How to use this folder

### Re‑infer topology

```bash
cd topology_analysis
python infer_topology.py    # writes basin_topology.txt
```

### Re‑run structural analysis

```bash
cd topology_analysis
python graph_analysis.py    # prints and writes graph_analysis_summary.txt
```

### Test / inspect the MPNN

```bash
cd topology_analysis
python mpnn_verification.py
```

### Reproduce signal‑decay experiments

```bash
cd topology_analysis
python run_signal_decay.py
```

New runs will appear under `signal_decay_outputs/` with fresh metrics, summaries, and plots.

---

## 8. Big‑picture interpretation

1. **Topology as a constraint**:
   - The inferred CAMELS basin graph for this subset is **fully disconnected** under reasonable physical heuristics.
   - This means that, as far as a graph model is concerned, each basin is an island.

2. **Message passing as low‑pass filtering**:
   - The minimal MPNN you implemented behaves like a **low‑pass filter over the graph**.
   - On connected synthetic graphs, perturbations decay with hop distance and layer depth, even before any learning or geometry.

3. **Over‑squashing symptoms without geometry**:
   - The signal‑decay experiments already show information bottlenecks purely from topology + sum aggregation + shallow MLPs.
   - Trees exhibit stronger compression at branching points than chains, hinting at where real‑world basin hierarchies could lose upstream detail.

4. **Motivation for future work**:
   - Once real basin topology is better characterized (or recovered from data), you can reuse this code to study:
     - How deep chains and dense confluences create bottlenecks.
     - How rewiring or geometry‑aware constructions might relieve over‑squashing.
   - The current folder gives you a **clean, testable scaffold** for those future experiments.

