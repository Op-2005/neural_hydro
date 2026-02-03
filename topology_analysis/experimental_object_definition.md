# Phase 1: Experimental Object Definition

## Overview

This document formally defines the minimal experimental object for analyzing information-flow bottlenecks in basin graph neural networks. No geometry, curvature, or rewiring mechanisms are introduced at this stage.

---

## 1. Node Set

**Definition**: The node set $V$ is a finite set of basin identifiers.

**Concrete specification**:
- $V = \{01013500, 01022500, 01030500, 01031500, 01047000, 01052500, 01054200, 01055000, 01057000, 01073000\}$
- Each element $v \in V$ is a USGS gauge ID (8-digit string) corresponding to a CAMELS basin in HUC region 01 (Maine/New Hampshire).

**Cardinality**: $|V| = 10$

---

## 2. Edge Set

**Definition**: The edge set $E \subseteq V \times V$ defines a directed graph $G = (V, E)$ where $(u, v) \in E$ indicates that basin $u$ is upstream of basin $v$ (parent → child relationship).

**Concrete specification**:
- Edges are inferred from physical/hydrological relationships using heuristics:
  - Catchment area accumulation: $area(v) \geq area(u) \cdot \theta_{area}$ where $\theta_{area} = 1.1$
  - Elevation gradient: $elev(v) < elev(u)$
  - Spatial proximity: $distance(u, v) \leq d_{max}$ where $d_{max} = 50$ km
- **Current state**: $E = \emptyset$ (no parent-child relationships inferred for this basin subset)
- **Note**: An empty edge set is valid for Phase 1; it represents disconnected basins or basins that do not form a connected river network in this subset.

**Graph structure**: $G = (V, \emptyset)$ is a directed graph with no edges (10 isolated nodes).

---

## 3. Node State Space

**Definition**: At each time step $t$, each node $v \in V$ carries a state vector $h_v^{(t)} \in \mathbb{R}^d$ where $d$ is the state dimension.

**Concrete specification** (Option A - Fixed feature vector):
- $d = 64$ (matches LSTM hidden size from baseline experiment)
- $h_v^{(t)}$ is constructed from:
  - Recent hydrometeorological inputs: precipitation, temperature, potential evapotranspiration (last $\tau = 30$ days, matching `seq_length`)
  - Discharge history: observed streamflow (last $\tau$ days)
  - Static basin attributes: elevation, catchment area, soil properties (time-invariant)
- **Feature construction**: Concatenate normalized time-series features and static attributes, then project to $d$ dimensions via a learned or fixed linear transformation.

**Alternative specification** (Option B - Learned latent representation):
- $d = 64$ (LSTM hidden size)
- $h_v^{(t)}$ is the hidden state $h_n$ from a trained LSTM model (e.g., `CustomLSTM`) at time $t$
- Extracted from: `model.forward(data)['h_n']` where `data` contains the input sequence for basin $v$ up to time $t$

**Phase 1 choice**: **Option A** (fixed feature vector) for explicit control and interpretability.

**State tensor shape**: For a batch of $|V|$ nodes at time $t$: $H^{(t)} \in \mathbb{R}^{|V| \times d}$ where $H^{(t)}[v, :] = h_v^{(t)}$.

---

## 4. Time Indexing

**Definition**: Time is discretized into daily time steps indexed by $t \in \{1, 2, \ldots, T\}$ where $T$ is the total number of days in the observation period.

**Concrete specification**:
- **Temporal resolution**: Daily (1 day per time step)
- **Observation period**: Matches CAMELS dataset temporal coverage (typically 1980-2014 for training, with train/val/test splits)
- **Time alignment**: All nodes $v \in V$ share the same time index $t$ (synchronous updates)
- **Time step semantics**: $t$ represents the end of day $t$; state $h_v^{(t)}$ incorporates information up to and including day $t$

**Notation**: 
- $H^{(t)}$ denotes the state matrix for all nodes at time $t$
- $h_v^{(t)}$ denotes the state vector for node $v$ at time $t$

---

## 5. Summary

**Experimental object**: A directed graph $G = (V, E)$ with:
- **Nodes**: 10 CAMELS basins (Maine/NH, HUC 01)
- **Edges**: Parent → child relationships (currently empty: $E = \emptyset$)
- **Node states**: Time-indexed feature vectors $h_v^{(t)} \in \mathbb{R}^{64}$ constructed from hydrometeorological inputs and static attributes
- **Time**: Daily time steps $t \in \{1, \ldots, T\}$ with synchronous updates across all nodes

**Key properties**:
- No assumptions of global connectivity
- No distance-based or similarity-based edges
- No rewiring mechanisms
- Explicit, concrete state definitions (not abstract)
- Fixed for entire Phase 1 (no learning objectives or geometry yet)

---

## 6. Data Sources

**Basin attributes**: `datasets/camels_us/camels_attributes_v2.0/camels_topo.txt`
- Fields: `gauge_id`, `gauge_lat`, `gauge_lon`, `elev_mean`, `area_gages2`

**Time series data**: 
- Meteorological forcings: `datasets/camels_us/basin_mean_forcing/daymet/`
- Streamflow: `datasets/camels_us/usgs_streamflow/`

**Basin list**: `experiments/1_basin.txt`

---

## 7. Implementation Notes

- Topology inference script: `infer_topology.py`
- Saved topology: `basin_topology.txt` (edge list format: `parent_id child_id`)
- Graph representation: Can be implemented as adjacency list, edge list, or NetworkX `DiGraph`
