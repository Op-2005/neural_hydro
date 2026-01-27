# Phase 4: Signal Decay Analysis

## Overview

Phase 4 demonstrates signal decay (over-squashing symptoms) through perturbation experiments on the MPNN model. This phase produces empirical evidence of information propagation limits without invoking geometry, curvature, or rewiring.

## How to Run

```bash
python run_phase4.py
```

This will:
1. Run perturbation experiments on both real (disconnected) and synthetic (chain/tree) graphs
2. Test both 1-layer and 2-layer MPNNs
3. Generate outputs in `outputs/YYYYMMDD_HHMMSS_phase4/`

## Output Files

Each run creates a timestamped directory containing:

- **`metrics.csv`**: Detailed results with columns:
  - `regime`: "real" or "synthetic"
  - `graph_type`: "disconnected", "chain", or "tree"
  - `layers`: Number of MPNN layers (1 or 2)
  - `source_node`: Perturbed node ID
  - `target_node`: Node where signal is measured
  - `hop`: Hop distance from source (or "inf" if unreachable)
  - `delta_norm`: Signal metric ||Z(+)_i - Z_i||_2

- **`run_config.json`**: Experiment configuration (seed, epsilon, etc.)

- **`summary.md`**: Analysis summary with key findings and interpretation

- **`plot.png`**: Optional visualization of hop distance vs mean delta

## Expected Behavior

### Success Criteria

1. **Real graph (E=∅)**:
   - Source node shows non-zero Δ (perturbation applied)
   - All other nodes show Δ ≈ 0 (no propagation)
   - Validates Phase 2 + Phase 3 consistency

2. **Synthetic graphs**:
   - Signal decays with hop distance
   - 1-layer vs 2-layer show different decay curves
   - Demonstrates propagation limits and over-squashing

### Key Interpretation Points

- Empty graph confirms no propagation (sanity check)
- Chain/tree graphs show hop-based attenuation
- Layer depth affects reach vs stability trade-off
- Provides concrete artifact motivating bottleneck analysis

## Code Structure

- `signal_decay.py`: Core perturbation experiment logic
- `synthetic_graphs.py`: Chain/tree graph construction
- `hop_distance.py`: BFS-based hop distance computation
- `run_phase4.py`: Main execution script
- `mpnn_layer.py`: MPNN implementation (from Phase 3)

## Protocol Notes

- **Reproducibility**: Fixed random seeds (seed=42)
- **No geometry**: No distances, coordinates, or curvature
- **Diagnostic only**: Synthetic edges are analysis scaffolds, not real topology
- **Artifact-first**: Outputs designed for direct use in deliverables
