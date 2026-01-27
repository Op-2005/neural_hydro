# Phase 4: Signal Decay Analysis Summary

**Run Configuration:**
- Seed: 42
- Epsilon: 0.05
- Source Node: 01013500
- Layers Tested: [1, 2]

## Key Findings

### Regime A: Real Graph Baseline (E=∅)
- Source node Δ: 0.004684
- Other nodes mean Δ: 0.000000
- **Observation**: Perturbations do not propagate (all other nodes ~0)

### Regime B: Synthetic Chain Graph

**1-layer MPNN:**
| Hop | Mean Δ | Median Δ | Max Δ | Count |
|-----|--------|----------|-------|-------|
| 0 | 0.006995 | 0.006995 | 0.006995 | 1 |
| 1 | 0.000999 | 0.000999 | 0.000999 | 1 |
| 2 | 0.000000 | 0.000000 | 0.000000 | 1 |
| 3 | 0.000000 | 0.000000 | 0.000000 | 1 |
| 4 | 0.000000 | 0.000000 | 0.000000 | 1 |

**2-layer MPNN:**
| Hop | Mean Δ | Median Δ | Max Δ | Count |
|-----|--------|----------|-------|-------|
| 0 | 0.002195 | 0.002195 | 0.002195 | 1 |
| 1 | 0.000421 | 0.000421 | 0.000421 | 1 |
| 2 | 0.000033 | 0.000033 | 0.000033 | 1 |
| 3 | 0.000000 | 0.000000 | 0.000000 | 1 |
| 4 | 0.000000 | 0.000000 | 0.000000 | 1 |

### Regime B: Synthetic Tree Graph

**1-layer MPNN:**
| Hop | Mean Δ | Median Δ | Max Δ | Count |
|-----|--------|----------|-------|-------|
| 0 | 0.008651 | 0.008651 | 0.008651 | 1 |
| 1 | 0.001326 | 0.001326 | 0.001345 | 2 |
| 2 | 0.000000 | 0.000000 | 0.000000 | 4 |

**2-layer MPNN:**
| Hop | Mean Δ | Median Δ | Max Δ | Count |
|-----|--------|----------|-------|-------|
| 0 | 0.001350 | 0.001350 | 0.001350 | 1 |
| 1 | 0.000284 | 0.000284 | 0.000298 | 2 |
| 2 | 0.000039 | 0.000041 | 0.000044 | 4 |

## Interpretation

1. **Real graph (E=∅)**: Confirms no propagation - validates Phase 2 + Phase 3 consistency.
2. **Synthetic topologies**: Show signal decay with hop distance, demonstrating propagation limits.
3. **Layer depth**: Compare 1-layer vs 2-layer to assess reach vs stability trade-offs.
