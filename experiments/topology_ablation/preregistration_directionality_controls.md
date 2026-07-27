# Pre-registration — Directionality & Topology-Specificity Controls (the Kirschstein mirror test)

**Date:** 2026-07-26. **Author:** /crs-unleashed session.
**Compute:** GPU (Colab T4). Stock cudalstm; only the edge set defining `upstream_q` changes.

## Why this experiment (the gap)

The upstream-flow gain (oracle +0.037) is currently supported by a temporal-shuffle null, an
upstream-precip contrast, a lag sweep, and a depth-gradient. What it is **missing** is the control
that most directly engages the literature it rebuts. Kirschstein & Sun (ICML 2024) diagnosed the
failure of river-network GNNs as **directional insensitivity**: *"GNNs typically yield similar
performance whether the original edge directions are maintained, reversed, or randomly perturbed."*
That insensitivity is *why* topology didn't help them.

If our static upstream-flow feature is **directionally sensitive** — the gain collapses when edges
are reversed — then we exhibit exactly the property whose absence explains the GNN null. This turns
the mechanism claim from correlational ("gain rises with depth") into causal ("gain requires correct
flow direction"), and closes the loop with both cited papers in a single controlled test.

## Design

All conditions are stock cudalstm, seed 11, **observed** discharge (oracle-style, isolates the
edge variable from prediction quality), lag 1, byte-identical config. **Only the edge set changes.**

| Condition | `upstream_q` aggregates | Edge set |
|---|---|---|
| **L** | (none) | — |
| **L+upQ (forward)** | true upstream parents' observed Q | `component0_edges.csv` (existing oracle) |
| **L+upQ_reversed** | true *downstream* children's observed Q | edges reversed (parent↔child) |
| **L+upQ_random** | *randomly chosen* basins' observed Q | degree-preserving random rewire (seed 42) |

- **Reversed** tests **directionality**: is upstream→downstream flow special, or does any adjacent
  basin's flow work? (Direct analog of Kirschstein's "edges reversed".)
- **Random** tests **topology-specificity**: is it the *real river structure*, or any spatial
  aggregation of regional flow? (Analog of Kirschstein's "randomly permuted"; preserves each
  basin's in-degree so only *which* neighbors changes, not *how many*.)
- All Δ computed **paired per-basin vs L on the forward-connected basin set** (the 150 basins with
  true upstream neighbors — where the routing question is even defined), for apples-to-apples.

## Predictions & success criteria

**Directional sensitivity (primary):** `forward Δ − reversed Δ ≥ +0.015`.
**Topology specificity:** `forward Δ − random Δ ≥ +0.015`.

**Honest note on reversed:** reversed is NOT expected to reach exactly 0. Downstream flow is
statistically correlated with the target (shared precipitation; the basin's own water passes
downstream), so reversed may carry a *residual* positive signal. The **contrast** forward > reversed
is the test — not reversed ≈ 0. A large gap = directional routing; a small gap = generic correlation.

## Falsification (this is a real risk, not a rubber stamp)

- If **reversed ≈ forward** (gap < +0.005): the gain is **direction-insensitive** → it is generic
  spatial correlation, **not routing**. This would seriously undermine the routing narrative and
  must be reported as such — the paper's mechanism story would need rewriting, not re-scoping.
- If **random ≈ forward** (gap < +0.005): the gain is **not topology-specific** → any regional flow
  aggregate suffices; the "river network" framing is overstated. Report honestly.

## Discipline

- Pre-registered before execution. Amend only by dated append.
- Observed discharge for all variants (one variable — edges — changes).
- Feature index named `'date'` (NH concatenation requirement; the 52bd535 bug).
- Random rewire uses a fixed RNG seed (42) for reproducibility.
- Single seed (11) for the first pass, per established protocol; multi-seed only if the signal lands
  and we take it to publication. A falsification is reported, not redesigned around.
- Robustness (bundled): report Δ over all 183 basins too, and note the reversed-connected set differs.
