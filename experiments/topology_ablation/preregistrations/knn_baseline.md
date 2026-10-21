# Pre-registration — nearest-neighbour baseline (is the river graph necessary at all?)

**Written before any k-NN run.**

## Why

The distance-preserving control established that the gain does not require the true upstream
topology. The immediate consequence, which the study has never tested, is that the river graph may be
unnecessary scaffolding. If the useful quantity is "recent discharge from nearby gauges," then the
*k* geographically nearest basins should serve as well as the graph-selected ones — with no edge
inference, no area-ratio rule, no elevation rule, and no network at all.

Once the paper is titled around proximity, this is the first baseline a reviewer will demand, and the
paper currently infers proximity only by elimination.

## Design

For each basin, take the *k* nearest **other** basins by great-circle gauge distance and area-weight
their lag-1 discharge exactly as Eq. 4 does. `k ∈ {2, 4}`; `k=4` is chosen to sit near the graph's
mean in-degree of 4.16, so the k-NN and graph inputs average a comparable number of neighbours.
Seeds 11, 13, 17.

Note one structural difference, which is itself a result: the k-NN input is defined for **all 183
basins**, including the 33 headwaters that receive `u_i ≡ 0` under the graph. The verdict cell reports
both the connected-basin contrast (paired against the graph conditions) and the all-basin contrast.

## Hypothesis

k-NN performs close to the graph-derived input on connected basins, because the graph's contribution
is neighbour selection and distance-matched selection already reproduces the gain.

## Pre-registered read-out

Let `G` = cross-seed mean Δ for `L_upQ` (graph, observed) and `N` = the best cross-seed mean Δ across
`k ∈ {2,4}`, both on the forward-connected basins.

- **Graph is scaffolding** (expected): `N ≥ G − 0.005`. Report k-NN as the primary baseline and state
  that the network is a convenience for enumerating neighbours, not a requirement. This simplifies the
  paper's claim and makes it transferable to regions with no inferred hydrography.
- **Graph adds something** (`N < G − 0.005`): upstream selection beats plain geography. This would be
  the first positive evidence for topology anywhere in the study and must be reported as such — it
  would partially rehabilitate the network and require softening the title.
- **Headwater probe:** if k-NN produces a detectable gain at the 33 headwaters, that confirms the
  paper's "gain vanishes at headwaters" observation is definitional (`u_i ≡ 0`) rather than physical,
  which the paper currently concedes only in the appendix.

## Interpretation limits

k-NN neighbours are unconstrained in area and elevation, so some will be much smaller or upstream-ish
by accident. This makes the comparison conservative for the graph, not for k-NN.

## Cost

6 runs (2 values of k x 3 seeds) x ~40 min on a T4 (~4 GPU-hours).
