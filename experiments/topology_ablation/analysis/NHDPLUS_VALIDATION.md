# NHDPlus validation of the inferred edge set

**New for the camera-ready (2026-10-21).** Requested by Reviewer BaCR: *"Checking edges against
NHDPlus (e.g., via USGS NLDI) would fix this."*

Regenerate: `python experiments/topology_ablation/validate_edges_nhdplus.py <edges.csv> <basins.txt>`
Saved query result: `nldi_upstream.json` (per child gauge, the upstream NWIS sites NLDI returns).

## Method

For each of the 150 connected basins, query the USGS NLDI upstream-main navigation
(`/nldi/linked-data/nwissite/USGS-<id>/navigation/UT/nwissite?distance=1000`) and ask whether each
inferred parent appears in that child's genuine upstream network.

## Result

| Quantity | Value |
|---|---|
| Child gauges queried | 150 / 150 (0 failures) |
| Edges checkable | 624 / 624 |
| Edges confirmed genuinely upstream in NHDPlus | **11 (1.8%)** |
| Connected basins with >=1 true upstream study gauge | **14 / 150** |
| True NHDPlus pairs among study basins recovered by the heuristic | 11 of 15 |
| Mean separation: confirmed edges | **31.9 km** |
| Mean separation: all inferred edges | 91.6 km |

Sanity check: every query returned the queried gauge itself, confirming correct indexing.

## Reading

The heuristic edge rule recovers almost no real hydrological connectivity: fewer than 2% of its
edges are genuine upstream relations, and only 14 of 150 connected basins have any true upstream
gauge within the study set. CAMELS basins are largely non-nested, exactly as the reviewer suspected.

**This strengthens the paper's conclusion rather than weakening it.** The distance-matched
substitution control removed a relation that was rarely real connectivity to begin with, so the
observed gain never could have come from routing. The measurement now states that directly instead
of leaving it as an acknowledged unknown.

It also explains the headline geometry: genuine upstream pairs average 31.9 km, far closer than the
inferred set's 91.6 km and comparable to the nearest-gauge selector's 46.7 km. Real connectivity in
this dataset *is* mostly local, which is why a plain distance rule captures it better than the
area/elevation rule does.

**Scope.** This bounds what our edge rule recovers. It does not re-run any condition on surveyed
hydrography, and the paper must not imply that it does.
