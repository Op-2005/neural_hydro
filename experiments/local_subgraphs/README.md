# local_subgraphs — was the network simply too large?

After the 5-condition factorial found that graph features and message passing *hurt* at 183-basin
scale, the natural objection was scale itself: 183 basins span several HUC regions, so a graph
signal might wash out across that heterogeneity. This experiment tested that by rebuilding the
same comparison on smaller, climatically coherent regional subgraphs.

**What it settled: scale was not the explanation.** The same pattern appeared on single regional
subgraphs, which pointed at the experimental design rather than the size of the network. Together
with the architecture post-mortem, that is what triggered the restart as a controlled
input-only ablation.

- `build_local_subgraphs.py` partitions the study network into regional subgraphs and writes the
  basin and edge lists under `basin_lists/`, with `subgraph_manifest.csv` as the inventory.
- `run_subgraph_sweep.py` (and `run_all_local.sh`) trains the conditions per subgraph.
- `analyze_subgraphs.py` compares them; `analysis/INVARIANT.md` records what held across subgraphs.
- `preregistration_local_scale.md` is the hypothesis as written before the runs.

The subgraph basin and edge lists are still read by some feature builders in
`../topology_ablation/` when a non-`component0` network is requested. The paper uses `component0`
only, so that branch is never taken for any reported result.
