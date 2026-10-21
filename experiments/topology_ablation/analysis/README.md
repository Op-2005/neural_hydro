# analysis/ — re-analysis of stored runs (zero training)

Each file is a Markdown result produced by an `analyze_*.py` / `build_*.py` script in the parent
directory, consuming the `test_metrics.csv` / `test_results.p` under
`runs/topology_ablation/component0/`.

**Start with `VERIFIED_NUMBERS.md`** — it is the generated single source of truth for every
load-bearing number, with both tails. Cite a row rather than recomputing.

**Read this before reading the rest.** This directory is a chronological experimental record,
so it contains results whose *interpretation* was later falsified by our own controls. The files
are kept because the measurements are correct and the sequence is the honest history; the
headings below mark what each one still supports. The two reversals that matter:

- **2026-08-10 — topology-specificity falsified.** A distance-matched control (non-upstream
  basins at the same distance as true parents) reproduced the full gain. **The gain is
  proximity, not connectivity, and not physical routing.** Every file below tagged *(retired
  interpretation)* was written before this and argues a routing mechanism that no longer holds.
- **2026-08-12 — the depth gradient is definitional, not physical.** Headwaters receive a
  constant-zero input by construction, so "gain rises with depth" followed from the feature
  definition rather than from routing. The depth stratification is absent from the paper.

## Current results — what the paper rests on

| File | What it establishes |
|---|---|
| `VERIFIED_NUMBERS.md` | **Every load-bearing number, both tails. Generated. Cite this.** |
| `XAI4SCIENCE_NUMBERS.md` | The same, regenerated for the workshop submission by `../recompute_xai4science.py` |
| `RESULTS_2X2_MULTISEED.md` | Static-topology 2×2, 3-seed — static features are inert (the paper's static null) |
| `MULTISEED.md` | 3-seed confirmation of the neighbour-discharge headline |
| `SIGNIFICANCE.md` | Deployable (realizable) gain: paired Wilcoxon vs `L` and vs the shuffled null |
| `MECHANISM_MULTISEED.md` | **forward / reversed / random / distance-control, 3-seed.** Holds the 2026-08-10 distance-control result that falsified topology-specificity. |
| `TOST_DISTANCE_CONTROL.md` | Equivalence testing on that control. **Pooled TOST passes at only 1 of 3 seeds**, which is why the paper claims indistinguishability at this sample size and *not* equivalence. |
| `COMPLETION_SUITE.md` | The 2026-08-12 suite: distance sweep, k-nearest-gauge selectors, precipitation decomposition, clean random rewire |
| `INDEGREE_CONFOUND.md` | Neighbour count does not explain the kNN advantage (in-degree-stratified) |
| `GEOMETRIC_CONTAINMENT.md` | Nesting bounded geometrically from gauge coordinates and areas (**supersedes** `CONTAINMENT.md`, which used an area proxy) |
| `NHDPLUS_VALIDATION.md` | **New for the camera-ready.** Validates the inferred edges against NHDPlus via USGS NLDI: under 2% are genuine upstream relations, and real connectivity is far more local than our rule. Strengthens the proximity reading. |
| `EDGE_RULE_SENSITIVITY.md` | Mean edge length barely moves as the area ratio is swept — the length is set by the distance cap, *not* the area threshold |
| `METRIC_HONESTY.md` | log-NSE ε-sensitivity and the KGE decomposition (the gain does **not** replicate under KGE) |
| `COMPLIANCE.md` | Gain persists on already-well-predicted basins (not baseline-rescue); also the byte-identical config diff |
| `STAGE1_GAP.md` | Stage-1 train-vs-test degradation for the realizable model |
| `PAPER_TABLE.md` | Consolidated publication table — `../build_paper_table.py` |
| `contrasts.csv`, `table.csv` | tabular dumps used by the above |

## Superseded or retired-interpretation files

Kept for provenance. The numbers are correct; the mechanism they were taken to support is not.

| File | Status |
|---|---|
| `RESULTS.md` | Single-seed 2×2, superseded by `RESULTS_2X2_MULTISEED.md` |
| `DEPTH_SIGNIFICANCE.md` | *(retired interpretation)* Per-depth Wilcoxon. The depth gradient is definitional — see above. |
| `CONFOUND.md` | *(retired interpretation)* Depth gradient vs basin size |
| `FEATURE_MAGNITUDE_CONFOUND.md` | *(retired interpretation)* Partial Spearman: depth survives area and feature magnitude |
| `ROUTING_BASELINE.md`, `ROUTING_BASELINE_3SEED.md` | *(retired interpretation)* No-ML routing baselines vs the LSTM. The comparison is valid; "routing" is the wrong label for what the LSTM exploits. |
| `GRAPH_ROBUSTNESS.md` | *(retired interpretation)* R1-proxy: gain lives in nearest-parent structure rather than excess edges |
| `K2_GRAPH_CHECK.md` | ≤2-parent pruned-graph check, 3-seed — the gain is not a heuristic-density artifact. Still load-bearing; the routing framing in its prose is not. |
| `DIRECTIONALITY.md` | Single-seed directionality controls, superseded by `MECHANISM_MULTISEED.md` |
| `CONTAINMENT.md` | Superseded by `GEOMETRIC_CONTAINMENT.md` (the gauge sits at the catchment outlet, which the area proxy got wrong) |

Regenerate any file by running its script from the repo root, e.g.
`python experiments/topology_ablation/build_paper_table.py`. Re-run
`python experiments/topology_ablation/verify_all_numbers.py` after any change to `runs/`.
