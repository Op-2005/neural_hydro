# analysis — early-phase re-analysis and its outputs

Post-hoc scripts for the **pre-ablation** experiments: the 23-basin Texas pilot, the
dynamical-systems probes, the first scaled A/B/C comparison, and the 5-condition factorial. None of
them train anything; each reads stored run outputs and writes artifacts into `outputs/`.

The current study has its own analysis directory under `topology_ablation/`. Nothing here is used by
the paper.

## Layout

Scripts sit at the top level; `outputs/` holds what they generate, one subfolder per experimental
phase. Each output subfolder carries its own `README.md` or `RESULTS.md` stating what that phase
found.

Run any script from the repo root; each writes into its own `outputs/` subfolder and prints the
values it saves.

## What each script answers

| Script | Question it was written to answer |
|---|---|
| `compare_results.py` | Per-basin and depth-stratified NSE across a baseline and one or more graph variants. The main tool for the early "graph vs baseline" comparisons. |
| `analyze_results.py` | Hydrographs, learned-weight inspection, and per-basin ΔNSE against static catchment attributes. |
| `ensemble_analysis.py` | Whether the graph variants make *different* errors. They did not — their error correlations were near-unity, so ensembling bought almost nothing, which was early evidence that the variants were not learning distinct mechanisms. |
| `investigate_correlation.py` | What per-basin improvement correlates with. Strongly with baseline skill, which is how the "the graph helps already-well-predicted basins" reading arose. |
| `compare_abc_component0.py` | The first scaled A/B/C result on the 183-basin network. |
| `compare_5conditions.py` | The 5-condition factorial. See `../5cond_factorial/analysis/` for the write-up and the post-mortem. |
| `plot_e0_5_saturation.py`, `plot_e0_5_multiseed.py` | Whether the baseline's training loss had saturated, and how much it moved across seeds. The multi-seed answer is why this project treats any gain smaller than the baseline's own seed spread as not a result. |

Probe scripts that generate the `dynamical_systems_probes/` outputs live in `../probes/`.
