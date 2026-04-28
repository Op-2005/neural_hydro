# Analysis Outputs

Generated artifacts from `../analysis/` (post-hoc analysis scripts) and
`../probes/` (model-behavior probes). Do not edit by hand — these files
are reproducible from the scripts in those folders. If you need to
regenerate, see `../analysis/README.md` and `../probes/README.md`.

## Top-level files (23-basin pilot, runs 05–11)

Produced by scripts in `../analysis/`. Each row in the table tells you
which script produced the file and which `INSIGHTS.md` finding it
supports.

| File | Produced by | Supports finding |
|---|---|---|
| `per_basin_analysis.csv` | `compare_results.py` | Per-basin NSE across baseline / Graph+Edge / Graph+Frozen — feeds Findings #1, #3 |
| `per_basin_edge_warm.csv` | `compare_results.py` | Per-basin detail for run 06 (headline) |
| `ensemble_analysis.csv` | `ensemble_analysis.py` | Finding #4 — error correlations 0.994–0.999, ensemble gain only +0.0005 |
| `attention_weights.csv` | `analyze_results.py` | Run 09 learned attention weights per edge |
| `sigmoid_gates.csv` | `analyze_results.py` | Run 10 learned sigmoid gates per edge |
| `learned_weights_edge_warm.json` | `analyze_results.py` | Run 06 W_msg_edge / W_out statistics |
| `learned_weights_edge_warm.png` | `analyze_results.py` | Visualization of run 06 weight stats |
| `learned_weights_frozen.png` | `analyze_results.py` | Same for run 07 (frozen) |
| `learned_weights_summary.json` | `analyze_results.py` | Cross-run weight summary |
| `hydrograph_08158700.png` | `analyze_results.py` | Observed vs baseline vs graph for basin 08158700 (depth-1 leaf) |
| `hydrograph_08165300.png` | `analyze_results.py` | Same for outlier basin 08165300 (baseline NSE −6.25) |
| `hydrograph_08189500.png` | `analyze_results.py` | Same for deep well-predicted basin 08189500 (NSE 0.89) |
| `delta_vs_properties.png` | `analyze_results.py` | Per-basin ΔNSE against area, elevation, depth, n-upstream — Finding #3 |

## Subdirectories

| Folder | What's inside | When created |
|---|---|---|
| `e0/` | E0 self-stabilization probe outputs (Probe A perturbation recovery + Probe B forcing replacement); decision records at σ=0.5 canonical and σ=2.0 sensitivity; meeting-ready figure. | 2026-04-24 |

Each subdirectory has its own `NOTES.md` describing what's there and how
it fits in the project narrative.

## How to read these results

1. Look up which script produced a file in the top-level table.
2. Open `../analysis/README.md` or `../probes/README.md` for the script
   description.
3. Cross-reference with `../../INSIGHTS.md` (numbered findings) and
   `../../runs/README.md` (per-run NSE numbers) to put the artifact in
   context.
4. For everything that has a per-result `NOTES.md` (subfolders),
   open that first — it's the curated narrative.
