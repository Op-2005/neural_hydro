# Pilot 23-Basin Outputs

All post-hoc analysis outputs from the April 2026 pilot on the 23-basin
HUC-12 Texas network (runs 03–13). Each file is reproducible from one of
the analysis scripts in `../../analysis/`. Per-run NSE numbers and
narratives live in the per-run `NOTES.md` under `../../../runs/`.

| File | Produced by | Shows |
|---|---|---|
| `per_basin_analysis.csv` | `compare_results.py` | Per-basin NSE across baseline / Graph+Edge / Graph+Frozen on the 23-basin pilot. The headline-comparison table in raw form. |
| `per_basin_edge_warm.csv` | `compare_results.py` | Detailed per-basin numbers for run 06 (the +0.078 headline). |
| `ensemble_analysis.csv` | `ensemble_analysis.py` | Error-correlation matrix across attention/sigmoid/mean variants (runs 06, 09, 10). Showed correlations 0.994–0.999 — variants converge to nearly identical predictions. |
| `attention_weights.csv` | `analyze_results.py` | Learned attention weights per edge (run 09). |
| `sigmoid_gates.csv` | `analyze_results.py` | Learned sigmoid gates per edge (run 10) — empirically all converged near 0.70. |
| `learned_weights_edge_warm.{json,png}` | `analyze_results.py` | W_msg_edge and W_out statistics + plot for run 06. |
| `learned_weights_frozen.png` | `analyze_results.py` | Same for run 07 (frozen-LSTM isolation). |
| `learned_weights_summary.json` | `analyze_results.py` | Cross-run weight-statistics summary. |
| `hydrograph_08158700.png` | `analyze_results.py` | Observed vs baseline vs graph time series, basin 08158700 (depth-1 leaf). |
| `hydrograph_08165300.png` | `analyze_results.py` | Same for 08165300, the outlier basin (baseline NSE −6.25). |
| `hydrograph_08189500.png` | `analyze_results.py` | Same for 08189500 (depth-3, well-predicted). |
| `delta_vs_properties.png` | `analyze_results.py` | Per-basin ΔNSE plotted against area, elevation, depth, n-upstream. |

Status: these are historical pilot artifacts. The pilot's +0.078 NSE
headline did not replicate at Component-0 scale (see
`../abc_component0/RESULTS.md`). These files are kept for reference
and are still valid as descriptions of pilot-scale behavior.
