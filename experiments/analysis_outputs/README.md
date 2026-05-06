# Analysis Outputs

Generated artifacts from analysis scripts in `../analysis/` and probe scripts
in `../probes/`. Organized into three subfolders by experimental phase.

| Subfolder | What's in it | When generated |
|---|---|---|
| [`pilot_23basin/`](pilot_23basin/) | All analysis outputs from the 23-basin Texas pilot (runs 03–13). Hydrographs, learned-weight inspections, ensemble correlations, per-basin tables, sigmoid/attention weight dumps. | April 2026 |
| [`dynamical_systems_probes/`](dynamical_systems_probes/) | E0 self-stabilization probes and E0.5 loss-saturation analysis on the 23-basin baseline. Underpinned the dynamical-systems framing of the project. | Apr 24–26 2026 |
| [`abc_component0/`](abc_component0/) | A/B/C ablation results on the 183-basin Component 0 network (runs 14, 15, 16, single seed). The first scaled-experiment headline numbers. | May 5–6 2026 |

The slideshow-ready summaries live in:
- `abc_component0/RESULTS.md` — A/B/C scaled-run findings + figures.
- `dynamical_systems_probes/e0/NOTES.md` — what E0 tested and what we found.

For the master research direction see `../../idea1.md`.
For the chronology of how these were generated see `../../JOURNAL.md`.
