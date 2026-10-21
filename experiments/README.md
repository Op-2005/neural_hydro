# Experiments

Every experiment this project ran, in the order it ran them. Each folder holds the **setup** — the
configs, feature builders, runners and pre-registrations — plus the analysis that reads the results
back. Training outputs live under `runs/`, not here.

Each folder has its own README covering what the experiment asked, what it showed, and how to
reproduce it. Read that rather than this file for anything specific.

## The current study

| Folder | What it asks |
|---|---|
| `topology_ablation/` | **The paper.** Does a streamflow LSTM actually use the river network it is given? Stock `cudalstm` held fixed; only the spatial input varies. Start at its README, then `topology_ablation/analysis/VERIFIED_NUMBERS.md` for any number. |

## Earlier experiments, and what each one settled

These ran before the current design and are kept because each one changed the direction of the
project. Their results are superseded as *interpretations* in some cases — the folder READMEs say
which — but the measurements stand and the reasoning is why the current study looks the way it does.

| Folder | What it asked | What it settled |
|---|---|---|
| `configs/`, `basin_lists/` | — | The NH configs and gauge lists for the early baselines. The current study's own configs live under `topology_ablation/`. |
| `training/` | Can a custom graph-LSTM beat a plain multi-basin LSTM? | Built the `DirectedGraphLSTM` and its ablation variants. The architecture was abandoned; see `5cond_factorial/analysis/` for why. |
| `analysis/` | — | Post-hoc comparison scripts for those early runs, with the artifacts they produced under `analysis/outputs/` (pilot, dynamical-systems probes, the first scaled A/B/C result). |
| `probes/` | Does the trained LSTM behave as a self-stabilizing dynamical system? | Yes — hidden-state perturbations are absorbed within a step or two. This came out of the April 2026 PI meeting and motivated asking what *external* signal the model can actually use. |
| `5cond_factorial/` | Do graph features and message passing help at 183-basin scale? | **No — and the comparison turned out to be unusable.** Three confounds (training budget, basin-one-hot redundancy, from-scratch noise) made every negative uninterpretable. Diagnosing that is what produced the current design: hold the model fixed, change only the input. `5cond_factorial/analysis/architecture_analysis.md` is the post-mortem. |
| `local_subgraphs/` | Was 183 basins simply too large a network for the graph signal to survive? | No — the same pattern appeared on single regional subgraphs, which confirmed the problem was the design rather than the scale. |

## Conventions

- **Pre-registration before runs.** Each experiment records its hypothesis, success criterion and falsification
  condition before the runs exist — in `topology_ablation/preregistrations/`, or a
  `preregistration_*.md` beside the older experiments. A pre-registration records intent at the time of writing, never current status.
- **Numbers come from the runs.** `topology_ablation/analysis/VERIFIED_NUMBERS.md` is generated from
  the raw metrics and is the single source of truth for every reported value. Cite a row; do not
  recompute or transcribe.
- **Three seeds, judged by the weakest.** Significance is never pooled across seeds. For a null,
  the weakest seed is the one with the largest p.
- **Results are read back from disk.** The analysis scripts here train nothing; they re-analyse
  stored run outputs, so they are cheap to re-run and safe to trust.
