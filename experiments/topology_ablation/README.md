# Topology ablation — what this experiment is and how to reproduce it

A controlled test of whether a rainfall–runoff LSTM actually uses the river network it is
given. The model and its full training configuration are held fixed; a single input
coordinate changes, so any movement in skill is attributable to the spatial information
supplied.

## Start here

| File | What it gives you |
|---|---|
| `analysis/VERIFIED_NUMBERS.md` | **Every load-bearing number, with both tails.** Generated, not hand-written. Cite a row rather than recomputing. |
| `verify_all_numbers.py` | Regenerates that file from the raw runs. Run this after any change to `runs/`. |
| `analysis/README.md` | What each result file establishes — **and which carry an interpretation we later retired.** Read it before quoting any single one. |
| `preregistrations/` | Hypotheses and falsification criteria, one per experiment, written before its runs. |

## Finding your way around the scripts

They are named by what they do, so the prefix is the index:

| Prefix | Role |
|---|---|
| `build_*` | Construct an input feature file from the committed basin and edge lists. One per input family. |
| `run_*` | Train and evaluate a condition: writes its config, trains, renames the run to its canonical name, evaluates. Idempotent — an finished run is skipped. |
| `analyze_*` | Read stored run outputs and write one `analysis/*.md`. Train nothing, so they are cheap and safe to re-run. |
| `plot_*` | Build a paper figure, recomputing its values from the raw runs at plot time. |
| `verify_all_numbers.py` | Regenerate `analysis/VERIFIED_NUMBERS.md`, the single source of truth. |
| `recompute_xai4science.py` | Regenerate the submission's numbers directly from run metrics. |
| `validate_edges_nhdplus.py` | Check the inferred edges against NHDPlus via USGS NLDI. |

`features/` holds the generated input files (large, gitignored, rebuilt by the `build_*` scripts);
`configs/` the generated run configs; `notebooks/` the Colab notebooks that ran the GPU work.

## Conditions

Every condition is the stock `cudalstm` from NeuralHydrology at seeds 11/13/17, identical
configuration throughout, differing only in what is appended to the per-timestep input:

- `L` — baseline, no spatial information
- `L_T` — five static topology features (graph depth, upstream count, upstream area,
  in-degree, upstream-area fraction); `L_noID_T` repeats it without the basin one-hot
- `L_upQ` — area-weighted mean **observed** discharge of the upstream set, lagged one day
- `L_upQpred` — the same input, with neighbour discharge from a first-stage model
- `L_upQshuf` — the same input with its time index shuffled (capacity control)
- `L_upQdistctrl`, `L_upQknn2`, `L_upQrandclean`, … — alternative neighbour selectors

Raw results live at
`runs/topology_ablation/component0/<condition>_component0_seed<s>/test/model_epoch030/test_metrics.csv`.

## Statistical protocol

Comparisons are paired per basin against the matching baseline seed and tested with a
Wilcoxon signed-rank test. **Significance is judged by the weakest of the three seeds and
never by pooling them.** For a null condition the weakest seed is the one with the
*largest* p, since that is the seed least able to reject. Directional contrasts are
one-sided; null conditions are two-sided. Two estimators appear and differ in the third
decimal: the cross-seed mean of per-seed medians, and the pooled median over all
basin–seed pairs.

## Reproducing

```bash
python3 experiments/topology_ablation/verify_all_numbers.py      # regenerate the ledger
python3 experiments/topology_ablation/plot_distance_decay.py     # figures
python3 experiments/topology_ablation/plot_flow_bands_figure.py
python3 experiments/topology_ablation/plot_concept_figure.py
```

Figure generators recompute their values from the raw runs at plot time, so a figure
cannot drift from the numbers it reports. Shared typography and palette live in
`figstyle.py`.

## Scope

One connected component of an inferred river network in the eastern United States. The edges are
heuristic — inferred from drainage area, elevation and a distance cap rather than surveyed
hydrography; `topology_analysis/` defines the rule and `analysis/NHDPLUS_VALIDATION.md` measures how
much real connectivity it recovers. Three seeds. Effects are a few hundredths of NSE. These bounds
are real and are stated wherever the results are reported.
