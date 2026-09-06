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
| `analysis/*.md` | One file per result, each naming the script that produces it. |
| `preregistration_*.md` | Hypotheses and falsification criteria, written before the runs. |

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

183 eastern-US CAMELS basins forming one connected component of an inferred river network.
Edges are heuristic — area ratio ≥ 1.5, decreasing elevation, within 150 km — not surveyed
hydrography such as NHDPlus. Three seeds. Effects are a few hundredths of NSE. These bounds
are real and are stated wherever the results are reported.
