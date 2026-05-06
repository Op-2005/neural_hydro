# Run 16 — Graph-LSTM with full edges + message passing (Condition C), seed 42

**Status.** Single-seed first scaled run for Condition C. Completes the
A/B/C triple at Component-0 scale.

**Model.** DirectedGraph-LSTM with full Component-0 edges (624 edges, 183
basins) and edge features `[log_dist, log_area_ratio, elev_drop]`. Mean
aggregation over parents. **From scratch** (`--no-warm-start`), 30 epochs,
seed 42. Per the locked Condition C specification in `idea1.md`.

**Script.** `experiments/training/train_graph_component0.py --variant
warm --seed 42 --no-warm-start --epochs 30 --baseline-run runs/A_baseline_seed42_*`
(Cell 11 of `notebooks/colab_publication_run.ipynb` on Colab Pro L4).

**Result.** Median test NSE **0.578** (mean 0.545). NSE range
[-0.568, 0.766].

**Per-basin contrasts:**
- vs A (baseline): median Δ **−0.077**; 113 of 183 basins worse by ≥ 0.05;
  15 better by ≥ 0.05; std 0.526 (heavy-tailed)
- vs B (topology features): median Δ **−0.021**; 57 of 183 basins worse by
  ≥ 0.05; 15 better by ≥ 0.05; std 0.082 (much tighter than B−A)

**Why it matters.** **Message passing on top of basin encoding does NOT
help at Component-0 scale, single seed.** Two clear takeaways:

1. **C is worse than A by 7.7 NSE-points at the median.** The pilot's
   +0.078 NSE on 23 basins (run 06) does not replicate at 183 basins. The
   framing's prediction that graph topology supplies useful destabilizing
   forcing is wounded at scale.

2. **C is only slightly worse than B (median Δ −0.021, std 0.082).**
   That tells us most of the gap to A is shared between B and C — i.e.,
   it's not the *message passing* that hurts, it's *whatever both
   ablations have in common* relative to A. Two candidates:
   - Both train from scratch with the modified DirectedGraphLSTM
     architecture, which has different optimization dynamics than
     NH's batched cudalstm.
   - Both have augmented input dimensionality (B adds 5 topology
     scalars; C adds the same effectively-zero-init message channel).

**Depth-stratified pattern (see `experiments/analysis_outputs/abc_component0/nse_by_depth.png`).**
A wins at every depth except depth 4 (n=2, noise-dominated). The gap
(A − C) is roughly constant ~0.05 NSE across depths 0–3, suggesting the
deficit is global, not depth-dependent. This argues against the framing's
prediction that message passing helps deep nodes more than shallow ones.

**Single-seed caveat.** Same as runs 14 and 15: cross-seed variance from
yesterday's E0.5 multi-seed result was ±0.111 NSE on 23 basins. The
single-seed Component-0 numbers could shift by similar magnitudes.
Multi-seed verification (the `'full'` MODE of the notebook) is now the
load-bearing follow-up before any framing-level claim.

**Associated outputs.**
- Per-basin NSE: `test_metrics.csv` (183 rows)
- Run config: `run_config.json` (variant, hyperparameters, final NSE,
  loss + epoch-time history)
- Cross-condition analysis: `experiments/analysis_outputs/abc_component0/`
  (summary.json, per_basin_long.csv, per_basin_deltas.csv,
  delta_distributions.png, nse_by_depth.png)

**Note on baseline-run path.** `run_config.json` records
`baseline_run: /content/nh/runs/A_baseline_seed42_0605_000314` — the
INCOMPLETE first A run on Colab. This is fine: with `--no-warm-start`,
the baseline is only used for cfg/scaler/id_to_int (which depend on data
structure, not training quality). Training was genuinely from scratch.
