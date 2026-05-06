# A/B/C First Scaled Run — Component 0 — Results Brief

Single-seed (seed=42) ablation on the 183-basin Component 0 network. Runs 14, 15, 16 in `runs/`. This document is the slideshow-ready summary. Generated 2026-05-06.

---

## The headline numbers

| Condition | Run dir | Median NSE | Mean NSE | Range |
|---|---|---|---|---|
| **A** baseline (NH cudalstm + basin encoding) | `runs/14_lstm_component0_baseline_seed42/` | **0.648** | 0.586 | [-6.50, 0.85] |
| **B** topology-as-features (no message passing) | `runs/15_graph_c0_topology_features_seed42/` | **0.591** | 0.575 | [-0.32, 0.78] |
| **C** full graph-LSTM (edges + message passing) | `runs/16_graph_c0_warm_seed42/` | **0.578** | 0.545 | [-0.57, 0.77] |

**Both ablations are worse than the baseline.** Pilot's +0.078 NSE on 23 basins (run 06) does not replicate at scale.

## The contrasts

| Contrast | Median Δ | Mean Δ | Std | Basins where +Δ > 0.05 | Basins where Δ < −0.05 |
|---|---|---|---|---|---|
| B − A | −0.050 | −0.011 | 0.487 | 20 / 183 | 92 / 183 |
| C − A | **−0.077** | −0.041 | 0.526 | 15 / 183 | 113 / 183 |
| C − B | −0.021 | −0.030 | **0.082** | 15 / 183 | 57 / 183 |

The diagnostic finding: **C − B has much tighter spread (std 0.082)** than C − A or B − A (both ≈ 0.5). Most of the deficit C and B share against A is *not* unique to message passing — it's something B and C have in common relative to A.

## What we attribute the negative result to

Three candidate mechanisms, ranked by likelihood:

1. **Architecture confound.** Condition A uses NH's `cudalstm` (an `nn.LSTM` wrapper, fully CUDA-batched). Conditions B and C use the `DirectedGraphLSTM` class (an `nn.LSTMCell` Python timestep loop). These are not the same architecture; their optimization trajectories differ. The tight C − B distribution is consistent with this — both ablations share whatever the architecture switch costs.

2. **At 183-basin scale, basin encoding alone is enough.** A's median NSE jumped from 0.423 (23-basin pilot) to 0.648 (Component 0). The 183-dim basin one-hot gives the LSTM enough per-basin capacity that topology features become redundant. This is the "graph substitutes for basin encoding" finding from `INSIGHTS.md` Finding #2 finally manifesting at scale.

3. **From-scratch protocol vs. pilot's warm-start.** The pilot's +0.078 used warm-start from run 05. The locked publication protocol uses from-scratch training across all conditions for fair comparison. The pilot's gain may have been mostly an LSTM-drift / warm-start optimization-trajectory artifact (consistent with run 07 showing only +0.013 from "pure graph" with the rest being LSTM weight drift). At scale with no warm-start, that channel disappears.

## Depth-stratified pattern

See `nse_by_depth.png`.

A wins at every depth except depth 4 (n=2 basins, noise-dominated). The gap A − C is roughly **constant ~0.05 NSE across depths 0–3** — a global deficit, not depth-dependent. This argues against the dynamical-systems framing's prediction that message passing should help deeper basins more.

## Per-basin distributions

See `delta_distributions.png`.

- **B − A** and **C − A** are heavy-tailed: both peaked near zero, with a few outlier basins where the ablation helps strongly (Δ > +1) and many where it hurts mildly (Δ ≈ −0.1).
- **C − B** is tight and roughly symmetric around zero (median −0.021, std 0.082), with very few outliers — most basins move only slightly between B and C.

## Caveats — what is NOT yet established

1. **Single seed only.** Yesterday's E0.5 multi-seed result on 23-basin baselines showed cross-seed variance of ±0.111 NSE. With similar variance at scale, the C − A delta could plausibly shift on other seeds. **Multi-seed verification is the load-bearing follow-up before any publishable claim.**
2. **Architecture confound is suspected but not yet measured.** A clean test (Condition G in the 5-condition revised framework — DirectedGraphLSTM with empty edges, no topology features) has not yet been run. The "shared B+C deficit" tightness is suggestive evidence, not proof.
3. **Heuristic edges, not NHDPlus.** Component 0's 624 edges are inferred from area/elevation/proximity, with a 150 km radius cutoff that crosses 6 HUC regions. Many edges likely connect basins with no actual hydrological link. NHDPlus replacement is queued as a robustness check.

## What this means for the project

The negative result is real and meaningful — but its *interpretation* depends on the methodology fixes still to come. Three branches:

1. **If multi-seed confirms** the negative across 5 seeds: workshop-publishable as a strong negative result aligned with Kirschstein 2024, with mechanistic decomposition explaining why pilot's positive result didn't generalize.

2. **If multi-seed disagrees**: cross-seed variance is itself the headline (graph methods are seed-fragile at this scale).

3. **If Condition G (architecture control) reveals** that most of the deficit is the architecture switch, not the topology signal: the result reframes as "DirectedGraphLSTM as currently implemented underperforms NH's cudalstm; we cannot make a topology claim until the architectures are matched." A methodology update, not a topology finding.

The 5-condition revised framework (see `idea1.md`) addresses all three branches simultaneously.

## Files in this folder

| File | What it contains |
|---|---|
| `summary.json` | Cross-condition median NSE, per-seed lists, per-basin Δ statistics |
| `summary_table.txt` | Human-readable summary (same numbers as above table) |
| `per_basin_long.csv` | One row per (condition, seed, basin) — for downstream analyses |
| `per_basin_deltas.csv` | Wide table with A, B, C per-basin NSE + B−A, C−A, C−B deltas |
| `delta_distributions.png` | Histograms of per-basin ΔNSE for the three contrasts |
| `nse_by_depth.png` | Depth-stratified median NSE for each condition |
| `depth_stratified.csv` | Same data as the depth plot, in tabular form |

## Slideshow-ready figures

- **`nse_by_depth.png`** — the cleanest single figure showing A > B > C across depths.
- **`delta_distributions.png`** — three-panel comparison; the tight C − B panel is the diagnostic story.
