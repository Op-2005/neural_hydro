# Local-Subgraph Batch — Finding the Right Scale

**Created:** 2026-05-13 (post-professor-meeting).
**Status:** methodology step toward the paper. Local-first (CPU); no Colab needed.

## The hypothesis

The 5-condition factorial on 183 basins (spanning 6 HUC regions) found that graph
features and message passing *hurt* relative to a plain LSTM. The meeting takeaway:

> **183 basins may be fundamentally too large** for the LSTM to benefit from graph
> structure. The signal washes out across a heterogeneous, multi-region network.
> Test on small, locally-coherent subgraphs where graph structure should matter.

This batch builds the optimized small-scale testing environment to find the scale
at which graph signal reappears (if it does). That's the methodology path: establish
the right unit of analysis *before* committing GPU time to large runs.

## Why this is local-first (no Colab)

These subgraphs are **13–23 basins**. A 3-condition × 3-seed sweep runs in
**~15–30 min on a laptop CPU**. Colab / GPU is reserved for *large* runs that
genuinely need GPU speed — the eventual scale-up of a winning configuration, not
this fast iteration loop. (An optional Colab notebook for that scale-up lives in
`notebooks/_optional_scaleup/`.)

## The tracked invariant

Per the professor: report the **loss distribution across 3 seeds** as the stable
quantity. Concretely: per-seed median NSE → **mean ± std across seeds**. As we revise
the model and the data, the mean should improve while the std stays roughly constant.
`analyze_subgraphs.py` computes this.

The single number that answers the paper question: **G+T+M − L** (paired per-basin
median ΔNSE) per subgraph. Positive ⇒ graph features beat the standard LSTM there.

## The 6 subgraphs

Built by a shortest-path walker on the basin distance graph (the professor's
prescription: "make the basin graph with distances, use a shortest-path walker,
choose a node, trim within shortest-path distance — guaranteed local geography").
Seeds are pre-committed (fixed standard test set, not re-randomized per run).

| Subgraph | Basins | HUCs | Seed basin |
|---|---|---|---|
| sg_midatlantic | 16 | 02/05 | 01594950 |
| sg_ohio | 15 | 02/05 | 03026500 |
| sg_tennessee | 15 | 03/06 | 03455500 |
| sg_southeast | 13 | 02/03 | 02055100 |
| sg_northeast | 16 | 02/04 | 01516500 |
| sg_texas_pilot | 23 | 12 (one region) | historical pilot |

`sg_texas_pilot` is the climate-coherent anchor where the original +0.078 NSE was seen.

## Conditions (minimal informative set)

| ID | What |
|---|---|
| L | NH cudalstm baseline (field standard) |
| G | DirectedGraphLSTM, empty edges (architecture-matched control) |
| G+T+M | DirectedGraphLSTM, full edges + topology features (the full model) |

G+T and G+M are dropped from the default to keep iteration fast; add them back per
subgraph via `--conditions L G G_T G_M G_T_M` once a subgraph is chosen for a deeper dive.

## How to run (local)

```bash
# All 6 subgraphs, 3 conditions, 3 seeds (~15-30 min on CPU):
bash experiments/local_subgraphs/run_all_local.sh

# One subgraph:
bash experiments/local_subgraphs/run_all_local.sh sg_northeast

# Override knobs:
SEEDS="11 13" CONDITIONS="L G" bash experiments/local_subgraphs/run_all_local.sh
```

Idempotent — completed runs skip. Outputs land in `runs/local_subgraphs/<sg>/<cond>_seed<N>/`.

## Files

| File | Purpose |
|---|---|
| `build_local_subgraphs.py` | Shortest-path-walker subgraph generator. Writes `basin_lists/`. |
| `run_subgraph_sweep.py` | Trains L + graph conditions for one subgraph × seeds. CPU by default. |
| `run_all_local.sh` | One-command local entry point: build → sweep all → analyze. |
| `analyze_subgraphs.py` | Computes the 3-seed loss-distribution invariant + G+T+M−L contrast. Writes `analysis/INVARIANT.md`. |
| `basin_lists/` | The 6 subgraph basin/edge files + `subgraph_manifest.csv`. |
| `configs/` | Auto-generated per-subgraph L configs. |
| `analysis/` | Invariant table, contrasts, INVARIANT.md. |
| `notebooks/_optional_scaleup/` | Colab notebook — ONLY for the eventual large scale-up run. Not needed for this batch. |

## Where this sits in the paper methodology

1. 5cond factorial (183 basins) → graph hurts. **[done]**
2. **This batch: does graph help at local scale? Find the right unit of analysis. [now]**
3. If yes → deepen the factorial + architectural revisions on the winning scale (Phase 4 of `../5cond_factorial/analysis/post_meeting_plan.md`).
4. Scale the winning config back up (Colab/GPU) → confirm it holds.
5. Robustness (NHDPlus edges, seed expansion, out-of-time) → paper.
