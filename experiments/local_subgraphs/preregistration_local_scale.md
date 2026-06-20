# Pre-registration — Local-Scale Test (does graph signal reappear below 183 basins?)

**Status:** Pre-registered 2026-06-20, before observing any local-subgraph results.
**Framework:** `../5cond_factorial/analysis/post_meeting_plan.md` Phases 2–3.
**Author session:** `/crs` 2026-06-20.

---

## Hypothesis

The 5cond factorial (183 basins, 6 HUC regions) found graph features hurt:
G+T+M − L = −0.067 mean NSE. The meeting hypothesis: **183 basins is too large**
for the LSTM to extract benefit from graph structure; on small, locally-coherent
subgraphs the graph signal should reappear.

**Prediction:** on at least one of the 6 local subgraphs (13–23 basins), the paired
per-basin contrast `G+T+M − L` will be **positive** (graph beats plain LSTM).

## Pre-registered design

- 6 subgraphs (walker-built, fixed seeds): sg_midatlantic, sg_ohio, sg_tennessee,
  sg_southeast, sg_northeast, sg_texas_pilot. Sizes 13–23 basins.
- 3 conditions: L (cudalstm), G (graph, no edges), G+T+M (graph, full).
- 3 seeds {11, 13, 17}. 30 epochs. Train 1990–1999, test 2005–2008. Maurer forcings.
- Metric: per-seed median NSE → **mean ± std across seeds** (the tracked invariant).

## Success criterion

`G+T+M − L` paired median ΔNSE **> 0** with the bulk of basins positive
(frac_positive > 0.5) on **≥ 1 subgraph**. Strong success: positive on ≥ 3 of 6.

## Falsification criterion

`G+T+M − L` ≤ 0 on **all 6 subgraphs**. This would mean the graph signal does not
reappear at local scale — the LSTM-fundamentally-doesn't-benefit interpretation holds,
and the next move is architectural redesign (Phase 4) or a pivot to predict-then-route,
NOT smaller scale.

## Smallest decisive result

A single subgraph with `G+T+M − L > +0.02` and frac_positive > 0.6 is enough to
justify the deeper Phase-3 factorial on that scale. We do not need all 6 positive.

## What to do if it fails

If all 6 are ≤ 0: do not shrink further (we're already at 13 basins). Move to Phase 4
architectural revisions (area-weighted aggregation, drop one-hot) on the *least-bad*
subgraph, per the decision tree in post_meeting_plan.md §6.

## Compute

CPU-local, ~15–30 min per the smoke projection (0.9 min/epoch on 13 basins × 30 epochs
× ~few conditions). No GPU. Reported as wall-clock in the run.

## What we will not do

- Will not re-pick subgraph seeds after seeing results (they're a fixed standard).
- Will not extend epochs mid-run to chase a positive result.
- Will not drop the subgraphs where graph loses from the reported table.

---
