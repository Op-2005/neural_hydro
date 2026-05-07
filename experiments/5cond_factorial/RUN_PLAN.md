# 5-Condition Factorial Run Plan

Operational checklist. Each phase is gated; do not advance until the prior
checkpoint is verified. Last updated 2026-05-06.

## Locked decisions

- [x] Forward-pass optimization: **YES** (using `torch.compile`)
- [x] Number of seeds: **3** (seeds 11, 13, 17)
- [x] Metrics: **NSE + KGE + log-NSE**
- [x] Run-output folder: `runs/5cond_factorial/`
- [x] Experiment master folder: `experiments/5cond_factorial/`

## Phase 1 — Code infrastructure

- [ ] **1.1** Add `empty_graph` variant to `experiments/training/train_graph_component0.py` (DirectedGraphLSTM with empty edges, no topology features). Becomes Condition **G**.
- [ ] **1.2** Add `full_graph_with_topology` variant (DirectedGraphLSTM with full edges + 5 topology static features + edge features). Becomes Condition **G+T+M**.
- [ ] **1.3** Add `torch.compile` wrapping for the DirectedGraphLSTM forward pass. Validate correctness against existing pilot runs (NSE matches within 1e-4).
- [ ] **1.4** Update `experiments/training/README.md` with the 5-variant flag table.

## Phase 2 — Notebook + pipeline

- [ ] **2.1** Build `experiments/5cond_factorial/notebooks/colab_5cond_run.ipynb`:
  - 5 conditions × 3 seeds = 15 training cells
  - Skip-if-done filter that excludes `_SMOKE_` folders
  - Pre-flight smoke test of all 4 graph variants
  - Auto-cleanup of smoke folders after smoke completes
  - End-of-run aggregate analysis call
  - Writes results to `runs/5cond_factorial/<condition>_seed<N>/`
- [ ] **2.2** Generate per-seed config templates for Condition L in `experiments/5cond_factorial/configs/`.
- [ ] **2.3** Smoke-test the notebook through Cell 7 locally to verify cell flow.

## Phase 3 — Analysis pipeline

- [ ] **3.1** Build `experiments/analysis/compare_5conditions.py`:
  - Loads from `runs/5cond_factorial/` (architecture-matched 5-condition design)
  - Three metrics (NSE, KGE, log-NSE) with bootstrap 95% CIs
  - Six clean pairwise contrasts: L − G, (G+T) − G, (G+M) − G, (G+T+M) − (G+T), (G+T+M) − (G+M), (G+T+M) − G
  - Interaction term: (G+T+M) − (G+T) − (G+M) + G
  - Depth-stratified plot, area-stratified plot, outlier-trimmed metrics
  - Auto-fills a `RESULTS.md` template in `experiments/analysis_outputs/5cond_component0/`

## Phase 4 — Pre-flight smoke (small compute, ~30 min CPU)

- [ ] **4.1** Run each new/changed variant for 2 epochs on the 23-basin pilot.
- [ ] **4.2** *(if forward-pass was optimized)* Verify a single 30-epoch run matches existing pilot's NSE within 1e-4.
- [ ] **4.3** Smoke folders auto-clean by the pipeline.

## Phase 5 — Production sweep (user launches on Colab)

- [ ] **5.1** Open `experiments/5cond_factorial/notebooks/colab_5cond_run.ipynb` from GitHub in Colab.
- [ ] **5.2** Set runtime to **T4 GPU** (cheapest compute units).
- [ ] **5.3** Verify Cell 2 auto-detects `camels_us` on Drive.
- [ ] **5.4** Runtime → Run all.
- [ ] **5.5** Wait ~30 hr (multiple sessions; notebook is idempotent).
- [ ] **5.6** Pull `experiments/analysis_outputs/5cond_component0/RESULTS.md` and the figures locally; ask CRS to interpret.

## Notes

- All graph variants (G, G+T, G+M, G+T+M) use the same DirectedGraphLSTM architecture with `torch.compile`. This is the architecture-matched control structure that eliminates the prior B/C confound.
- Condition L (NH cudalstm) is reported as the field-standard baseline; the **L − G** contrast is reported separately as a methodology note (cudalstm wrapper vs DirectedGraphLSTM with empty edges, both with no graph signal).
- Existing single-seed runs (14, 15, 16) are NOT moved into this folder — they remain the historical pilot-at-scale evidence and are cited in the paper as motivating prior work.
