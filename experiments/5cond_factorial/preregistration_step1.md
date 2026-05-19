# Pre-registration — Step 1: Matched-Budget L Control

**Status:** Pre-registered 2026-05-12, before any data is observed.
**Framework reference:** `experiments/5cond_factorial/analysis/testing_framework_proposal.md` §3, Step 1.
**Author session:** `/crs-unleashed` 2026-05-12.

---

## Hypothesis

The L − G NSE gap of +0.050 observed in the 5-condition factorial is **dominantly a training-budget confound, not an architecture confound.** NH `cudalstm` benefits from ~186× more gradient updates per epoch than the DirectedGraphLSTM trainer (per-window batching forces all 183 basins into a single sample, giving ~14 vs ~2,610 steps/epoch at batch=256).

If NH `cudalstm` is constrained to the same total gradient-update budget as the graph trainer used (~420 total steps over the 30-epoch sweep), it should land within 0.01 NSE of G's median.

## Pre-registered design

- **Model:** NH `cudalstm` exactly as in the 5cond runs (hidden_size=64, dropout=0.4, initial_forget_bias=3, MSE loss, Adam lr=1e-3, clip_gradient_norm=1).
- **Data:** Component 0 (183 basins). Train 1990–1999, test 2005–2008. Maurer forcings + 5 static attrs + basin one-hot encoding (671-dim). Identical to L_seed{11,13,17} configs.
- **Training budget:** Exactly **420 gradient updates total** with batch_size=256. After step 420 the model is evaluated on the test period.
- **Seeds:** {11, 13, 17}. Three seeds for paired comparison with the 5cond L and G runs.
- **Output dir:** `runs/5cond_factorial/L420_seed{N}/`. Same layout as L_seed* but produced by a standalone script (see `experiments/training/train_matched_budget_lstm.py`).
- **Metric stack:** NSE, KGE, log-NSE — computed identically to `compare_5conditions.py` so contrasts are paired with the existing G runs.

## Success criterion

Paired per-basin median Δ NSE (L_420 − G) ∈ [−0.01, +0.01], with bootstrap 95% CI excluding both +0.03 and −0.03.

This translates to:
- L_420 median NSE in [0.59, 0.62] (G's median was 0.609).
- The 549 paired (basin, seed) ΔNSE values, sorted, should have a median near zero.

## Falsification criterion

Paired per-basin median Δ NSE (L_420 − G) ≥ +0.03, with bootstrap CI excluding zero.

Translation: even when limited to 420 gradient steps, cudalstm beats G by ≥ 3 NSE points. This would mean the architecture difference between `nn.LSTM` (cuDNN-fused) and DirectedGraphLSTM-with-no-edges (`nn.LSTMCell` Python loop) genuinely matters at this scale, beyond what training budget explains.

## Pre-committed analyses

1. **Headline number:** L_420 median NSE per seed, cross-seed median, bootstrap CI.
2. **Paired contrast:** L_420 − G per basin × seed (n=549). Median Δ + bootstrap CI + n strongly + / − at ±0.05.
3. **Trajectory sub-plot:** NSE at steps {50, 100, 200, 420} from saved checkpoints — shows where cudalstm is in its descent.

## Pre-committed null control

L at step 0 (random init, no training). Expected median NSE ≪ 0 (random predictions vs observed flow). If somehow this comes out > 0.1, indicates a data-leak issue that invalidates all conclusions.

## Pre-committed robustness check

Re-run seed 11 with a **different effective batch size** that produces 420 steps (e.g., batch_size=128 → 5,222 steps/epoch, so train 0.08 epochs; batch=512 → 1,306 steps/epoch, train 0.32 epochs). If the L_420 NSE varies by > 0.02 across these batch-size variants, the "matched-steps" framing is itself a confound — the right matching is "matched-examples" instead.

## Compute estimate

- ~420 forward+backward steps on Component 0 (~668k samples, batch=256, model ~50k params).
- On Apple Silicon CPU: ~150–250 ms per step → ~1.5–2 min per seed.
- 3 seeds = ~5 min. Robustness check adds ~5 min. Total ~10 min.

## Reporting protocol

- Append results to this file under a "Results (post-run)" section, dated.
- Update `experiments/analysis_outputs/5cond_component0/RESULTS.md` with the L_420 row.
- Append concise audit entry to `CURRENT_STATE.md`.
- Append result-and-interpretation entry to `JOURNAL.md`.
- Commit and push (`eff54e2`-style commit message).

## Pre-committed paper-narrative implications

- If hypothesis confirmed (success criterion met): the headline "L beats G by 0.05 NSE" is replaced by "L and G are equivalent at matched budget; G's apparent loss in the 5cond run was a training-budget artifact." `experiments/5cond_factorial/analysis/5cond_run_analysis.md` gets an addendum, not a rewrite. Future work targets G+T+M − G under proper training budget.
- If hypothesis falsified: there IS a residual architecture difference. The architecture-confound framing of the original A/B/C analysis was partially right after all. A separate investigation is needed before claiming anything about graph signal at scale.

## Pre-committed *what we will not do*

- Will not extend training beyond step 420 mid-run.
- Will not change the metric stack.
- Will not filter basins by quality.
- Will not run additional seeds if the 3-seed result is unclear — will pre-register Step 1B with the additional seeds explicitly.

---

## Results (post-run, 2026-05-12)

### Headline numbers

| Trainer | Total gradient steps | Total examples seen | Cross-seed median NSE |
|---|---|---|---|
| L (cudalstm, 30 epochs) | 78,330 | ~20M | **0.653** |
| G (graph trainer, 30 graph-epochs) | 420 | ~20M | 0.609 |
| **L_420 (cudalstm, matched steps)** | 420 | ~107k | **0.502** |

Per-seed L_420 medians: 0.521 / 0.502 / 0.501 (seeds 11/13/17).

### Pre-registered paired contrasts

**L_420 − G (n=549 basin × seed pairs):**
- Median Δ NSE: **−0.100**
- Bootstrap 95% CI: [−0.105, −0.092]
- n strongly positive (Δ ≥ +0.05): 9
- n strongly negative (Δ ≤ −0.05): **431**
- Fraction of paired comparisons where G > L_420: **94.5%**

**L − L_420 (full vs matched-budget cudalstm, n=549):**
- Median Δ NSE: +0.147
- Bootstrap 95% CI: [+0.138, +0.158]
- Confirms cudalstm needs full training; at 420 steps it is far from converged.

### Decision per pre-registration

- **Success criterion (L_420 − G in [−0.01, +0.01]):** NOT met. Δ = −0.100, far outside the band.
- **Falsification criterion (L_420 − G ≥ +0.03):** NOT met. Δ went in the opposite direction.
- **Result: third-category — L_420 ≪ G.** This was not anticipated by either pre-registered outcome. Reporting it as-is per the pre-registration discipline.

### Interpretation

The "matched gradient steps" framing was **biased toward whichever trainer has the larger effective per-step batch.** Each graph-trainer step processes 256 windows × 183 basins ≈ 47k (basin, window) examples; each cudalstm step processes 256 examples. At 420 steps:
- Graph trainer has seen ~20M examples (full data, 30 revisits each).
- L_420 has seen 107k examples (~16% of one pass through the data).

So matched-steps is *not* a meaningful matching variable when batch shapes differ by 200×.

**What this DOES tell us:**
- The original L − G gap (+0.050 in 5cond) is NOT explained by gradient-step count alone. Both L (78k steps) and G (420 steps) saw the same 20M examples; L still beats G by 0.05 at matched examples.
- The gap must be due to one of: gradient noise scale (smaller batches → higher variance → effective regularization), data-exposure pattern (random `(basin, window)` sampling vs whole-window sampling), or a real architectural difference (cuDNN `nn.LSTM` vs Python `nn.LSTMCell` loop). **TBD by next experiment.**

**What this DOES NOT tell us:**
- Whether the graph trainer can be made to beat NH cudalstm by changing its training regime (smaller batches, more epochs, etc.). That's a separate hypothesis test.

### Implication for the paper narrative

The earlier framing "L − G is dominantly a training-budget confound" was a *partial* truth. Step count was the wrong matching variable. Example count matched ⇒ L still beats G. So either we need a redesigned graph trainer (more gradient steps per epoch via smaller batches) to test the "more updates → closes the gap" hypothesis, or we accept a genuine residual L > G effect at matched data exposure.

The architecture-confound framing of the original A/B/C analysis is partially vindicated by this result: the L > G effect SURVIVES matching gradient steps in the wrong direction (confirms a non-step-count cause) AND matching data exposure (the original 5cond comparison). Whatever drives L > G isn't simply about update count.

### Pre-committed actions completed

- [x] Append results to this file (this section).
- [x] Cross-seed paired Δ + bootstrap CI computed.
- [ ] Update `RESULTS.md` with L_420 row (deferred to next session).
- [ ] Append concise audit entry to `CURRENT_STATE.md` (this session, below).
- [ ] Append result-and-interpretation entry to `JOURNAL.md` (this session, below).

### Pre-committed *what I did not do*

- Did not extend training beyond step 420 mid-run (would have been a pivot under contact).
- Did not re-run with different batch sizes to "find the right matched-budget framing" (would have been re-fitting hypotheses to results).
- Did not silently abandon the pre-registered hypothesis — it failed, reported as third-category.
