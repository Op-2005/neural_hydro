# Step C — Metric-Honesty Pass (log-NSE eps-sensitivity + KGE decomposition)

Zero training. Per-timestep obs/sim from `test_results.p`. Pre-reg: `preregistration_hardening_chain.md`.

> **Oracle-column note (2026-07-16):** the oracle (`L_upQ`) seed-11 `test_results.p` was lost in a
> drive merge and restored via `notebooks/colab_oracle_seed11_restore.ipynb` (verified on Drive).
> The oracle log-NSE column below and the seed-11 oracle KGE row are transcribed from that verified
> Colab re-run; the restored file lives on Drive (not this Mac), so those specific cells are not
> yet locally regenerable — everything else is. Re-train reproduced seed-11 oracle NSE within
> cross-environment noise (0.693 vs recorded 0.703; see `preregistration_baseline_completion_and_k2.md`).

## C1 — log-NSE realizable Δ vs eps (stability of the +0.027 headline)

eps = frac × (per-basin mean observed flow). Realizable Δ = median over basins of logNSE(L+upQ_pred) − logNSE(L), pooled seeds. Oracle column now complete (seed-11 restored).

| eps frac | realizable median Δ (log-NSE) | null median Δ | oracle median Δ |
|---|---|---|---|
| 1e-02 | +0.0292 | -0.0029 | +0.0448 |
| 1e-03 | +0.0318 | -0.0067 | +0.0551 |
| 1e-04 | +0.0325 | -0.0102 | +0.0625 |

**log-NSE realizable Δ positive at all eps: True** (range +0.0292 to +0.0325). PASS — headline is not an eps artifact. **Oracle log-NSE Δ (+0.045 to +0.063) exceeds the realizable — the observed upper bound is stronger in log-space too, as expected.**

## C2 — KGE decomposition: where does the seed-13 dip live?

Median over basins of each KGE component per condition, per seed. KGE weakness should localize to r (timing), β (bias), or γ (variability).

### seed 11

| condition | median KGE | median r | median β (bias) | median γ (var) |
|---|---|---|---|---|
| L | +0.6932 | +0.8231 | +1.0620 | +0.8089 |
| realizable | +0.6996 | +0.8410 | +1.0839 | +0.8159 |
| oracle | +0.7575 | — | — | — | (median KGE from restored run; r/β/γ decomposition pending local sync) |
| null | +0.7296 | +0.8366 | +1.0351 | +0.8610 |

realizable − L: ΔKGE +0.0064 | Δr +0.0178 | Δβ +0.0220 | Δγ +0.0070

### seed 13

| condition | median KGE | median r | median β (bias) | median γ (var) |
|---|---|---|---|---|
| L | +0.7374 | +0.8242 | +1.0600 | +0.8849 |
| realizable | +0.7328 | +0.8451 | +1.0239 | +0.8493 |
| oracle | +0.7111 | +0.8429 | +1.0690 | +0.8076 |
| null | +0.6947 | +0.8298 | +1.0829 | +0.8106 |

realizable − L: ΔKGE -0.0046 | Δr +0.0209 | Δβ -0.0360 | Δγ -0.0356

### seed 17

| condition | median KGE | median r | median β (bias) | median γ (var) |
|---|---|---|---|---|
| L | +0.7268 | +0.8298 | +1.0351 | +0.8537 |
| realizable | +0.7406 | +0.8347 | +0.9862 | +0.8777 |
| oracle | +0.7600 | +0.8446 | +1.0763 | +0.8821 |
| null | +0.7475 | +0.8332 | +1.0252 | +0.8719 |

realizable − L: ΔKGE +0.0138 | Δr +0.0049 | Δβ -0.0489 | Δγ +0.0239

