# Multi-Seed Confirmation — Upstream-Signal Headline

Component 0 (183 basins), stock cudalstm, seeds 11/13/17. Seed-11 oracle/predicted medians are the recorded originals (folders overwritten in a drive merge); seeds 13/17 are freshly measured.

## Per-condition median NSE (mean ± std across seeds)

| Condition | mean ± std | per-seed |
|---|---|---|
| L | +0.6532 ± 0.0018 | 11:0.653, 13:0.651, 17:0.656 |
| L_upQ | +0.6910 ± 0.0086 | 11:0.703, 13:0.688, 17:0.682 |
| L_upQpred | +0.6785 ± 0.0085 | 11:0.683, 13:0.686, 17:0.667 |
| L_upQshuf | +0.6655 ± 0.0077 | 11:0.673, 13:0.655, 17:0.669 |

## Paired Δ vs L (per seed; measured seeds only)

| Contrast | per-seed Δ | cross-seed mean |
|---|---|---|
| oracle (observed) | 11:+0.0370, 13:+0.0474, 17:+0.0213 | +0.0352 |
| realizable (predicted) | 11:+0.0265, 13:+0.0258, 17:+0.0131 | +0.0218 |
| null (shuffled) | 11:-0.0056, 13:+0.0093, 17:+0.0040 | +0.0026 |

## Step A — realizable − null (capacity-controlled clean effect)

Per-seed (realizable − null): 11:+0.0274, 13:+0.0151, 17:+0.0078

**Cross-seed mean +0.0168 ± 0.0081; all positive: True.** PASS (>= +0.010)

## Step B — depth-stratified realizable gain (routing check)

| depth | n | median realizable Δ (pooled seeds) |
|---|---|---|
| 0 | 99 | +0.0018 |
| 1 | 243 | +0.0199 |
| 2 | 153 | +0.0307 |
| 3 | 48 | +0.0443 |
| 4 | 6 | +0.0152 |

depth≥2 median Δ +0.0329 vs headwater (depth 0) +0.0018 → diff +0.0311. PASS (routing signature: downstream benefits more)
