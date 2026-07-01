# Multi-Seed Confirmation — Upstream-Signal Headline

Component 0 (183 basins), stock cudalstm, seeds 11/13/17. Seed-11 oracle/predicted medians are the recorded originals (folders overwritten in a drive merge); seeds 13/17 are freshly measured.

## Per-condition median NSE (mean ± std across seeds)

| Condition | mean ± std | per-seed |
|---|---|---|
| L | +0.6532 ± 0.0018 | 11:0.653, 13:0.651, 17:0.656 |
| L_upQ | +0.6911 ± 0.0087 | 11:0.703, 13:0.688, 17:0.682 |
| L_upQpred | +0.6784 ± 0.0084 | 11:0.683, 13:0.686, 17:0.667 |
| L_upQshuf | +0.6655 ± 0.0077 | 11:0.673, 13:0.655, 17:0.669 |

## Paired Δ vs L (per seed; measured seeds only)

| Contrast | per-seed Δ | cross-seed mean |
|---|---|---|
| oracle (observed) | 13:+0.0474, 17:+0.0213 | +0.0343 |
| realizable (predicted) | 13:+0.0258, 17:+0.0131 | +0.0194 |
| null (shuffled) | 11:-0.0056, 13:+0.0093, 17:+0.0040 | +0.0026 |

## Step A — realizable − null (capacity-controlled clean effect)

Per-seed (realizable − null): 13:+0.0151, 17:+0.0078

**Cross-seed mean +0.0115 ± 0.0037; all positive: True.** PASS (>= +0.010)

## Step B — depth-stratified realizable gain (routing check)

| depth | n | median realizable Δ (pooled seeds) |
|---|---|---|
| 0 | 66 | -0.0027 |
| 1 | 162 | +0.0191 |
| 2 | 102 | +0.0291 |
| 3 | 32 | +0.0341 |
| 4 | 4 | +0.0152 |

depth≥2 median Δ +0.0291 vs headwater (depth 0) -0.0027 → diff +0.0319. PASS (routing signature: downstream benefits more)
