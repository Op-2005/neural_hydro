# Methodology-Compliance Analysis (Steps A & B)

Re-analysis of stored predictions, seeds [11, 13, 17]. No training.

## Step A — headline contrasts in NSE / KGE / log-NSE

Realizable (L+upQ_pred − L) and oracle (L+upQ − L), paired per basin, pooled seeds.

| Metric | oracle Δ | realizable Δ | null Δ |
|---|---|---|---|
| NSE | +0.0351 | +0.0225 | +0.0039 |
| log-NSE | +0.0159 | +0.0270 | -0.0029 |

## Step B — does the realizable gain persist on WELL-predicted basins?

| L baseline NSE bucket | n | median realizable Δ (NSE) |
|---|---|---|
| <0.3 (bad) | 26 | +0.2365 |
| 0.3-0.6 (mid) | 181 | +0.0529 |
| >0.6 (good) | 342 | +0.0115 |

**Realizable Δ on already-good basins (L NSE > 0.6): +0.0115.** PASS — real signal, not baseline-rescue