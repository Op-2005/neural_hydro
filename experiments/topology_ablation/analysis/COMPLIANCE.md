# Methodology-Compliance Analysis (Steps A & B)

Re-analysis of stored predictions, seeds [13, 17]. No training.

## Step A — headline contrasts in NSE / KGE / log-NSE

Realizable (L+upQ_pred − L) and oracle (L+upQ − L), paired per basin, pooled seeds.

| Metric | oracle Δ | realizable Δ | null Δ |
|---|---|---|---|
| NSE | +0.0351 | +0.0215 | +0.0068 |
| log-NSE | +0.0159 | +0.0187 | -0.0226 |

## Step B — does the realizable gain persist on WELL-predicted basins?

| L baseline NSE bucket | n | median realizable Δ (NSE) |
|---|---|---|
| <0.3 (bad) | 17 | +0.2237 |
| 0.3-0.6 (mid) | 119 | +0.0572 |
| >0.6 (good) | 230 | +0.0115 |

**Realizable Δ on already-good basins (L NSE > 0.6): +0.0115.** PASS — real signal, not baseline-rescue