# Consolidated Publication Results Table

Zero training — assembly of prior artifacts. Component 0, 183 basins, stock cudalstm,
seeds [11, 13, 17]. All values on held-out test 2005-2008.
Sources: MULTISEED / SIGNIFICANCE / METRIC_HONESTY / ROUTING_BASELINE_3SEED /
DEPTH_SIGNIFICANCE / MECHANISM_MULTISEED / FEATURE_MAGNITUDE_CONFOUND.

## Statistic conventions (read before quoting any number)

- **ΔNSE** = paired per-basin median delta vs L, **cross-seed mean of the per-seed medians**.
  This is NOT the difference of the two conditions' median-NSE columns, so it need not equal
  a column subtraction (realizable: +0.0218 paired vs +0.0253 as a difference of medians).
- **Significance is tested per seed (n=183 basins), never pooled across seeds.** Pooling the
  549 basin×seed observations would treat three correlated measurements of the same catchment
  as independent and understate p by many orders of magnitude. An earlier version of this file
  and of the manuscript reported pooled p-values; both have been corrected. Per-seed values
  live in `SIGNIFICANCE.md`.
- Tables 3-5 report the median over basins collected across seeds, the statistic each of those
  controls was pre-registered on. It differs from the cross-seed mean by < 0.003 NSE wherever
  both are available.

## Table 1 — median skill by condition (mean ± std across 3 seeds)

| condition | NSE | KGE | log-NSE | ΔNSE vs L | per-seed Δ (11/13/17) |
|---|---|---|---|---|---|
| L (baseline) | 0.653 ± 0.002 | 0.716 ± 0.016 | 0.634 ± 0.035 | — | — |
| L+upQ (oracle) | 0.691 ± 0.009 | 0.746 ± 0.014 | 0.715 ± 0.012 | **+0.035** | +0.037 / +0.047 / +0.021 |
| L+upQ_pred (realizable) | 0.678 ± 0.008 | 0.723 ± 0.005 | 0.672 ± 0.020 | **+0.022** | +0.027 / +0.026 / +0.013 |
| L+upQ_shuf (null) | 0.666 ± 0.008 | 0.718 ± 0.015 | 0.628 ± 0.023 | **+0.003** | -0.006 / +0.009 / +0.004 |

Per-seed one-sided Wilcoxon, realizable vs L: **4.6e-08 / 2.8e-12 / 1.4e-03** — significant at
every seed, worst case 1.4e-03. The shuffled null is tested two-sided and reaches significance
at no seed.

*Oracle log-NSE provenance: the 0.715 ± 0.012 value is the 2-seed (13/17) figure computed when
those files were intact, and is what the manuscript reports. Locally only seed 17 is currently
regenerable (giving 0.727); seed 11 was lost in a drive merge and restored to Drive
(`notebooks/colab_oracle_seed11_restore.ipynb`, seed-11 oracle log-NSE Δ vs L = +0.055 at
eps=1e-3), seed 13 is truncated on this machine. Re-sync from Drive to reproduce. The realizable
log-NSE, which is the load-bearing metric, is fully 3-seed and intact.*

## Table 2 — no-ML routing baselines vs LSTM (connected basins, mean ± std 3 seeds)

| predictor | median test NSE | ML? | uses upstream? |
|---|---|---|---|
| R1 — pure routing (a·upQ+b) | +0.324 ± 0.000 | no | yes |
| R2 — routing + local (a·upQ+c·L_sim+b) | +0.664 ± 0.008 | no | yes |
| L (LSTM baseline) | +0.655 ± 0.006 | yes | no |
| L+upQ_pred (realizable) | +0.683 ± 0.008 | yes | yes |
| L+upQ (oracle) | +0.706 ± 0.009 | yes | yes |

*The realizable LSTM beats every no-ML baseline at all 3 seeds. Its margin over the strong R2
baseline is +0.019: the LSTM integrates upstream flow WITH local rainfall-runoff, which linear
routing cannot. Source: ROUTING_BASELINE_3SEED.md.*

## Table 3 — realizable gain by graph depth (per-stratum Wilcoxon)

| depth | n | median Δ | p | sig |
|---|---|---|---|---|
| 0 (headwater) | 99 | +0.002 | 0.24 | no |
| 1 | 243 | +0.020 | 2.6e-9 | yes |
| 2 | 153 | +0.031 | 4.7e-12 | yes |
| 3 | 48 | +0.044 | 8.4e-4 | yes |
| 4 | 6 | +0.015 | 0.34 | no (n=6) |

*Routing signature: the gain is **undetectable** (not "absent" — p=0.24 is failure to detect)
exactly at the headwaters that receive no upstream input, and rises with depth through depth 3.
Depth 4 (n=6) is not interpretable. The gradient is not a size or magnitude proxy: feature
magnitude DECREASES with depth (Spearman -0.369), so the gain rises while its supposed driver
falls, and depth survives partialling out both area and magnitude (partial rho +0.149,
p=4.4e-04). See FEATURE_MAGNITUDE_CONFOUND.md.*

## Table 4 — graph robustness: the gain is not a heuristic-edge artifact

Heuristic edges over-connect (in-degree mean 4.16 / max 15 vs real confluences ~2-3). Pruning
to hydrography-realistic in-degree ≤ 2 (266 edges vs 624). Falsification condition: if the gain
came from the excess edges, pruning would remove most of it.

| level | metric | full graph | k=2 pruned | verdict |
|---|---|---|---|---|
| R1 signal proxy (zero-train) | median NSE | +0.325 | +0.326 | 100% retained |
| LSTM realizable (3-seed) | Δ NSE (connected) | +0.026 | +0.025 (p=1.3e-14) | holds |
| LSTM oracle (3-seed) | Δ NSE (connected) | +0.046 | +0.059 (p=2.8e-43) | strengthens |

*The routing gain lives in the physically-meaningful nearest-parent structure, not the
heuristic's excess edges, confirmed at both the signal-content and trained-model level across
3 seeds (GRAPH_ROBUSTNESS.md, K2_GRAPH_CHECK.md, MECHANISM_MULTISEED.md).*

## Table 5 — topology specificity (forward-connected basins, n=150/seed)

The headline mechanism result. Source: MECHANISM_MULTISEED.md §1.

| edge set | per-seed Δ (11/13/17) | median Δ |
|---|---|---|
| forward (true upstream) | +0.046 / +0.056 / +0.027 | **+0.046** |
| reversed (downstream) | +0.026 / +0.045 / +0.024 | +0.031 |
| random (degree-preserving rewire) | +0.014 / +0.019 / +0.003 | +0.012 |

*Forward − random = **+0.034**, positive at all 3 seeds: a degree-preserving random graph
retains only ~26% of the gain. Directionality (forward − reversed = +0.006, 54% of basins,
null at seed 17) is reported as a mild aggregate preference only, NOT a claim. Caveat carried
into the paper: reversed edges preserve true adjacency and flip only orientation, so they
retain spatial-proximity structure a random rewire destroys — the reversed condition is an
imperfect isolation of direction alone.*
