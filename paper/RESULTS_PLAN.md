# Results Section — plan + gap analysis (grounded)

Every number below is copied from an analysis file opened this session. Sources named per row.
Field calibration from RELATED_READING.md (median-across-basins headline; a CDF figure; quote
Kirschstein's null).

---

## GAP ANALYSIS — do we have every result the paper needs?

**Verdict: one gap. Everything is 3-seed EXCEPT the static-topology 2×2 (single-seed, seed 11).**

| Paper claim | Source | Seeds | OK? |
|---|---|---|---|
| C1a static topology ~0 | RESULTS.md (L+T −0.001, L_noID+T +0.003) | **1** | GAP |
| C1b dynamic flow helps (oracle +0.038 / realizable +0.025) | PAPER_TABLE | 3 | ✓ |
| C2 topology-specific (real−random +0.034, p=2e-14) | MECHANISM_MULTISEED | 3 | ✓ |
| C3 deployable (realizable, p=6e-19; vs null p=2e-12, CI) | PAPER_TABLE, SIGNIFICANCE | 3 | ✓ |
| routing mechanism (depth gradient, per-stratum sig) | DEPTH_SIGNIFICANCE | 3 | ✓ |
| confound-checked (area, feature-magnitude) | CONFOUND, FEATURE_MAGNITUDE_CONFOUND | 3 | ✓ |
| beats no-ML routing (R1/R2) | ROUTING_BASELINE_3SEED | 3 | ✓ |
| graph-robust (k=2 ≈ full) | MECHANISM_MULTISEED §2 | 3 | ✓ |
| metric-robust (log-NSE, KGE) | METRIC_HONESTY | 3 | ✓ |
| directionality (weak preference — scoped down) | MECHANISM_MULTISEED | 3 | ✓ |
| supporting: upstream-precip, lag sweep | RESULTS/JOURNAL | 1 | ✓ as supporting only |

**THE ONE EXPERIMENT TO RUN (recommended, before drafting Results):**
3-seed the static-topology 2×2 — train `L_T`, `L_noID`, `L_noID_T` at seeds 13 and 17 (L already
present). **6 runs, ~40 min each, turnkey** (run_2x2.py takes --seed, topology feature file exists,
L baselines at 13/17 present). Rationale: C1a is HALF the headline contrast (static null vs dynamic
gain); it is the only single-seed load-bearing result. A null is unlikely to flip, but the asymmetry
(this one single-seed, all else triple) is a reviewer flag the fix removes for 40 min. **If not run:**
defensible as an honest limitation, but a concession we don't need to make.

Supporting single-seed results (precip, lag sweep) do NOT need multi-seeding — they are secondary
color, not load-bearing, and reported as such.

---

## Results section structure (field-calibrated)

Lead each subsection with the QUESTION it answers; interpret tables, never paste them. Median across
basins is the headline summary. Add one CDF figure (field norm).

### R1 — Static topology is inert; dynamic upstream flow helps (the contrast)
- The 2×2: L 0.653, L+T 0.654 (Δ −0.001), L_noID 0.633, L_noID+T 0.625 (Δ +0.003). Static position
  adds ~0 with or without the one-hot. [RESULTS.md]
- Then the turn: oracle L+upQ +0.038 (p=2.6e-17), realizable +0.025. [PAPER_TABLE]
- The sentence: same network, opposite outcomes → structure-as-flow, not structure-as-label.
- **Fig 1** (core-idea diagram) referenced here.

### R2 — The gain is real and deployable
- Realizable +0.025, 3 seeds all positive; vs shuffled null: +0.017, p=2.3e-12, bootstrap CI
  [+0.011,+0.022]. [SIGNIFICANCE] Recovers ~55–72% of oracle. [MULTISEED]
- Beats no-ML routing: R1 +0.324, R2 +0.664±0.008, realizable +0.683±0.008. [ROUTING_BASELINE_3SEED]
- Not baseline-rescue: gain persists on well-predicted basins (L NSE>0.6): +0.012. [COMPLIANCE]
- **CDF figure** of per-basin NSE (L vs realizable vs oracle) here — the canonical field figure.

### R3 — The mechanism is routing, specific to the real network
- Depth gradient: 0.002/0.020/0.031/0.044 at depth 0/1/2/3; significant at depth≥1, absent at
  headwaters. [DEPTH_SIGNIFICANCE] Confound-checked vs area + feature magnitude. [CONFOUND, FEATURE_MAG]
- Topology-specificity (THE headline mechanism): forward +0.046 > reversed +0.031 > random +0.012;
  forward−random +0.034, p=2.3e-14, 3 seeds. Random rewire retains ~26%. [MECHANISM_MULTISEED]
- Directionality: honest one-paragraph scope — mild aggregate preference, seed-fragile, NOT a claim.
- **Position vs Kirschstein**: their adjacency×orientation sweep on learned GNNs → null; our same
  sweep on a fixed flow feature → real-edge signal survives. Quote their null verbatim.

### R4 — Robustness
- Graph density: k=2 realizable +0.025 ≈ full +0.026, 3 seeds. Not a heuristic-edge artifact. [MECHANISM §2]
- Metrics: realizable log-NSE +0.027 (all seeds +); KGE positive-on-average, seed-13 dip localized to
  variability ratio not timing. [METRIC_HONESTY]

---

## Figures/tables for Results (field-calibrated)
- **Fig 1**: core-idea contrast (static → 0 vs dynamic → gain). Conceptual diagram. [to draw]
- **Fig 2 (CDF)**: empirical CDF of per-basin test NSE, L vs realizable vs oracle. Canonical field
  figure (Kratzert Figs 3–5). Data: test_metrics.csv per condition. [to make from stored metrics]
- **Fig 3**: ΔNSE vs graph depth (the routing signature). Data: DEPTH_SIGNIFICANCE. [to make]
- **Table (main)**: conditions × NSE/KGE/logNSE × Δ-vs-L (p). [PAPER_TABLE Table 1]
- **Table**: mechanism controls (forward/reversed/random + k2). [MECHANISM_MULTISEED]

## Open items
- [ ] DECISION NEEDED FROM USER: run the 6-run 2×2 multi-seed now? (recommended) — else accept as
      limitation.
- [ ] Figs 1–3 to produce (CDF + depth are from stored data; Fig 1 is a diagram).
