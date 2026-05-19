# Idea 2 — Temporal-lag message passing as a fix for directional insensitivity (SET ASIDE)

*Set aside 2026-04-20 in favor of Idea 1 — see repository-root `idea1.md`.*

## One-sentence version

The DirectedGraph-LSTM's temporal lag `h_u(t-τ) → h_v(t)` is a time-domain analog
of Jiang et al.'s (ICML 2025) spatial gradient operator `D̂`; it should
preserve the high-frequency hidden-state content that standard symmetric-
aggregation GNNs (Kirschstein & Sun 2024) low-pass-filter away, producing an
NSE benefit that grows monotonically with river-network depth.

## Why it is interesting

- Positions our work *between* Kirschstein 2024 (null) and Jiang 2025 (positive,
  spatial-domain fix) as a distinct mechanistic contribution.
- Gives a falsifiable prediction the prior literature has not tested directly:
  per-basin ΔNSE vs. basin depth should be monotone-increasing.
- Reuses every past run as pilot evidence; nothing wasted.

## Why it is set aside

- The framing is mathematically elegant but harder to communicate — a reviewer
  (or collaborator) who has not internalized both Kirschstein's null and Jiang's
  low-pass diagnosis cannot follow the argument in a 15-minute conversation.
- The Apr 19–20 pilot on 23 basins was **scale-insufficient** to test the
  monotonicity prediction (n=2 at depths 2 and 3). The signal is directionally
  correct in the frozen-variant (depth 1: −10.6%, depth 2: −17.6% high-freq
  residual power) but below the pre-registered bar.
- Idea 1 (the simpler scaled-ablation framing) can be executed, defended, and
  published as a workshop paper on a shorter path. Idea 2 remains a candidate
  follow-up once Idea 1 is shipped.

## Artifacts preserved here

| File | Description |
|---|---|
| `HYPOTHESIS.md` | Pre-registered hypothesis, falsification conditions, evidentiary bar, Amendment 1 after the 23-basin spectral test. |
| `spectral_analysis.py` | Step-1 script that reloads runs 05/06/07, dumps per-window predictions, and computes Welch PSD of residuals stratified by depth. |
| `spectral_outputs/psd_by_basin.csv` | Per-basin low/mid/high frequency band power for each of the three models. |
| `spectral_outputs/high_freq_power_summary.csv` | Depth-stratified median high-freq residual power and graph-vs-baseline ratios. |
| `spectral_outputs/psd_by_depth.png` | Visual comparison of residual PSD across the three models, one panel per depth stratum. |
| `spectral_outputs/predictions.npz` | Cached [1461, 23] prediction arrays for baseline, graph+warm, graph+frozen. |
| `spectral_outputs/decision_record.json` | Machine-readable verdict against the pre-registered bar. |

## Resurrection criteria

Revisit Idea 2 if any of the following occur:
- Idea 1 ships and we want a follow-up contribution.
- Cloud compute arrives and we can run the Component-0 (183-basin) scaled
  version of the spectral test cheaply.
- A reviewer of the Idea 1 submission explicitly asks "what is the mechanism?"
  — Idea 2's framing is the mechanistic answer.

## Pointers back

- Master file for the current direction: `../idea1.md`
- Pilot-scale empirical findings that seeded Idea 2: `../INSIGHTS.md`
- Chronological log of experiments: `../CURRENT_STATE.md`
