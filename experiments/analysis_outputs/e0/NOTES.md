# E0 — Self-stabilization probes (outputs)

Outputs produced by `../../probes/e0_self_stabilization.py` on the run-05
strong baseline. Pre-registered in `../../../idea1.md` §E0.

## What E0 tested

The dynamical-systems framing in `idea1.md` rests on the claim that the
trained multi-basin LSTM exhibits *self-stabilizing dynamics*: its
rolled-out predictions are dominated by the model's own hidden-state
evolution, not by external forcings. E0 is the gate experiment for that
claim.

Two probes:
- **Probe A** — hidden-state perturbation recovery. At test-period
  timestep T=15 of a 30-step window, add Gaussian noise to the LSTM's
  hidden state. Measure how fast the perturbed prediction trajectory
  rejoins the unperturbed one.
- **Probe B** — forcing replacement. Replace the forcing at T=15 with
  the t-1 forcing. Measure prediction-space deviation.

## Result — PASS

Both probes 100% of basins on the pre-registered ≥ 50% bar. Robust to
perturbation magnitude:

| Probe | σ | Pass rate | Median recovery |
|---|---|---|---|
| A — perturbation recovery | 0.5 × natural h-std | 100% (23/23) | 1 step |
| A — perturbation recovery | 2.0 × natural h-std | 100% (23/23) | 2 steps |
| B — forcing replacement | n/a | 100% (23/23) | max dev 0.007 |

**Caveat on Probe B**: "replace forcing with t-1's forcing" is a
near-null replacement on most days (rainfall = 0 → 0; smooth
meteorology). The 0.007 median deviation is partly LSTM
self-stabilization and partly weak replacement. Probe A is the
diagnostic test; cite it primarily.

## Files

| File | Description |
|---|---|
| `decision_record.json` | Canonical (σ=0.5) decision record: pass rates, median recovery step, per-probe verdicts, machine-readable interpretation. |
| `decision_record_sigma_2_0.json` | Sensitivity check (σ=2.0) decision record. |
| `probe_a_recovery.png` | Canonical figure: median normalized prediction deviation by timestep relative to perturbation, both probes side-by-side. **This is the meeting-ready figure.** |
| `probe_a_recovery_sigma_2_0.png` | Same figure for the σ=2.0 sensitivity. |
| `probe_a_recovery.csv` | Per-basin Probe A details: recovery step, success flag, max post-T deviation. |
| `probe_b_forcing.csv` | Per-basin Probe B details: max post-T deviation, success flag. |

The `predictions.npz` cache that the spectral-analysis used to share
(idea2) is **not** needed here — E0 generates its own per-window
trajectories on the fly.

## Where this fits in the project narrative

E0 PASS is the empirical green-light for the dynamical-systems framing
that came out of the 2026-04-21 PI meeting (see `JOURNAL.md`). With this
in hand, the +0.013 / +0.065 decomposition of the pilot's +0.078 NSE
headline gets its mechanistic interpretation:

- +0.013 (frozen-graph isolation, run 07) = small **destabilizing
  forcing** the upstream messages provide.
- +0.065 (LSTM drift during joint training, run 06 vs run 07) = LSTM
  finding a **new self-consistent attractor** that incorporates the
  forcing more deeply.

Neither is a confound; they are two regimes of the same mechanism.

## Next gate

E0.5 — 60-epoch loss-saturation curve on the strong baseline. ~20 min
of CPU. After E0.5: forcing-comparison sub-experiment (C-rand,
C-precip, C-lag) per `idea1.md`.
