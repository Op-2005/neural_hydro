# Dynamical-Systems Probes

Outputs from the experimental program designed to test the dynamical-systems
framing (originating from the 2026-04-21 PI meeting): the trained LSTM
exhibits self-stabilizing dynamics, and external information like graph
topology helps only when it can break that self-stabilization.

Two probe families:

## `e0/` — Self-stabilization verification

The gate experiment for the framing. Two probes:

- **Probe A — perturbation recovery.** Inject Gaussian noise into the
  trained LSTM's hidden state mid-rollout. Measure how fast the perturbed
  prediction trajectory rejoins the unperturbed one.
- **Probe B — forcing replacement.** Replace forcing at a single timestep
  with a synthetic alternative (t-1's forcing, zero-out, random historical
  day). Measure how prediction-space deviation propagates.

Files include canonical run outputs, multi-seed verification across 6
seeds, σ-magnitude sensitivity, weak-baseline (run 03) replication, t=29
(no-recovery-time) variant, and state-space recovery measurement. Result:
**self-stabilization confirmed** — across all probes, the LSTM exhibits
contracting dynamics in both prediction and state space within 1–5
timesteps of perturbation.

See `e0/NOTES.md` for the curated narrative.

## `e0_5/` — Loss-saturation curve

A 60-epoch retrain of the strong baseline (vs the pilot's 30) to verify
the pilot's training loss had saturated. Plus a multi-seed (5-seed)
extension that revealed cross-seed variance in baseline plateau NSE
(0.111 NSE spread, larger than the pilot's +0.078 headline). This was
the finding that flagged the pilot's +0.078 as multi-seed-contingent
*before* we even ran scaled experiments.

Files: per-run loss-saturation plot, multi-seed band plot, and a
decision_record.json summarizing per-seed plateaus and cross-seed
disagreement.

## How these fit in the overall narrative

E0 + E0.5 results were the final empirical evidence supporting the
dynamical-systems framing on the pilot scale. They informed the locked
publication protocol in `idea1.md` (matched-from-scratch training, 5-seed
multi-seed, etc.). When the scaled A/B/C run (`../abc_component0/`)
showed the pilot's +0.078 did not generalize, these probe results
remained valid as descriptions of LSTM behavior — the framing stands as
*interpretation* even when the headline numbers move.
