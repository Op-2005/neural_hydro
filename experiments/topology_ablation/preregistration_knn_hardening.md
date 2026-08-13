# Pre-registration — hardening the nearest-gauge headline

Written before either experiment runs. Both target the paper's current headline: discarding the
river network and averaging each basin's two nearest gauges gives +0.081 against the true network's
+0.043 (paired +0.034, p=1.4e-24, better at all three seeds).

## Experiment 1 — minimum-separation k-NN

**The objection.** k-NN optimises purely for distance, so it can select a gauge a few km away.
Verified on disk: 19 of 366 pairs at k=2 are under 10 km, the nearest 1.6 km. Such a pair is
plausibly nested or near-duplicate, and its discharge is then partly a measurement of the target.
A reviewer will raise this, and it is the single objection that can sink the headline.

**Design.** Rebuild the k=2 input with a floor on neighbour distance at 10 km and 15 km, changing
nothing else. Verified locally: both build 183 basins with a 'date'-named index, and mean neighbour
distance rises from 46.7 km to 48.5 km (10 km floor) and 50.9 km (15 km floor). Seeds 11/13/17.

**Read-out.** Let `K_m` be the cross-seed mean gain at floor m and `G = +0.043` the true network.
- **Holds (expected):** `K_15 > G + 0.005`. The objection is closed; report the floored number
  alongside the unfloored one.
- **Falsified:** `K_15 < G`. The advantage was substantially near-duplicate leakage. The headline
  must be withdrawn and the paper rewritten around the distance dose-response instead.
- **Partial:** advantage shrinks but survives. Report the floored value as the headline.

## Experiment 2 — deployable k-NN

**The gap.** The k-NN comparison uses observed neighbour discharge on both sides, so it is
oracle-to-oracle. It shows distance selects better neighbours than topology; it is not yet a model
anyone can run.

**Design.** Build the two-stage version: stage one predicts each basin's discharge from its own
forcings over the full record, and those predicted series are averaged over the two nearest gauges,
using each seed's own stage-one model exactly as the network-derived deployable input does. Seeds
11/13/17.

**Read-out.** Let `D_knn` and `D_net = +0.022` be the deployable gains.
- **Survives:** `D_knn > D_net`, positive at all seeds. The headline becomes a deployable claim.
- **Does not survive:** `D_knn <= D_net`. Then nearest-gauge selection helps only when neighbour
  discharge is observed, which is itself informative: it would mean stage-one prediction error is
  larger for the nearer gauges, and the paper reports k-NN as an oracle bound.

Either outcome is reportable. Neither is a rubber stamp.

## Cost
9 runs x ~40 min on a T4 (~6 GPU-h).
