# Pre-registration — Distance-Preserving Graph Control (experiment #1)

Written before any training. Closes the proximity confound in the headline mechanism result.

## Motivation

The paper's headline mechanism claim is topology-specificity: the upstream-flow gain requires the
**real** river network. The evidence is `true edges (+0.046) >> random rewire (+0.012)` on connected
basins (oracle, 3 seeds, MECHANISM_MULTISEED.md). A reviewer can object that the degree-preserving
**random** rewire destroys *two* things at once — the true topology **and** spatial proximity. On
this basin set the random edges average **511 km** (basins span the eastern US) versus **92 km** for
true edges, so the random condition may fail simply because its "parents" are far away and share no
weather, not because the topology is wrong.

## The control

A **distance-preserving** rewire: substitute each true edge `(p -> c)` with `(q -> c)` where `q` is
the available basin whose distance to `c` is closest to the true edge length, `q` is not a true
parent of `c`, and in-degree is preserved exactly. Proximity is held nearly fixed while the true
topology is destroyed.

Builder + local validation: `build_distance_control.py` (dry-run verified on committed inputs):
- in-degree preserved exactly: **True**
- rewired edges identical to a true edge: **0%** (topology fully destroyed)
- edge length: distctrl mean **101 km** (median 103) vs true **92 km** (median 96) vs random **511 km**

The control sits right next to the true graph in proximity and nowhere near the random rewire.

## Hypothesis

The gain is topology-specific, not proximity-driven. Holding proximity fixed but destroying the true
topology removes most of the gain: distctrl behaves like the random rewire, not like the true graph.

## Conditions

`L_upQdistctrl` = stock L + oracle-style observed upstream Q aggregated over the distance-preserving
graph (area-weighted mean, lag 1), byte-identical config otherwise. Seeds 11/13/17, 30 epochs.
Compared, paired per basin on **connected basins**, against the existing true/reversed/random oracle
conditions.

## Pre-registered outcomes (paired median oracle Δ vs L, connected basins, 3 seeds)

- **Confirms topology-specificity (predicted):** distctrl retains little of the gain — its Δ is close
  to random (~+0.012) and well below true (+0.046), i.e. `true - distctrl` is large and
  `distctrl - random` is small and not significant. The random-edge objection is answered: proximity
  alone does not recover the gain.
- **Falsifies / weakens (must report if seen):** distctrl ≈ true (retains most of the gain). Then the
  effect is driven by proximity, not the specific upstream topology, and the topology-specificity
  claim must be reframed to "proximity-specific."
- **Ambiguous:** distctrl sits roughly midway (e.g. +0.025 to +0.035). Report as partial: proximity
  carries some of the signal, true topology adds the rest.

## Known limitation (report regardless)

Short true edges (7% are <30 km) cannot be distance-matched — there are no nearby non-parent basins —
so those substitutes are longer than their true counterparts (per-edge |Δdistance| median ~36 km) and
distctrl's mean edge length (101 km) is ~10% above true (92 km). This makes the control mildly
**conservative**: distctrl is if anything slightly *less* proximate than true, so a distctrl gain that
still falls to the random level is strong evidence for topology-specificity. It remains 5x more
proximate than the random rewire.

## Compute

3 training runs (L_upQdistctrl x seeds 11/13/17), stock cudalstm, 30 epochs, 183 basins. ~40 min each
on a T4, ~2 GPU-hours total. Idempotent: skips any seed already complete. Not required for a workshop
submission; it is a strengthening control that pre-empts the proximity objection.
