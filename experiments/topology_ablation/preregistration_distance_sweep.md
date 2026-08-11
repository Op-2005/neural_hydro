# Pre-registration — proximity dose-response (distance sweep)

**Written before any sweep run.** Dry-runs of the graph construction only.

## Why

The paper is titled around proximity, but proximity is measured at exactly two points: the
distance-preserving control at ~101 km (gain intact, +0.049) and the random rewire at ~511 km (gain
mostly gone, +0.012). Two points cannot distinguish

- a smooth decay with distance,
- a threshold somewhere in the 100–500 km gap, or
- a failure of the 511 km condition for a reason other than distance (the random rewire also
  randomises which basins are chosen, so distance is confounded with assignment).

A reviewer will lead with this: the paper's headline variable is measured at two levels, one of them
confounded. This experiment turns an argument by elimination into a measured curve.

## Design

`build_distance_control.py --target-km X` substitutes each true edge with a non-parent basin at
approximately X km, holding in-degree exactly and excluding all true parents. This isolates distance:
every arm uses the same substitution rule and the same in-degree, and only the target separation
changes. Dry-run validation (committed inputs, no training):

| target | achieved mean km | overlap with true edges |
|---|---|---|
| canonical (match true) | 100.6 | 0 / 624 |
| 175 km | 176.9 | 0 / 624 |
| 250 km | 249.6 | 0 / 624 |
| 350 km | 350.4 | 0 / 624 |
| 500 km | 499.5 | 0 / 624 |

Arms to train: **175, 250, 350, 500 km**, seed 11 only. The existing canonical control (~101 km,
+0.049) and forward (+0.046) supply the near end; the existing random rewire (~511 km, +0.012)
corroborates the far end.

## Hypothesis

The gain decays monotonically with substitute distance, from ~+0.049 near 100 km toward the random
level (~+0.012) by ~500 km.

## Pre-registered read-out

Let `Δ(X)` be the paired median gain vs `L` on the forward-connected basins at target X.

- **Dose-response confirmed:** `Δ(X)` decreases monotonically across 101 → 175 → 250 → 350 → 500 km
  (allowing one non-monotone step within noise), with `Δ(500) < Δ(101)`. Spearman correlation between
  target distance and `Δ` is negative. The paper then reports a decay curve and can state the length
  scale over which neighbour discharge is informative.
- **Threshold, not gradient:** `Δ` stays near +0.049 out to some distance then drops sharply. Also a
  publishable and more interesting result: it identifies a spatial scale.
- **Falsification — distance is not the operative variable:** `Δ(500) ≈ Δ(101)` (within ~0.005).
  Then the random rewire's failure was caused by something other than distance, the proximity framing
  is wrong, and the paper's central claim must be reopened. This is a real risk, not a rubber stamp:
  the random rewire confounds distance with random assignment, and only this sweep separates them.

## Interpretation limits, stated in advance

Single seed. This buys the shape of the curve, not per-point significance. If the curve is clean,
the two endpoint arms can be extended to 3 seeds later. Distance is straight-line gauge-to-gauge, so
the sweep varies geographic separation and not travel time or shared-storm exposure.

## Cost

4 runs x ~40 min on a T4 (~3 GPU-hours).
