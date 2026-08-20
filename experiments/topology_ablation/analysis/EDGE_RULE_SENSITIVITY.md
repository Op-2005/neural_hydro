# Edge-rule sensitivity: is the distance profile an artifact of our thresholds?

The nearest-gauge result turns on the true edges averaging ~92 km while the two nearest
gauges average ~47 km. A reviewer may object that our area-ratio requirement
($A_i \ge \rho A_j$) is what pushes the selection outward. It is not.

## Mean true-edge length over the threshold grid

| $\rho$ | $D_{max}$ (km) | edges | mean edge length (km) |
|---|---|---|---|
| 1.0 | 50 | 140 | 32.7 |
| 1.0 | 100 | 445 | 63.2 |
| 1.0 | 150 | 811 | 91.6 |
| 1.2 | 50 | 127 | 32.7 |
| 1.2 | 100 | 392 | 62.8 |
| 1.2 | 150 | 719 | 91.7 |
| 1.5 | 50 | 110 | 32.8 |
| 1.5 | 100 | 342 | 63.3 |
| 1.5 | 150 | 624 | 91.6 |
| 2.0 | 50 | 91 | 33.3 |
| 2.0 | 100 | 275 | 62.8 |
| 2.0 | 150 | 493 | 91.0 |
| 3.0 | 50 | 69 | 34.8 |
| 3.0 | 100 | 202 | 63.5 |
| 3.0 | 150 | 361 | 91.1 |

**Mean edge length is invariant to $\rho$.** At $D_{max}=150$ km it is 91.6 km at $\rho=1.5$ and 91.6 km with the area filter removed entirely ($\rho=1.0$), despite the edge count rising from 624 to 811. Dropping the elevation filter gives 93.7 km over 1298 edges. Only $D_{max}$ moves the mean, and it does so by construction.

## Why the two selections differ

The two nearest gauges average 39.0 km. For only **62 of 150 (41%)** connected basins does the nearest pair contain a true parent.

The gap is therefore not a threshold artifact. Requiring a neighbour to lie *upstream* is
a directional constraint, and it excludes most of a basin's nearest gauges because they
are lateral or downstream rather than above it. Any edge set that encodes upstream-ness,
surveyed or inferred, inherits this: the nearest gauge is usually not the upstream one.

This is what the paper means by the drainage graph encoding a causal relation where the
model needs a statistical one. The finding does not depend on our particular $\rho$,
$D_{max}$, or elevation rule.
