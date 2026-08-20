# Does catchment nesting explain the nearest-gauge advantage?

CAMELS gauges nest. A neighbour whose catchment is largely contained in the target's is
partly a measurement of the target, which would inflate an oracle contrast specifically.
We use max/min catchment area $<2$ as a containment proxy, the same threshold the paper
applies to its own edges.

## The distance floors do not test containment

| selector | pairs flagged (area ratio $<2$) |
|---|---|
| nearest 2 | 149/366 (40.7%) |
| nearest 2, $\ge 10$ km apart | 143/366 (39.1%) |
| nearest 2, $\ge 15$ km apart | 146/366 (39.9%) |
| true upstream edges | 131/624 (21.0%) |

Distance and area-containment are nearly independent: a 15 km floor moves the flagged
fraction by under one point. The nearest-gauge selector is about twice as flagged as the
network selector it outperforms, so containment must be tested directly rather than
inferred from the separation floors.

## Stratifying the advantage by nesting risk

| basins | n (basin-seed) | median kNN $-$ network | $p$ |
|---|---|---|---|
| nearest pair NOT area-nested | 141 | $+0.0285$ | $4.3e-08$ |
| nearest pair area-nested | 309 | $+0.0353$ | $9.4e-19$ |

**Containment inflates the advantage but does not create it.** Among basins whose two
nearest gauges are not area-nested, the nearest-gauge input still beats the network by
**+0.0285** ($p=4.3e-08$, $n=141$
basin-seed pairs). The advantage is larger where nesting is present (+0.0353),
which is the direction containment predicts, so the effect is real and partly inflated
rather than wholly artifactual.

Scope: area ratio is a proxy. Catchment-boundary overlap from the CAMELS shapefiles would
measure containment directly and is the stronger test.
