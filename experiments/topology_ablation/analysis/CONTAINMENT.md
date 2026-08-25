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

| basins | n (distinct) | per-seed median | weakest-seed $p$ |
|---|---|---|---|
| nearest pair NOT area-nested | 47 | +0.0241 / +0.0145 / +0.0412 | $0.021$ |
| nearest pair area-nested | 103 | +0.0278 / +0.0410 / +0.0410 | $0.000$ |

**Containment inflates the advantage but does not account for it.** Among the 47
basins whose two nearest gauges are not area-nested, the nearest-gauge input still beats
the network at every seed (+0.0241 / +0.0145 / +0.0412, weakest-seed
$p=0.021$). The advantage is larger where nesting is present
(+0.0366 cross-seed mean), which is the direction containment predicts, so the
effect is real and partly inflated rather than wholly artifactual. The weakest-seed
$p$ is judged by the rule the paper applies elsewhere; pooling the 141 basin-seed pairs
would give $4\times10^{-8}$ and would treat dependent observations as independent.

Scope: area ratio is a proxy. Catchment-boundary overlap from the CAMELS shapefiles would
measure containment directly and is the stronger test.
