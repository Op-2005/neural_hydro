# Can catchment nesting explain the nearest-gauge advantage?

`CONTAINMENT.md` stratifies on a max/min area ratio $<2$ proxy. That proxy is neither
necessary nor sufficient for nesting, and it is blind in exactly the direction that
matters: a small gauge sitting inside a much larger target passes the filter, because
their areas are dissimilar. This test replaces the proxy with a necessary geometric
condition.

## The instrument

Two catchments can nest only if the smaller basin's gauge lies inside the larger
catchment. Approximating the larger catchment by its equivalent-circle radius
$r=\sqrt{A/\pi}$, nesting requires

    d(i,j) < r_large.

Where $d(i,j) \ge r_{\text{large}}$, nesting is geometrically impossible for any
catchment shape, so no containment leak can exist. This needs only gauge coordinates
and areas from `camels_topo.txt`; no shapefiles are required.

The bound is conservative in the safe direction. A real catchment is elongated rather
than circular, so some pairs flagged as *possibly* nested are not nested in fact. The
geometrically-clean stratum is therefore a strict subset of the truly-unnested pairs.

## Exposure of the two arms

| selector | pairs where nesting is geometrically possible |
|---|---|
| nearest 2 gauges | 22/300 (7.3%) |
| true upstream edges | 6/624 (1.0%) |

The nearest-gauge selector is more exposed, as expected — it optimises for proximity,
and proximity is a precondition for nesting. But the absolute exposure is small. The
area proxy's 40.7%-vs-21.0% figure overstates the threat by roughly six-fold, because
most area-flagged pairs are merely *similar-sized* rather than nested.

The two instruments disagree substantially: 91 of 150 basins are area-flagged but
geometrically clean, and 8 are geometrically flagged while the area proxy passes them.
Those 8 are the blind spot the proxy was criticised for, and they are now covered.

## The advantage on geometrically-clean basins

Paired kNN2-minus-network on the connected basins, per seed, weakest seed judged.

| stratum | basins | per-seed median | weakest-seed $p$ |
|---|---|---|---|
| **nesting geometrically impossible** | **130** | **+0.0287 / +0.0261 / +0.0402** | **$1.0\times10^{-6}$** |
| nesting geometrically possible | 20 | +0.0083 / +0.0514 / +0.0440 | $0.27$ |

**Containment does not account for the advantage.** On the 130 basins where nesting is
geometrically impossible — 87% of the connected set — the nearest-gauge input beats the
true network at every seed with weakest-seed $p=1.0\times10^{-6}$, and the effect size
(+0.032 cross-seed mean) matches the full-set value (+0.033). The 20-basin flagged
stratum is underpowered and does not reach significance on its own.

This supersedes the area-ratio stratification of `CONTAINMENT.md` as the primary
containment control: it tests a necessary condition rather than a correlate, it covers
the asymmetric small-inside-large case, and it retains 130 basins rather than 47.

## Scope

The equivalent-circle radius is an approximation to catchment extent. True
boundary-overlap fractions from the CAMELS shapefiles would measure realised overlap
rather than bound its possibility, and remain the stronger test. The shapefiles are not
distributed with the attribute package used here.
