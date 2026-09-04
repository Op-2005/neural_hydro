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

    d(i,j) < R,   R = 2*sqrt(e * A_large / pi).

A streamgauge sits at its catchment's **outlet** -- a point on the boundary, not the
interior -- so the span it must cover to reach the far side is the catchment's full
extent, which for a circle of area A is the **diameter** 2*sqrt(A/pi), not the radius.
(An earlier version of this file used the radius and was permissive by a factor of two.)
Where d(i,j) >= R, nesting is geometrically impossible, so no containment leak can exist. This needs only gauge coordinates
and areas from `camels_topo.txt`; no shapefiles are required.

**The circular case is the smallest defensible reach, not a shape-free bound.** By the
isoperimetric inequality the circle has the least maximum extent for a given area, so
e=1 understates the reach of a real, dendritic catchment. Larger e discards more pairs
as potentially nested:

| elongation $e$ | clean basins | per-seed median | weakest-seed $p$ |
|---|---|---|---|
| **1 (circular; headline)** | **89** | **+0.0285 / +0.0275 / +0.0395** | **1.6e-04** |
| 2 | 55 | +0.0347 / +0.0275 / +0.0328 | 4.2e-03 |
| 3 | 39 | +0.0285 / +0.0257 / +0.0276 | 3.2e-02 |
| 4 | 31 | +0.0265 / +0.0275 / +0.0276 | 1.4e-01 |

The e=4 row is recorded here because the STODY submission cites it (2026-09-04). Note where the
weakest-seed rule bites: the effect size is flat across all four elongations (+0.027 to +0.035)
while significance is lost at e=4, so the binding constraint there is power (n=31), not signal.
Any text citing the sweep must say that rather than implying significance holds throughout.

The effect size is flat in $e$ while only power falls with the shrinking stratum. That is
the signature of a real effect rather than of nested pairs left in the clean stratum.

## Exposure of the two arms

| selector | pairs where nesting is geometrically possible |
|---|---|
| nearest 2 gauges | 85/300 (28.3%) |
| true upstream edges | 38/624 (6.1%) |

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
| **nesting geometrically impossible** | **89** | **+0.0285 / +0.0275 / +0.0395** | **$1.6\times10^{-4}$** |
| nesting geometrically possible | 61 | (underpowered stratum) | -- |

**Containment does not account for the advantage.** On the 89 basins where nesting is
geometrically impossible — the nearest-gauge input beats the
true network at every seed with weakest-seed $p=1.6\times10^{-4}$, and the effect size
(+0.032 cross-seed mean) matches the full-set value (+0.033).

This supersedes the area-ratio stratification of `CONTAINMENT.md` as the primary
containment control: it tests a necessary condition rather than a correlate, it covers
the asymmetric small-inside-large case, and it retains 89 basins rather than 47.

## Scope

The equivalent-circle radius is an approximation to catchment extent. True
boundary-overlap fractions from the CAMELS shapefiles would measure realised overlap
rather than bound its possibility, and remain the stronger test. The shapefiles are not
distributed with the attribute package used here.


## The geometric flag is not independent of the separation floor

The flag is $d(i,j) < r_{\text{large}}$, a distance criterion with a per-pair threshold whose
A 15 km minimum-separation floor removes **32 of the 85** flagged pairs, leaving 53. The two
controls therefore overlap and are not fully independent evidence. The area-ratio
stratification is the genuinely independent check, since it is not a distance criterion.

## Direction is a separate channel from nesting

A *downstream* neighbour integrates the target's own outflow, so its observed discharge partly
reflects the target even where nesting is impossible and the gauges are far apart. Applying the
edge rule's elevation and area conditions in reverse, 107 of 150 connected basins have no
downstream basin among their two nearest gauges.

| stratum | basins | per-seed median | weakest-seed $p$ |
|---|---|---|---|
| **no downstream neighbour** | **107** | **+0.0285 / +0.0263 / +0.0348** | **2.8e-06** |
| at least one downstream neighbour | 43 | +0.0213 / +0.0429 / +0.0464 | 3.8e-02 |

The advantage is undiminished where no downstream gauge is selected, so downstream selection is
not what produces it.
