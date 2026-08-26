# Pre-registration — does the nearest-gauge advantage survive a SINGLE neighbour?

Written before the k=1 runs. The k=1 feature is built and verified (183 basins, 'date' index,
mean separation 37.3 km) but untrained.

## The gap this closes

The nearest-gauge input averages k=2 neighbours; the true network averages a mean in-degree of
4.16. Averaging more series reduces variance regardless of separation, so neighbour count is a
candidate explanation for the +0.034 advantage.

`analysis/INDEGREE_CONFOUND.md` addresses this by stratifying on the network's in-degree: where the
network supplies at most two parents, the advantage is +0.028/+0.033/+0.039 (weakest-seed p=0.001,
n=57), with no rank correlation between in-degree and advantage (r_s=-0.07).

**That is necessary but not sufficient.** The stratification holds count fixed on the *network*
side only. In the in-degree<=2 stratum the nearest-gauge input still averages two series, so a
variance-reduction advantage of 2-over-1 is not excluded. Only an arm that averages a SINGLE
neighbour removes averaging from the comparison entirely.

A swept-distance arm cannot substitute for this: the substitution control preserves in-degree, so
its achievable floor is ~76 km against kNN2's 46.7 km (verified by dry-run). Distance and count are
entangled in that design. k=1 is the only manipulation that isolates count.

## Design

`L_upQknn1`: Eq. (feature) with P(i) replaced by the single nearest non-parent gauge, area weighting
and lag-1 unchanged. Seeds 11/13/17. Feature already built and schema-verified.

Mean separation 37.3 km (k=1) against 46.7 km (k=2) and 91.6 km (true edges), so k=1 is both
*closer* and *less averaged* — the two effects push in opposite directions, which is what makes the
test informative.

## Read-out

Let `A1` = cross-seed mean paired advantage of kNN1 over the true network on connected basins,
judged by the weakest seed as elsewhere.

- **Count is not the mechanism (expected):** `A1 > 0` at every seed with weakest-seed p < 0.05.
  Averaging is ruled out entirely; the advantage survives on one series. Combined with the
  in-degree stratification this closes the confound.
- **Count contributes materially:** `A1` falls well below the k=2 advantage (+0.034) and toward
  zero. Then part of the headline is variance reduction from averaging, and the paper must say so
  and report the k=1 value alongside k=2.
- **Falsification of the distance reading:** `A1 <= 0`. The single nearest gauge is the closest
  possible neighbour, so if proximity drives the gain it should be strongest here. A null or
  negative result would mean something other than distance is operating and the mechanism section
  must be reopened.

Note the arms are not nested: k=1 is closer AND less averaged, so a value between 0 and +0.034 is
genuinely ambiguous between the two accounts and should be reported as such rather than resolved.

## Cost
3 runs x ~40 min on a T4 (~2 GPU-h). Feature is prebuilt, so the notebook trains only.
