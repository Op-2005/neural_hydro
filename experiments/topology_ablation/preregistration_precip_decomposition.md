# Pre-registration — shared weather vs shared water (upstream precipitation, 3 seeds)

**Written before seeds 13/17 are run.** Seed 11 already exists on disk (`L_upPrecip_component0_seed11`)
and is reported below as prior information, not as a result of this experiment.

## Why

The distance-preserving control falsified topology-specificity: the gain comes from spatially nearby
basins, not the true upstream network (`MECHANISM_MULTISEED.md`). That leaves the paper arguing by
elimination — not static, not topology, not direction, not depth — with "proximity" as an
uncharacterised residual. Two failed mediation analyses (headroom, baseflow index) are already on
record as failures to characterise it.

This experiment is the cheapest available *positive* decomposition. A neighbour's discharge could help
for two reasons:

1. **Shared weather.** Nearby basins experience the same storms. The baseline already receives its own
   precipitation, so an area-weighted mean of *upstream precipitation* is a spatial smooth of a channel
   the model has. If this recovers the whole gain, the contribution is regional forcing smoothing.
2. **Shared water / catchment state.** Discharge integrates soil moisture, baseflow, and snowmelt
   storage that precipitation does not carry. Whatever discharge adds *over* precipitation is this.

The contrast `realizable − upPrecip` isolates (2) from (1).

## Conditions

- `L` (baseline, on disk, 3 seeds)
- `L_upPrecip` — area-weighted mean upstream **precipitation**, lag 1. Seed 11 on disk; **seeds 13, 17
  are what this experiment runs.**
- `L_upQpred` (realizable, on disk, 3 seeds)

Feature: `features/upstream_precip_component0_lag1.p`, built by
`build_upstream_variants.py --variant precip --lag-days 1`. Column name is `upstream_q` so the config
is byte-identical to the other structural conditions apart from the feature file.

## Prior information (seed 11, connected basins n=150)

| condition | paired median ΔNSE vs L |
|---|---|
| `L_upPrecip` | +0.016 |
| `L_upQpred` (realizable) | +0.034 |

Precipitation recovers ~47% of the realizable gain at seed 11.

## Hypothesis

Upstream precipitation carries a real but partial share of the proximity gain. Discharge adds
information beyond shared weather.

## Pre-registered read-out

Let `P` = pooled 3-seed paired median Δ(upPrecip − L) and `R` = Δ(realizable − L), both on the
forward-connected basins (n=150/seed).

- **Discharge adds beyond weather** (expected): `R − P > 0`, paired Wilcoxon p < 0.05, and positive at
  all three seeds. Report `P/R` as the shared-weather share.
- **Falsification — the gain is just weather smoothing:** `P ≈ R` (paired `R − P` not significant, or
  `P/R > 0.85`). Then the deployable contribution is a spatial average of forcings the baseline already
  receives, the two-stage discharge model is unnecessary machinery, and the paper must say so.
- **Third outcome — precipitation beats discharge** (`P > R`): would mean the stage-one discharge
  prediction is actively harmful relative to using raw upstream forcings. Reportable either way.

## What gets reported regardless of outcome

The three-seed `L_upPrecip` row goes into the conditions table and the mechanism section, and the
shared-weather share `P/R` is stated in the Discussion. This is the decomposition the current
Conclusion promises as future work; after this run it is a result, not a promise.

## Cost

2 runs (seeds 13, 17) x ~40 min on a T4.
