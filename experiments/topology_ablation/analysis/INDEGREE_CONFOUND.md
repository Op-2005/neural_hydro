# Is the nearest-gauge advantage an averaging-count artifact?

The nearest-gauge input averages $k=2$ neighbours while the network averages a mean
in-degree of 4.16. Averaging more series reduces variance regardless of distance, so
neighbour count is a candidate explanation for the advantage.

## Why a swept-distance control cannot settle this

The distance-substitution control preserves in-degree by construction. Holding in-degree
at the network's 4.16 puts a floor of roughly **76 km** on achievable mean separation
(54 km even ignoring the non-parent constraint), while nearest-gauge selection reaches
**46.7 km** precisely *because* it uses two neighbours rather than four. Distance and
count are therefore structurally entangled in this design: no degree-preserving arm can
match the nearest-gauge separation, so a sweep arm below 101 km cannot isolate distance.

## Stratifying the existing contrast on network in-degree

If count drove the advantage, it should shrink where the network already averages many
parents. Paired kNN2-minus-network on the connected basins, per seed, weakest seed judged.

| network in-degree | basins | per-seed median | weakest-seed $p$ |
|---|---|---|---|
| $\le 2$ (count matched to $k{=}2$) | 57 | +0.0276 / +0.0333 / +0.0395 | $0.001$ |
| $3$--$4$ | 42 | +0.0285 / +0.0316 / +0.0424 | $0.008$ |
| $\ge 5$ (network averages more) | 51 | +0.0171 / +0.0211 / +0.0366 | $0.005$ |

Rank correlation between the network's in-degree and the advantage is
**-0.068** ($p=0.15$): no trend. The advantage is positive at every seed in every
stratum, including where the network averages more parents than the nearest-gauge input.

**Neighbour count does not account for the advantage.** The count-matched stratum
($\le 2$ parents) shows it at full size, and it does not decay as the network's averaging
increases. Note the exact in-degree$=2$ stratum holds only 23 basins and is positive at
every seed but underpowered (weakest-seed $p=0.24$), so the count-matched evidence rests
on the $\le 2$ grouping rather than an exact match.

---

## Single-neighbour arm (2026-08-26) — averaging ruled out

The in-degree stratification above holds neighbour count fixed on the *network* side, but the
nearest-gauge input still averages two series inside every stratum, so a two-over-one variance
advantage is not excluded by it. The k=1 arm removes averaging entirely.

| condition | mean separation | per-seed median vs network | weakest-seed p |
|---|---|---|---|
| nearest 2 gauges | 46.7 km | +0.019 / +0.032 / +0.041 | <0.001 |
| **nearest 1 gauge** | **37.3 km** | **+0.019 / +0.036 / +0.024** | **0.002** |

**Averaging is not required for the advantage.** A single neighbour still beats the true network at
every seed. The k=1 arm sits ~0.004 below k=2, so averaging may contribute without being the
mechanism.

**The arms are not nested**, which the pre-registration anticipated: k=1 is both *closer*
(37.3 vs 46.7 km) and *less averaged* than k=2, and those push in opposite directions. The ~0.004
gap is therefore ambiguous between "averaging helps a little" and "the second neighbour adds
distance", and we report both values rather than resolving it.

**Provenance note.** These figures come from the Drive copy of the runs, which holds the *restored*
`L_upQ` seed-11 oracle (median NSE 0.6931) rather than the original local copy (0.7027) -- see
JOURNAL 2026-07-16. That shifts the seed-11 column of both arms by the same +0.008 and leaves the
k1-vs-k2 gap at -0.004 under either copy, so no conclusion depends on which is used.
