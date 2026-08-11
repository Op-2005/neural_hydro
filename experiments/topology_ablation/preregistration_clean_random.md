# Pre-registration — clean random rewire (removing a 4.3% contamination)

**Written before the clean-random runs.**

## The defect

`build_directionality_variants.py` draws random parents with
`choices = basin_arr[basin_arr != c]`. It excludes the basin itself but **not its true parents**, so
true edges recur by chance. Verified locally by re-running the draw deterministically at
`RNG_SEED = 42`: **27 of 624 random edges (4.3%) are true edges, affecting 23 of 150 connected
basins.**

## Why it matters more now than it used to

Under the retired topology framing, the random rewire tested "does the gain need the real graph?"
and the contamination was benign: it inflated the low arm, so the forward-versus-random contrast was
understated.

Under the current proximity framing the random rewire plays a different role. It is the **far end of
the distance axis** (~511 km mean edge length) and it anchors the claim that distant neighbours carry
little. Recurring true parents are *nearby* (~92 km), so the contamination injects short-distance
edges into the long-distance arm. The bias now runs the other way: it makes the far end look
**better** than it is, so the reported distance decay is shallower than the truth.

## Design

Redraw the random graph with `forbidden = {c} ∪ true_parents(c)`, mirroring
`build_distance_control.py`, keeping the same in-degree, the same RNG seed, and the same feature
construction. Train at seeds 11, 13, 17. The notebook prints the overlap count for both the old and
the new draw so the correction is auditable.

## Hypothesis

The clean control scores at or slightly below the contaminated one, and the distance decay steepens.

## Pre-registered read-out

Let `S` = (clean cross-seed mean Δ) − (contaminated cross-seed mean Δ), on connected basins.

- **Contamination was inflating the far end** (expected): `S < −0.002`. Update the random row
  everywhere it appears and note that the corrected decay is steeper.
- **Immaterial:** `|S| ≤ 0.002`. Report the clean number, and state that the contaminated one was
  within noise of it. This is a satisfactory outcome: it closes the defect without changing any
  conclusion.
- **Unexpected:** `S > +0.002`. The clean control scoring *higher* would mean excluding true parents
  helped, which no account predicts. Investigate before writing anything.

No outcome here overturns a headline claim. This experiment exists to remove a known defect from a
control the paper relies on, so that a reviewer who reads the builder does not find it first.

## Cost

3 runs x ~40 min on a T4 (~2 GPU-hours).
