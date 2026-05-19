# Pre-registration — Step 2: Basin-Encoding Ablation

**Status:** Pre-registered 2026-05-12, before any data is observed. *Conditional* on Step 1 outcome (gates whether this experiment is the right next step).
**Framework reference:** `testing_framework_proposal.md` §3, Step 2.
**Depends on:** Step 1 (`preregistration_step1.md`) — must complete before launching.

---

## Hypothesis

The 5-condition factorial showed `(G+T) − G ≈ −0.001` (paired NSE Δ): topology features are inert when the 671-dim basin one-hot encoding is also in the static input. The proposed mechanism is **informational redundancy** — the per-basin one-hot already perfectly identifies each basin and the LSTM can learn any per-basin response from it, so the 5 hand-designed topology scalars (0.7% of static input) carry no additional signal.

If we turn off `use_basin_id_encoding` (i.e., the LSTM no longer sees the basin ID as a one-hot), the topology features should become non-redundant and provide a measurable lift over the no-topology baseline.

## Pre-registered design

- **Architecture:** DirectedGraphLSTM with `edges=[]` (no message passing). Variant flag set to either `empty_graph` (G_no_oh) or `topology_features` (G+T_no_oh).
- **Data:** Component 0 (183 basins), Maurer forcings, 5 static attrs. **`use_basin_id_encoding: False`** in the NH config that the graph trainer pulls (changes the static input dim from 5+671=676 to just 5).
- **Conditions:** Two — `G_no_oh` and `G+T_no_oh`. Comparison is paired per basin × seed.
- **Seeds:** {11, 13, 17}. Three seeds for paired contrast with the existing G runs from 5cond.
- **Training:** matched budget per Step 1's outcome (default: same 30-epoch sweep as 5cond unless Step 1 reveals a budget issue requiring change).
- **Output dirs:** `runs/5cond_factorial/G_no_oh_seed{N}/`, `runs/5cond_factorial/G_T_no_oh_seed{N}/`.

## Success criterion

Paired per-basin median Δ NSE (`G+T_no_oh − G_no_oh`) **≥ +0.01** with bootstrap 95% CI excluding zero on the positive side.

This is the threshold at which we'd say "topology features have real signal." For reference: G+T − G in the original 5cond was −0.001 (CI [−0.002, +0.001]), so success is a *qualitatively different* result, not a small change.

## Falsification criterion

Paired per-basin median Δ NSE (`G+T_no_oh − G_no_oh`) **≤ +0.005** AND CI overlaps zero.

This would mean: the topology features are intrinsically weak — they have no signal even when given a chance to contribute. Need architectural redesign of the topology-feature pathway (per `architecture_analysis.md` §2: embed discrete features, replace network-relative with absolute attributes, etc.) before any further test.

## Pre-committed secondary observation

Also report:
- `G_no_oh` median NSE vs `G` median NSE — how much does removing the basin one-hot cost the no-topology baseline? Quantifies what the one-hot was buying us.
- `G+T_no_oh` median NSE vs `L` (full cudalstm) — does the no-one-hot graph variant catch up to or beat the standard baseline?
- Per-depth breakdown of (G+T_no_oh − G_no_oh): does topology help more for deeper basins (intuition: deeper basins need network-position signal more)?

## Pre-committed null control

Train `G+T_no_oh` with the topology features SHUFFLED across basins (so the topology row that says "depth=3 basin" is randomly assigned to a depth-1 basin, etc.). If shuffled-topology gives the same NSE as real-topology, the features are noise the model is treating as drag-along — confirms falsification.

## Pre-committed robustness check

Re-run seed 11 with `use_basin_id_encoding: False` but ALSO drop the 5 base static attrs (elev_mean, area_gages2, slope_mean, p_mean, pet_mean). This produces a model that sees ONLY dynamic forcings + topology features. If topology gives ≥ +0.02 NSE lift in this maximally-handicapped variant, that's stronger evidence the features have substantive signal.

## Compute estimate

- 6 main runs (2 conditions × 3 seeds) at ~30 min each on T4 = **~3 hr**.
- 1 null-control run + 1 robustness run on seed 11 = **~1 hr**.
- Total Step 2: **~4 hr on Colab T4** or **~1.5 hr on L4**.

## Reporting protocol

Same as Step 1: append results section to this file post-run, update `RESULTS.md`, update `CURRENT_STATE.md` and `JOURNAL.md`, commit + push.

## Pre-committed paper-narrative implications

- **If hypothesis confirmed (success):** the 5cond G+T result is recharacterized as "topology features are redundant under basin one-hot encoding," and the paper's contribution direction shifts to "graph-based features outperform basic LSTM when fair comparison conditions hold (i.e., no per-basin one-hot)." This is a publishable and defensible finding.
- **If hypothesis falsified:** the topology features as designed don't carry signal. Either redesign them (architecture_analysis.md §2) or drop topology features from the paper's claim entirely. Refocus on message passing as the sole "added feature."

## Pre-committed *what we will not do*

- Will not iterate on the topology-feature design within this step (e.g., adding new features mid-run). Any redesign requires a new pre-registration.
- Will not change the seed set if results are noisy with 3 seeds. Will pre-register Step 2B with additional seeds explicitly.
- Will not introduce ensembling across `_no_oh` conditions.
- Will not compare against L without acknowledging the asymmetry (L still uses one-hot encoding in standard configuration).

---
