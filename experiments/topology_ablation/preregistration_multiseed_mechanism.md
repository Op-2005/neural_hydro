# Pre-registration — Multi-Seed Consolidation of the Mechanism Experiments

**Date:** 2026-07-27. **Author:** /crs-unleashed session.
**Compute:** GPU (Colab T4), ~40 min total. Extends the two single-seed *mechanism* results to
seeds 13 and 17, so every load-bearing claim in the paper is multi-seed.

## Why this experiment (necessity, not polish)

The **core** claim (realizable upstream-flow gain, +0.022) is already 3-seed, significant
(p=2.3e-12), all-positive. It needs nothing more. But the two experiments that make this a
**mechanism** paper — the ones that engage the rebutted literature — are single-seed (seed 11):

1. **Directionality / topology-specificity** (`L_upQrev`, `L_upQrand`) — the Kirschstein mirror.
2. **k=2 graph-robustness** (`L_upQ_k2`, `L_upQpred_k2`) — the over-connectivity defense.

A single-seed mechanism claim is a specific, correct reviewer objection ("error bars on key
results"). Moreover the directionality result has an **unresolved** number: forward−reversed
paired Δ = +0.008, p=0.19 at seed 11 — at n=1 we cannot say whether the directional preference is
a real small effect or noise. This must be resolved before writing, either way.

## Conditions to add (seeds 13, 17)

All prerequisites verified on disk (L baselines + per-seed fullspan evals present; k=2 edges
committed; directionality features are observed-Q hence seed-independent to build).

| Condition | feature | per-seed dependency |
|---|---|---|
| `L_upQrev_seed{13,17}` | reversed-edge observed-Q | L_seedN baseline (Δ ref) |
| `L_upQrand_seed{13,17}` | random-rewire observed-Q | L_seedN baseline |
| `L_upQ_k2_seed{13,17}` | k=2 observed-Q (oracle) | L_seedN baseline |
| `L_upQpred_k2_seed{13,17}` | k=2 **predicted**-Q (realizable) | `_Lfullspan_eval_seed{N}` (present) |

## Success / falsification (pre-registered)

**Directionality (the unresolved one):**
- Report cross-seed mean ± std of forward Δ, reversed Δ, random Δ, and the paired forward−reversed
  and forward−random per basin×seed (pooled Wilcoxon).
- **Topology-specificity holds** if forward−random stays positive and significant across seeds
  (expected — seed 11 was p=3e-4). This is the headline; high prior.
- **Directionality resolution (the actual question):** is pooled forward−reversed significant
  (p<0.05) across 3 seeds, or does it stay n.s.?
  - If **significant** → upgrade the claim to "direction-sensitive" (the stronger mirror).
  - If **still n.s.** → keep the honest "directionally-preferential, not strictly directional"
    framing from seed 11. Either outcome is reportable; NEITHER is a failure — this run *resolves*
    a claim, it does not gate the paper.

**k=2 graph-robustness:**
- **Holds** if the k=2 realizable Δ stays positive and within ~±0.010 of the full-graph realizable
  across all 3 seeds (seed 11 was +0.021). Confirms the over-connectivity defense is not a
  single-seed artifact.
- If k=2 realizable **collapses** at 13/17 (negative or ≪ full-graph): the model-level graph
  robustness was seed-fragile → report honestly, scope back to the R1-proxy result.

## Discipline

- Pre-registered before execution. Amend only by dated append.
- Observed-Q for reversed/random/oracle-k2 (seed-independent build); predicted-k2 uses the
  per-seed fullspan eval. Feature index named 'date'.
- Random rewire uses the SAME fixed RNG seed (42) as seed 11 — the graph is identical across
  training seeds, so only the LSTM init/training seed varies (correct: isolates training noise).
- Single notebook, idempotent (skip completed runs), run-all. A falsification is reported.
- This is the LAST pre-writing consolidation. No 4th seed, no re-running the already-3-seed core,
  no national scale-up (workshop-tier decision, logged 2026-07-26).
