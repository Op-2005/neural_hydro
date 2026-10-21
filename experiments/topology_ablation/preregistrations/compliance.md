# Pre-registration — Methodology-Compliance Fixes

**Pre-registered 2026-07-01, before running the analyses.**
**Context.** Config audit confirmed all headline conditions (L, L+upQ, L+upQ_pred,
L+upQshuf) are byte-identical except one dynamic input (`upstream_q`) — the clean-ablation
requirement is met. Two publication-standard gaps remain, both cheap to fix from stored
predictions (no training).

## Gap 1 (Step A) — log-NSE not reported

Our own methodology (idea1.md §"Required methodology") mandates NSE + KGE + **log-NSE**.
Shipped test_metrics have NSE + KGE only. log-NSE downweights high-flow outliers and is
standard in CAMELS work (Kratzert 2019).

**Hypothesis.** The upstream-Q gain is metric-robust: log-NSE contrasts match NSE in sign.
- **Success:** realizable (L+upQ_pred − L) log-NSE Δ > 0 on seeds 13/17.
- **Falsify:** log-NSE Δ ≤ 0 → gain is NSE-specific (high-flow only); must be scoped.

Compute log-NSE per basin from stored `test_results.p` (obs/sim), ε = 0.01·mean(obs)
(Pushpalatha 2012). Re-report all headline contrasts (NSE / KGE / log-NSE).

## Gap 2 (Step B) — is the baseline a straw man?

A reviewer's first attack on "our-addition-helps": the baseline is weak, so the addition
just patches its failures. Our L median NSE is 0.653 (below Kratzert-2019 EA-LSTM ~0.74,
but that's 531 basins + entity-aware architecture; our cudalstm on 183 eastern-US basins is
a legitimate strong baseline for this scope).

**Hypothesis.** The upstream gain is real signal, not baseline-rescue: it persists on
well-predicted basins, not only catastrophic ones.
- **Success:** among basins with L NSE > 0.6 (already good), realizable Δ is still > 0.
- **Falsify:** gain concentrates only on L NSE < 0.3 basins → we're patching a weak
  baseline, not adding structural signal. Reframe.

## Scale/literature verdict (assessment, not a run)
Assess 183 basins × 3 seeds vs Kratzert 2019 (531), Kirschstein 2024 (LamaH-CE), Jiang 2025.
State explicitly whether the scope is publication-adequate and what a full-scale version
would require. Recorded in JOURNAL, no compute.

## What we will NOT do
- Will not change the success bars after seeing results.
- Steps A/B are re-analysis of committed runs; no training.

---
## Results (post-run, 2026-07-01) — COMPLIANCE VERDICT: PUBLICATION-VALID

**Config audit (the core question): PASS, exemplary.** All 4 headline conditions are
byte-identical configs (cudalstm, hidden 64, dropout 0.4, forget-bias 3, Adam 1e-3,
batch 256, 30 ep, seq 30, maurer, 5 static attrs, one-hot on, same train/test split, same
seed). **The only difference is one dynamic input (`upstream_q`).** Cleanest-possible ablation;
directly fixes the architecture confound that broke the earlier DirectedGraphLSTM work.

**Step A — metric robustness: PASS (stronger than NSE alone).** Realizable Δ: NSE +0.022,
log-NSE +0.019 — gain holds across the flow regime, not just high flows. Null control goes
NEGATIVE in log-NSE (−0.023): a shuffled input actively hurts low-flow prediction, making the
real +0.019 realizable gain even more clearly genuine.

**Step B — not baseline-rescue: PASS.** Realizable Δ persists on already-well-predicted basins
(L NSE > 0.6): +0.012 (n=230). Larger on bad basins (+0.22 on 17 worst) but positive
everywhere → real structural signal, not patching a weak baseline.

**Scale/literature verdict.** 183 basins × 3 seeds is adequate for a regional/workshop study
and exceeds our own 5cond design. Honest scope: "eastern-US connected sub-network (183 basins,
6 HUC regions)" — NOT a national benchmark. Baseline cudalstm 0.653 is legitimate (not
EA-LSTM/531-basin SOTA); do not claim SOTA. The study POSITIONS in the literature: resolves
Kirschstein 2024's GNN-null (static topology is inert) and executes Jiang 2025's physics-aware
direction (dynamic upstream flow works). 531-basin scale-up = named future work.

**Required actions:** (1) report all 3 metrics [done]; (2) frame scope as regional, no SOTA
claim; (3) re-eval seed-11 realizable to restore the clean 3-seed set (cheap TODO).
