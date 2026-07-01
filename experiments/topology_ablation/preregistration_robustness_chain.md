# Pre-registration — Post-Multiseed Robustness Chain

**Pre-registered 2026-07-01, before running the analyses below.**
**Context.** Multi-seed (11/13/17) confirmed the realizable upstream-Q gain: predicted-Q Δ
vs L = +0.022 cross-seed mean, all 3 seeds positive (bar +0.015) → SUCCESS. But the
shuffled-Q null control crept from −0.002 (seed 11) to +0.004 cross-seed, so the honest
effect size is realizable-over-null, and two mechanistic robustness checks remain before
the story is paper-ready.

## Step A — Clean effect = realizable − null (CPU, re-analysis of existing runs)

**Hypothesis.** Predicted upstream Q beats a same-distribution shuffled control, i.e. the
gain is upstream *content*, not added capacity.
- **Success:** paired cross-seed mean (L+upQ_pred − L+upQshuf) ≥ +0.010, all 3 seeds positive.
- **Falsify:** ≤ +0.005 → gain is partly capacity; must be reframed.

## Step B — Depth-stratified gain (gated on A; CPU, re-analysis)

**Hypothesis.** Upstream discharge helps *downstream* basins more (mechanistic: routing).
Realizable gain should increase with graph depth.
- **Success:** median realizable Δ at depth ≥ 2 > median Δ at depth 0 (headwaters) by ≥ +0.01.
- **Falsify:** flat or inverted across depth → effect is not "upstream routing"; weakens the
  mechanistic claim (gain may be a generic extra-input effect).

## Step C — Local-subgraph scale curve (gated on B; Colab/GPU — pre-register only)

**Hypothesis.** The realizable gain grows on small locally-coherent networks (one upstream
basin is a larger share of the signal).
- Run L / L+upQ_pred on 3–4 walker subgraphs (13–23 basins), single seed, measure Δ vs
  network size.
- **Success:** Δ on subgraphs (<25 basins) > Δ on component0 (183) → "structure helps more
  at local scale," a strong secondary figure.
- Cost: ~30 min Colab T4. NOT run this session.

## What we will NOT do
- Will not drop seed 17 (the weaker seed) — report all three.
- Will not re-tune to chase thresholds.
- Steps A/B are re-analysis of already-committed runs; no new training.

---
## Results (post-run, 2026-07-01)

**Step A — realizable − null: PASS.** +0.0115 cross-seed (seeds 13/17: +0.015, +0.008), all
positive (bar +0.010). Gain beats a same-distribution shuffle → capacity artifact ruled out.

**Step B — depth-stratified: PASS (strong).** Realizable Δ rises monotonically with graph
depth: depth0 −0.003, depth1 +0.019, depth2 +0.029, depth3 +0.034. depth≥2 vs headwater diff
+0.032. **The routing signature** — downstream basins (more upstream contribution) benefit
more; headwaters (no upstream) get zero. Mechanistic confirmation, not a generic input effect.

**Step C (scale curve) — pre-registered, queued for Colab.**
