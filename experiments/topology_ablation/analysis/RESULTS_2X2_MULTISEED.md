# Static-Topology 2×2 — 3-Seed Consolidation

Seeds 11/13/17, stock cudalstm, Component 0 (183 basins). Supersedes the single-seed `RESULTS.md`
2×2 for the paper. The 2×2 varies only `use_basin_id_encoding` (one-hot on/off) and the 5 topology
static features (graph_depth, n_upstream, total_upstream_area, in_degree, frac_upstream_area).
Pre-reg: `preregistration_upstream_signal.md` context + the original 2×2 pre-reg.

## Per-condition median NSE (per seed)

| condition | seed 11 | seed 13 | seed 17 |
|---|---|---|---|
| L (one-hot, no topo) | 0.653 | 0.651 | 0.656 |
| L+T (one-hot + topo) | 0.654 | 0.649 | 0.663 |
| L_noID (no one-hot) | 0.633 | 0.629 | 0.628 |
| L_noID+T (no one-hot + topo) | 0.625 | 0.628 | 0.614 |

## The two headline contrasts (paired per-basin, pooled 3 seeds, n=549)

| contrast | question | per-seed Δ (11/13/17) | pooled median | two-sided p |
|---|---|---|---|---|
| **L+T − L** | does topology help the identity-encoded model? | −0.001 / +0.006 / −0.002 | **+0.0016** | 0.67 |
| **L_noID+T − L_noID** | does topology help when identity cannot be memorized? | +0.003 / +0.009 / +0.006 | **+0.0057** | 0.28 |

**Verdict: static topology features are inert, confirmed across 3 seeds.** Both contrasts are
small and not significant. The pre-registered prediction (both ≈ 0) holds. Static network position
adds essentially nothing to a strong multi-basin LSTM, with or without the basin one-hot.

**Honest nuance (report, do not overclaim).** The without-one-hot contrast is consistently, faintly
positive (+0.006, positive at all 3 seeds), the direction the pre-registration entertained ("structure
helps most where the model cannot memorize identity"). It is far too small and non-significant
(p=0.28) to claim. The accurate statement is: static topology is inert; any effect of removing the
one-hot is a faint, non-significant hint, not a result.

## Bonus (now 3-seed): the value of the basin one-hot

| contrast | per-seed Δ (11/13/17) | pooled median | p |
|---|---|---|---|
| **L − L_noID** | +0.012 / +0.012 / +0.025 | **+0.0161** | 2.5e-8 |

The basin identity encoding is worth a real, significant +0.016 NSE — memorizing per-basin behavior
helps, as expected. This frames why static topology (a constant per basin) cannot compete with the
one-hot: the one-hot already captures fixed per-basin identity.

## Role in the paper
This is the **static-null half of the headline contrast** (static topology ≈ 0 vs. dynamic upstream
flow gain). Every load-bearing claim in the paper is now 3-seed.
