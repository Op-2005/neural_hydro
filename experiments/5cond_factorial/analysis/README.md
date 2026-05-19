# 5cond_factorial — Analysis Documents

All written analysis of the 5-condition factorial run (May 2026, Component 0, 183 basins). These files describe results, design critique, and the forward plan. Read in this order if you're new:

| File | Read for | Length |
|---|---|---|
| `meeting_brief.md` | Plain-English summary + questions for the professor. Start here. | ~10 min |
| `5cond_run_analysis.md` | The numbers and what they mean. Statistical tables, contrasts, stratifications. | ~5 min |
| `architecture_analysis.md` | Deep code-level critique of every design choice (DirectedGraphLSTM, topology features, message passing, training pipeline). Prioritized improvement tiers. | ~20 min |
| `testing_framework_proposal.md` | 6-step pre-registered experimental ladder to make follow-up rigorous. | ~10 min |

## Related folders

- `../multi_condition_ablation/` — the 15 actual run output folders (model checkpoints + per-basin metrics + predictions).
- `../../analysis_outputs/5cond_component0/` — auto-generated artifacts (per-basin CSVs, figures, RESULTS.md) from `experiments/analysis/compare_5conditions.py`.
- `../preregistration_step1.md`, `../preregistration_step2.md` — formal pre-registrations for the follow-up experiments proposed in `testing_framework_proposal.md`.

The four files in this folder are stable — they describe what we did and learned. They do not get rewritten as new experiments land; instead, new findings produce new pre-registrations + new addenda.
