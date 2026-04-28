# Probes

Diagnostic scripts that probe **trained-model behavior** as a dynamical
system, distinct from the analysis scripts in `../analysis/` (which
compare aggregated metrics across runs). Probes do not produce new runs;
they re-evaluate existing checkpoints under controlled perturbations to
test specific claims about how the model behaves at inference.

| File | What it tests | Pre-registered in | Outputs |
|---|---|---|---|
| `e0_self_stabilization.py` | Whether the trained baseline LSTM exhibits self-stabilizing dynamics — the load-bearing claim of the dynamical-systems framing in `../../idea1.md`. Two probes: hidden-state perturbation recovery (Probe A) and forcing replacement (Probe B). | `idea1.md` §E0 | `../analysis_outputs/e0/` (decision_record.json, CSVs, PNGs) — see `../analysis_outputs/e0/NOTES.md` |

## How probes differ from analysis scripts

| | `analysis/` | `probes/` |
|---|---|---|
| Reads | `runs/<n>/test_metrics.csv` | `runs/<n>/model_*.pt` (weights) |
| Operates on | Aggregated per-basin metrics | Live model rollouts at inference |
| Tests | "Did model A beat model B?" | "How does model A *behave*?" |
| Result form | Comparison tables, correlation matrices, hydrographs | Decision records, behavior curves |

If you are adding a new diagnostic experiment that loads a checkpoint and
runs the model in a custom way (perturbation injection, ablation in the
forward pass, intervention on hidden states), it goes here. If you are
adding a new way to *summarize* metrics across existing runs, it goes in
`../analysis/`.

## Running probes

From the repo root:

```bash
# E0 — self-stabilization on the strong baseline (run-05)
/Applications/anaconda3/envs/nh/bin/python experiments/probes/e0_self_stabilization.py
```

The script's perturbation σ is set at the top of the file
(`PERTURB_SIGMA_FRAC`). The pre-registered canonical run is σ=0.5; a
sensitivity check at σ=2.0 has also been run and saved with `_sigma_2_0`
suffixes in `../analysis_outputs/e0/`.
