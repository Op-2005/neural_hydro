# probes — does the trained LSTM behave as a self-stabilizing system?

Diagnostic scripts that probe a **trained model's behaviour**, as opposed to the metric comparisons
in `../analysis/`. They create no runs: each re-evaluates an existing checkpoint under a controlled
perturbation. Outputs land under `../analysis/outputs/dynamical_systems_probes/`.

## Why these exist

They came out of the April 2026 PI meeting, whose framing was that a rolled-out LSTM tends to settle
into its own self-consistent dynamics, so the interesting question is what *external* information
can knock it out of that regime.

**What they found: the model is strongly self-stabilizing** — perturbations to the hidden state are
absorbed within a step or two. That result is why the project stopped asking whether the graph could
be injected into the model's internals and started asking which external input the model can
actually use, which is the question the current study answers.

## The probes

| Script | What it perturbs |
|---|---|
| `e0_self_stabilization.py` | The hidden state (does the trajectory recover?) and the meteorological forcings (does the model track them or its own momentum?). |
| `e0_state_space_recovery.py` | Whether the state trajectory returns to its attractor, and how fast. |

The `run_e0_multiseed.sh` and `run_e0_5_multiseed.sh` wrappers sweep seeds. The output folder's
`NOTES.md` records what each probe concluded.
