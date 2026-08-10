# notebooks/ — turnkey Colab notebooks (Run-all, persist to Drive)

Each notebook clones the repo from GitHub `main`, pins numpy<2 / pandas 2.1.4, symlinks CAMELS +
`runs/` to Drive, trains on a T4, and self-verifies against a pre-registration. Open via
`colab.research.google.com/github/<owner>/<repo>/blob/main/experiments/topology_ablation/notebooks/<file>`,
set Runtime → T4 GPU → Run all. Idempotent (skips completed seeds).

| Notebook | Trains / does | Pre-reg |
|---|---|---|
| `colab_topology_2x2.ipynb` | static-topology 2×2 (seed 11) | `preregistration_upstream_signal.md` |
| `colab_2x2_multiseed.ipynb` | static-topology 2×2 (seeds 13/17) | same |
| `colab_realizability.ipynb` | realizable (predicted-upstream) condition | `preregistration_realizability.md` |
| `colab_multiseed.ipynb` | headline conditions × seeds 13/17 | `preregistration_multiseed.md` |
| `colab_multiseed_mechanism.ipynb` | forward / reversed / random mechanism × 3 seeds | `preregistration_multiseed_mechanism.md` |
| `colab_directionality_controls.ipynb` | reversed / random directionality controls | `preregistration_directionality_controls.md` |
| `colab_oracle_completion_and_k2.ipynb` | oracle completion + k=2 pruned graph | `preregistration_baseline_completion_and_k2.md` |
| `colab_oracle_seed11_restore.ipynb` | re-train oracle seed 11 to restore lost `results.p` | — |
| `colab_distance_control.ipynb` | **distance-preserving control** (proximity vs topology) | `preregistration_distance_control.md` |

The verdict cells use a self-contained normal-approximation Wilcoxon (Colab's scipy is incompatible
with the pinned numpy<2), so they run without scipy.
