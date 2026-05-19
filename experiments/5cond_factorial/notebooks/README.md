# 5cond_factorial notebooks

| File | Purpose |
|---|---|
| `colab_5cond_run.ipynb` | Single-click Colab sweep runner for the 5-condition factorial (15 runs: 5 conditions × 3 seeds). Cells 1–7 set up the environment; Cell 8 is the (disabled by default) smoke test; Cell 9 runs Condition L; Cell 10 runs the 4 graph variants with cascade-detection; Cells 11–12 run analysis + summary. |

**To run:** open in Colab → set T4 GPU → Runtime → Run all. Notebook is idempotent: completed runs (those with `test_metrics.csv` in their target folder) are skipped on re-runs. See `../RUN_PLAN.md` for the operational protocol.

Outputs land in `runs/5cond_factorial/<cond>_seed{N}/` on the user's Google Drive (symlinked to `runs/` inside the Colab clone).
