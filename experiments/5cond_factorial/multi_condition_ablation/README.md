# multi_condition_ablation — 5cond factorial run outputs (Colab sweep, 2026-05-12)

Production output folder for the 5-condition × 3-seed factorial. **15 run folders** (5 conditions × 3 seeds):

| Condition folders | What it is |
|---|---|
| `L_seed{11,13,17}/` | NH cudalstm baseline. Full NH layout: `config.yml`, `model_epoch{001..030}.pt`, `test/model_epoch030/{test_metrics.csv, test_results.p}`, `train_data/`. |
| `G_seed{11,13,17}/` | DirectedGraphLSTM, empty edges, no topology features. Graph trainer layout: `model_epoch{001..030}.pt`, `test_metrics.csv`, `test_predictions.csv`, `run_config.json`. |
| `G_T_seed{11,13,17}/` | + 5 topology static features. Same layout as G_seed. |
| `G_M_seed{11,13,17}/` | + edge messages (full graph). Same layout as G_seed. |
| `G_T_M_seed{11,13,17}/` | Both topology features and edge messages. Same layout as G_seed. |

**Cross-reference:** the same runs are symlinked into `../../runs/5cond_factorial/` so that `experiments/analysis/compare_5conditions.py` can discover them via its expected `RUN_ROOT`. Either path resolves to the same files.

**Headline results:** see `../../5cond_run_analysis.md` (results digest) and `../../experiments/analysis_outputs/5cond_component0/RESULTS.md` (auto-generated, full statistical tables + figures).

**Reproduction:** Colab notebook `../notebooks/colab_5cond_run.ipynb` on T4 GPU. Each graph variant takes ~30–90 min per seed; L runs ~3 min per seed.
