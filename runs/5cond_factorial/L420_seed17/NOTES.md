# L420_seed17 — matched-budget cudalstm control

**Model.** NeuralHydrology `cudalstm` (hidden_size=64, dropout=0.4, initial_forget_bias=3, basin one-hot encoding ON, Maurer forcings).
**Script.** `experiments/training/train_matched_budget_lstm.py --seed 17 --max-steps 420 --device cpu`.
**Result.** Median NSE 0.501, mean NSE 0.443, median KGE 0.554 (n=183 basins, test 2005–2008).
**Why it matters.** Seed 17 of the 3-seed Step 1 sweep. See seed-11 NOTES for the cross-seed interpretation.
**Associated outputs.** Pre-registration + interpretation: `experiments/5cond_factorial/preregistration_step1.md`.
