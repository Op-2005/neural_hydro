# L420_seed11 — matched-budget cudalstm control

**Model.** NeuralHydrology `cudalstm` (hidden_size=64, dropout=0.4, initial_forget_bias=3, basin one-hot encoding ON, Maurer forcings).
**Script.** `experiments/training/train_matched_budget_lstm.py --seed 11 --max-steps 420 --device cpu`.
**Result.** Median NSE 0.521, mean NSE 0.450, median KGE 0.577 (n=183 basins, test 2005–2008).
**Why it matters.** Step 1 of the testing-framework ladder: tests whether the L − G gap of +0.050 NSE in the 5cond factorial is a step-count effect. **Result was third-category** — L_420 ≪ G (paired Δ = −0.100). The "matched gradient steps" framing was biased toward whichever trainer has the larger per-step batch.
**Associated outputs.** Pre-registration + interpretation: `experiments/5cond_factorial/preregistration_step1.md`. Paired contrast computed against `runs/5cond_factorial/G_seed*/`.
