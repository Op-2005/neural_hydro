# Run 15 — Graph-LSTM with topology features (Condition B), seed 42

**Status.** Single-seed first scaled run. **First scaled-graph result.**

**Model.** DirectedGraph-LSTM with `edges=[]` (no message passing) but
with **5 topology-derived static features appended** to the static
attribute vector: graph depth, in-degree, out-degree, transitive upstream
count, log upstream-area ratio. Per the locked Condition B specification
in `idea1.md`. 30 epochs, seed 42, **no warm-start** (matched-protocol
across A/B/C).

**Script.** `experiments/training/train_graph_component0.py --variant
topology_features --seed 42 --no-warm-start --epochs 30 --baseline-run
runs/A_baseline_seed42_*` (run from Cell 10 of
`notebooks/colab_publication_run.ipynb` on Colab Pro L4).

**Result.** Median test NSE **0.591** (mean 0.575). NSE range
[-0.320, 0.783].

**Per-basin contrast vs Condition A (run 14):**
- Median (B − A) per-basin ΔNSE: **−0.050**
- Mean: −0.011 (smaller in magnitude because A has more high-NSE basins)
- 51 of 183 basins where B > A; 92 of 183 where B is worse by ≥ 0.05
- Std of per-basin Δ: 0.487 — high variance, not a uniform shift

**Why it matters.** **Topology-as-static-features does not help Component-0
streamflow prediction at this seed.** It actively hurts at the median.
Two readings, both honest:
1. The 5 topology scalars contain redundant information already encoded
   by basin ID encoding + static attributes. Adding them moves the
   optimization off the baseline's good attractor.
2. The B model trains from scratch (no warm-start), so the weight
   initialization is independent of A. A − B at the median is partly
   structural (the augmented input vector) and partly seed-induced
   training-trajectory difference.

The +0.078 NSE pilot result on 23 basins (run 06) does not transfer to
B-style augmentation at Component-0 scale. The narrative shifts: at the
larger network, topology-as-position is not the kind of information the
LSTM is missing.

**This makes Condition C the key remaining experiment.** If C also fails
to beat A, the dynamical-systems framing's "external forcing helps" claim
is wounded at scale — at least for this network and seed.

**Associated outputs.**
- Per-basin NSE: `test_metrics.csv` (183 rows)
- Run config: `run_config.json` (variant, hyperparameters, final NSE)
- Training-loss + NSE history embedded in `run_config.json`

**Single-seed caveat.** All conclusions above are based on seed=42 only.
The multi-seed E0.5 finding (cross-seed val NSE spread of 0.111 NSE on
23 basins) suggests B − A might be substantially different at other seeds.
Multi-seed verification (the `'full'` MODE of the notebook) is the next
step before any framing-level claim is made.
