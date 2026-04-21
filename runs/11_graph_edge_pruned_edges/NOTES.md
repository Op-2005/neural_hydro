# Run 11 — Graph-LSTM with 26 pruned edges (from 34)

**Model.** DirectedGraph-LSTM, mean aggregation, edge features on. Uses a
pruned 26-edge graph — 8 edges from catastrophic-baseline-NSE parents were
removed. Warm-started from run 05.

**Script.** `experiments/training/train_graph_lstm.py` with `EDGE_FILE` pointed at
`study_network_edges_pruned.csv`.

**Result.** Median test NSE **0.504** — marginally the best pilot result
(+0.081 vs baseline).

**Why it matters.** Test of the **"bad-parent poisoning" hypothesis** — the
idea that graph gains were being dragged down by messages from basins the
baseline predicted poorly (e.g., 08165300 with baseline NSE −6.25).
**The hypothesis was WRONG.**

Evidence:
- 08195000 after pruning (edge to 08165300 removed → 0 parents): Δ NSE
  −0.082 (unchanged from unpruned run 06).
- 08190500 after pruning: Δ NSE −0.846 (slightly worse).
- Median NSE barely moved (+0.003).

The real mechanism is LSTM weight drift during joint training (per run 07),
which affects predictions for all basins regardless of their message
content. Drift is optimization-path-dependent, not graph-content-dependent.

**Where it fits.** Drives the "pruning doesn't rescue affected basins"
finding in `INSIGHTS.md`. Reinforces run 07's message: the graph's
contribution is small and mechanism is not what the intuitive story
predicted.
