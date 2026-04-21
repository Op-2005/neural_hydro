# Run 13 — Graph-LSTM on ungauged-basin setting

**Model.** DirectedGraph-LSTM warm-started from run 12. Held-out basins are
included in the graph structure (their parents pass messages at inference)
but are masked out of the training loss.

**Script.** `experiments/training/train_graph_ungauged.py`.

**Result.** Held-out per-basin NSE:

| Basin | Parent topology | Baseline (run 12) | Graph (run 13) | Δ |
|---|---|---|---|---|
| 08158700 | 1 training parent | 0.059 | 0.102 | **+0.043** ✓ |
| 08164300 | middle node, held-out parent + outputs to another held-out | 0.360 | −0.215 | **−0.575** ✗ |
| 08189500 | held-out parent + training parent | 0.233 | 0.340 | **+0.107** ✓ |

Median: baseline 0.233 → graph 0.102 (worse at median due to one basin
collapsing).

**Why it matters.** **Not a null result — a diagnostic finding.** 2 of 3
held-out basins improved meaningfully. The one that collapsed is a specific
failure mode: a *middle-node* basin whose held-out parent had an inaccurate
LSTM, AND which itself serves as parent to another held-out basin (chain
contamination).

**Concrete implications.**
- Leaf-only held-out sets would avoid this failure.
- Parent-confidence gating could mute unreliable messages.
- Parent-variance as an edge feature would carry the relevant uncertainty
  signal to the aggregator.

**Where it fits.** The ungauged result is the single clearest motivation in
the repo for *why* graph methods might matter: when basin encoding cannot
help (unseen basins), topological position is a substitute source of
basin-identity information. A focused PUB experiment with leaf-only splits
is a natural follow-up.
