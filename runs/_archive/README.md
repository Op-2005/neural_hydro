# Archived Runs

Runs preserved here for historical / methodological completeness but **whose
numbers should not be cited as primary evidence** in the project narrative.
The reason for archival differs per run — see each run's `NOTES.md`.

The archive policy is `/organize` pattern #8: *never delete; relocate or
rewrite.* These runs document specific experimental choices that informed
later decisions, even though the runs themselves were superseded.

## Inventory

| Folder | Why archived | Where its evidence ended up |
|---|---|---|
| `graph_lstm_warm_noedge_WEAK_1904_152734/` | Warm-started from the **weak** baseline (run 03, no basin encoding) instead of the strong baseline (run 05, with encoding). Pre-dated the `find_strong_baseline()` glob fix. Final median NSE 0.479. | Findings #2 / #3 in `INSIGHTS.md` — "graph substitutes for basin ID encoding": this was part of the evidence that without encoding, graph adds more (+0.072 vs +0.013). |
| `graph_lstm_edgefeat_warm_frozen_WEAK_1904_160530/` | Same root cause — warm-started from weak baseline. Edge features on, LSTM+head frozen. Final median NSE 0.493. | Same as above: the WEAK-baseline + frozen + edges contrast (+0.086 vs the +0.013 from STRONG-baseline + frozen + edges) is the cleanest evidence for Finding #2. |

## Why these are not the main runs

Both runs train against the **weak** baseline (`runs/03_lstm_23basin_baseline`,
no basin ID encoding) rather than the strong baseline (`runs/05_lstm_23basin_strong_baseline`,
with encoding). The strong baseline is the reference point for every
graph-variant comparison cited in `../../INSIGHTS.md` and `../README.md`,
because it matches Kratzert 2019's published setup. Running graph variants
against the weak baseline is a *separate* contrast — informative for the
"graph vs basin ID encoding substitution" question, but not the primary
+0.078 NSE headline.

After the `find_strong_baseline()` glob bug was fixed
(see `experiments/training/train_graph_lstm.py:75-91`), runs 06 onward
correctly resolve the strong baseline. These two archived runs were the
last to use the buggy resolution.

## Do not delete

Even when the project moves to scaled experiments, these runs stay.
They document the pre-fix experimental setup and remain the cleanest
WEAK-baseline contrasts on the 23-basin pilot network.
