# Idea 1 — Does river-network topology improve streamflow prediction?

**Master file. Current active direction. Last updated 2026-04-20 pre-professor-meeting.**

---

## One-sentence version

Using an NH-framework LSTM as the backbone, run a clean three-condition
ablation on a larger-than-23-basin CAMELS-US subnetwork to test whether
river-network topology improves streamflow prediction — and to isolate whether
any gain comes from topology-as-identity or from runtime message passing
between basins.

## The research question (plain)

*"On a connected network of basins, does adding river-network topology to an
LSTM improve streamflow prediction — and if so, does the improvement come from
knowing where each basin sits in the network, or from actually passing
information between basins at inference time?"*

## Why this question is worth asking

**Physical intuition.** Water flows downstream. A basin's discharge at time
`t` is physically a function of its own recent precipitation *and* the
discharge of its upstream neighbors hours-to-days earlier. Standard
multi-basin LSTMs (Kratzert 2019, the field's reference baseline) are
trained on every basin's time series *independently* during each forward
pass — two basins that sit on the same river get the same treatment as two
basins on opposite coasts. That seems like it ought to leave signal on the
table.

**Literature status.** Whether it actually leaves signal on the table is an
open empirical question with mixed published evidence:
- **Kratzert et al. 2019 (HESS):** Multi-basin LSTM with static attributes
  matches or beats physics-based models on CAMELS-US — establishing the
  strong baseline that any topology-aware method must beat.
- **Kirschstein & Sun 2024 (ICML):** GNNs on LamaH-CE river networks showed
  **no** benefit over a plain MLP, even with multiple adjacency definitions.
  The null result has not been fully explained.
- **Jiang et al. 2025 (ICML):** A specifically-designed directional operator
  + physics-regularized loss recovers a +31.6% gain on a river dataset,
  diagnosing Kirschstein's failure as low-pass filtering destroying
  high-frequency flow signals.

**The gap we address.** No one has run a **clean ablation at scale** that
separates the two distinct channels through which topology could help:
(1) *static identity* — "my graph position tells the LSTM what kind of basin
I am," and (2) *runtime information flow* — "my upstream neighbor's current
hidden state tells me something my own forcings do not." Our 23-basin pilot
(see `INSIGHTS.md`) suggested (1) and (2) overlap heavily, but n=23 with
heuristic edges was too small to be definitive. The scaled three-way
ablation below is designed to answer this at a statistical-power level we
can publish.

**Why it's worth a workshop paper either way.** The ablation result is
informative in all four directions:
- If C beats B by a margin → message passing is doing real work → positions
  against Kirschstein's null with a specific mechanism.
- If B ≈ C → topology is an identity signal, not a communication signal →
  consistent with Kirschstein but gives a clean explanation of *why*.
- If B ≈ A → topology as a static feature is redundant with basin encoding
  → negative result with a precise scope.
- If C > A but B > A too → both channels contribute; rare-but-publishable.

## The ablation — three conditions, same LSTM backbone

All three use the NH-framework LSTM (CudaLSTM) with basin ID encoding, trained
under identical hyperparameters on the same basin set.

| Condition | What it has | What it tests |
|---|---|---|
| **A. LSTM baseline** | Standard multi-basin LSTM + basin ID encoding (Kratzert 2019 style). No topology. | The strong published-style baseline. |
| **B. LSTM + topology-as-features** | 5 topology-derived scalars appended to the static attribute vector (see list below). **No message passing between basins.** | Does knowing your network position help, without any runtime communication? |
| **C. LSTM + topology + message passing** | DirectedGraph-LSTM: lagged upstream-parent hidden states aggregated as messages, edge features, warm-started from A. | Does runtime information flow between basins add anything beyond position-as-features? |

**What each margin tells us**

- **B − A** : the value of *topological identity* as a static signal
- **C − B** : the value of *runtime message passing* specifically
- **C − A** : total effect of topology + message passing combined

The pilot 23-basin result (C − A = +0.078 NSE median) collapsed largely into
B-like territory (graph ≈ basin encoding). Scaling and adding B as an explicit
condition resolves this ambiguity.

**Concrete Condition B feature list** (the 5 topology-derived scalars
appended to each basin's static attribute vector):

1. **Graph depth** — longest directed path length from any root to this
   basin. 0 = headwater.
2. **In-degree** — number of CAMELS basins immediately upstream.
3. **Out-degree** — number of CAMELS basins immediately downstream.
4. **Transitive upstream count** — number of CAMELS basins strictly upstream
   (all depths), divided by the network size for scale-invariance.
5. **Upstream-area ratio** — sum of upstream basins' areas divided by this
   basin's own area. Approximates "how much of my flow arrives from
   upstream." Log-transformed, z-score-normalized.

All five are computed from the same edge list used by Condition C, so A/B/C
rest on the same topology definition. They are *static* scalars (computed
once per basin) and are concatenated into the same static attribute vector
the LSTM already consumes — no architectural change required to the NH
CudaLSTM.

## Scale target

**Component 0** — 183 basins, eastern US, 6 HUC regions, max graph depth 4.
Already extracted to
`topology_analysis/phase1_network_discovery/outputs/component0_basins.txt` and
`component0_edges.csv`. Basins-per-depth: 33 / 81 / 51 / 16 / 2 — enough
statistical power at depths 1–3 (unlike the 23-basin network, which had n=2 at
depths 2 and 3).

Why not CAMELS-531: many basins in that benchmark have no graph neighbors, so
the topology variable is ill-defined for a large fraction of the data.
CAMELS-531 is a good follow-up for direct comparability to Kratzert 2019, not
the first target.

## What we are NOT claiming (bounds of contribution)

- We are not claiming state-of-the-art on CAMELS-US. EA-LSTM median ≈ 0.74 is
  the ceiling for a possible follow-up, not a requirement here.
- We are not claiming a new architectural primitive. The DirectedGraph-LSTM
  already exists in the repo; the paper's contribution is the **evaluation**.
- We are not claiming general-purpose GNN insights. The framing is hydrology-
  specific.

## What a "meaningful finding" looks like

Two publishable outcomes, both workshop-appropriate:

1. **Positive result:** C − A > 0 at 5-seed 95% CI on Component 0, AND C − B > 0
   at 5-seed 95% CI. Headline: *"topology helps streamflow prediction, and
   message passing is where the help comes from."*
2. **Negative-but-diagnostic result:** C − A > 0 but C − B ≈ 0. Headline:
   *"topology helps streamflow prediction, but only as a static identity
   signal — message passing adds nothing on top, and here is why."* Still
   publishable; positions against Kirschstein cleanly.

Null result (C − A ≈ 0) would be a negative finding consistent with Kirschstein
2024; also publishable as a scaled replication.

## Required methodology — non-negotiable for publication

1. **Multi-seed.** ≥ 5 seeds per condition, bootstrap 95% CI on per-basin ΔNSE.
2. **Matched hyperparameters across A, B, C** — no condition gets a tuned
   advantage.
3. **Matched training time** (epochs + LR schedule). Eliminates the LSTM-drift
   confound we found on 23 basins.
4. **Held-out test period** unchanged from pilot: train 1990-1999,
   validate 2000-2004, test 2005-2008.
5. **Report KGE and log-NSE in addition to NSE.** The 08165300-style
   variance-outlier-basin problem repeats at scale.

## What we need from the professor tomorrow

1. **Compute.** 3 conditions × 5 seeds × 15+ epochs on 183 basins. On CPU the
   graph runs are 8× slower than on 23 basins. Without GPU this is infeasible.
   Concrete ask: lab machine, cloud credits, or HPC access.
2. **Basin set sign-off.** Is Component 0 (183-basin eastern US) the right
   first-scale target, or does the PI have a reason to prefer CAMELS-531 or a
   different subnetwork?
3. **Methodology gaps we can't self-verify.**
   - Heuristic edges (area/elev/proximity, 150 km radius) vs. NHDPlus ground-
     truth edges — do we need NHDPlus before running, or can heuristic edges
     carry an initial submission?
   - The 23-basin pilot used Maurer forcings ending in 2008 (only 4 years of
     test). Component 0 spans regions where daymet/nldas extend to 2014 and
     give 9 years. Which forcing product?
4. **Publication framing.** Is a workshop submission (e.g. Climate Change AI,
   AI4Earth, ML4PS) the right target? Does the PI have a venue in mind?
5. **Positioning vs. prior work.** Kirschstein & Sun 2024 (null) and Jiang
   et al. 2025 (positive, Saint-Venant-regularized). Which do we cite as our
   primary point of comparison, and do we need to replicate their settings?

## Status as of meeting

- Pilot completed on 23-basin Texas network (INSIGHTS.md).
- Component 0 extracted and ready to train.
- NH training config for Component 0 baseline: `experiments/configs/lstm_component0_baseline.yaml`.
- DirectedGraph-LSTM training script parameterized for Component 0:
  `experiments/training/train_graph_component0.py`. Already supports --variant flag for
  "warm" (full), "frozen" (graph isolation), "gcn_lowpass" (a low-pass
  control). Adding a "topology_features" variant for Condition B is a small
  edit once we commit to the plan.
- Not yet launched: waiting on compute-resource decision.

## Pointers

- Alternative direction (set aside): `idea2/`
- Pilot findings: `INSIGHTS.md`
- Chronological log: `CURRENT_STATE.md`
- Experiment scripts and their status: `experiments/README.md`
- Per-run results: `runs/README.md`
