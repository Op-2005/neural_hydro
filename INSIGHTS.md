# Research Insights — 23-Basin Pilot

Concise summary of what the April 2026 pilot established. For the chronological
log see `CURRENT_STATE.md`. For the current active research direction see
`idea1.md`. For the per-run details see `runs/README.md` and each run's
`NOTES.md`.

---

## Headline

On the 23-basin Texas study network, a warm-started DirectedGraph-LSTM with
edge features achieves **median NSE 0.501 (+0.078 vs strong baseline 0.423)**.
The gain reproduces across multiple aggregation variants.

## Six findings that actually matter

### 1. The +0.078 gain is mostly LSTM adaptation, not message passing

Frozen-LSTM isolation (run 07): with only the message-passing parameters
trainable, the gain collapses to **+0.013**. The other ~+0.065 is LSTM weight
drift during joint finetuning, acting as a gradient-signal regularizer rather
than a physical information carrier. (Run 07)

### 2. Graph substitutes for basin ID encoding

With a **weak** baseline (no basin encoding, run 03 = 0.407), the frozen-graph
gains **+0.086**. With a **strong** baseline (encoding on, run 05 = 0.423), the
frozen-graph gains only **+0.013**. Graph topology and one-hot basin identity
carry overlapping information — each partially substitutes for the other.

### 3. Graph helps basins the baseline already predicts well

Per-basin correlation between ΔNSE (graph minus baseline) and own baseline
NSE: **r = +0.82**. Adding parents' NSE lifts R² from 0.665 to 0.775. The
graph amplifies good predictions rather than rescuing struggling ones.

### 4. Aggregation-family variants all converge

Mean (0.501), softmax attention (0.495), sigmoid gate (0.496), Jiang direction
term (0.492). Error correlations between variants: 0.994–0.999. Ensembling
gains only +0.0005 NSE. The space of aggregation mechanisms is saturated on
this dataset. (Runs 06, 08, 09, 10.)

### 5. The "bad-parent poisoning" hypothesis was wrong

Pruning the 8 bad-parent edges (run 11) did **not** rescue the downstream
basins the theory predicted. The real mechanism is optimization-path-dependent
LSTM drift, not graph-content-dependent poisoning.

### 6. Ungauged setting reveals a specific failure mode

Run 13 (3 held-out basins): 2 of 3 improve with the graph (+0.043 on a leaf,
+0.107 on a dual-parent node), 1 collapses (−0.575 on a middle-node with a
held-out parent that also serves as parent to another held-out basin). Chain
contamination is a concrete, diagnosable failure — not a general null.

---

## Honest limits of the pilot

- **Single seed.** Every number is from seed=42. Multi-seed bootstrap CIs are
  required before publication.
- **Heuristic edges.** 34 edges inferred from area/elev/proximity (150 km,
  area ratio ≥ 1.5, elevation-decreasing). Some are physically implausible
  (e.g., 08103900 → 08171300 at 104 km with a 3.7 m drop). NHDPlus ground-
  truth edges needed.
- **Low statistical power at depth.** n = 2 at depths 2 and 3. Any depth-
  stratified claim is noise-dominated on this network. Scaling to Component 0
  (183 basins, 51 at depth 2) addresses this.
- **Outlier sensitivity.** Basin 08165300 (baseline NSE −6.25) distorts
  aggregate statistics. KGE or log-NSE would be more robust.
- **Maurer forcings end 2008** → test window is only 4 years.

## Where this leaves us

The pilot is substantive enough to motivate a scaled experiment. It is not
large enough to carry a paper. The findings above — especially #1, #3, and
#6 — remain load-bearing contributions in whichever direction (`idea1.md` or
`idea2/`) we take forward.
