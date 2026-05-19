# Architecture Analysis — Why Our Advanced Features Underperform

**Date:** 2026-05-12
**Scope:** Deep technical critique of (a) the DirectedGraphLSTM model, (b) the 5 topology static features, (c) the message-passing mechanism, (d) the training pipeline that produces all 4 graph variants. Companion to `5cond_run_analysis.md` (results) and `testing_framework_proposal.md` (fix plan).

**Central question:** the paper claim is "our added features outperform a standard LSTM on discharge prediction." Currently we have G+T+M < G+M ≈ G+T < G < L. We are losing on *both* axes: against the architecture-matched control (G) AND against the cudalstm reference baseline (L). This document audits *each component* of the design and identifies the load-bearing defects.

**Stance.** I will be uncharitable. Where the design is questionable, I'll say so. Where it's reasonable, I'll say that too. This is not a "polish the existing system" pass — it's a "would I do this from scratch?" pass.

---

## Part 0 — The L−G gap: confirmed training-budget confound

Before diagnosing the *graph* design, we must accept that the L−G gap (~0.050 NSE, 5× larger than any other contrast in the run) is dominantly a training-budget confound, not an architecture confound. Mechanism:

- NH `cudalstm` trainer samples random `(basin, window)` pairs; batch=256 produces ~2,610 gradient steps per epoch (183 × 3,652 / 256).
- Our DirectedGraphLSTM trainer samples whole-windows-of-all-183-basins (required for inter-basin message passing); batch=256 produces ~14 gradient steps per epoch (3,652 / 256).
- 186× ratio. 30 epochs of graph training = 420 gradient steps vs ~78,000 for cudalstm.
- Loss is still falling at epoch 30 in all 4 graph variants (~3% decline in the trailing 4 epochs); convergence is not reached.

**Implication for this document.** All within-graph-trainer contrasts (G+T − G, G+M − G, G+T+M − G, the interaction term) sit on an undertrained baseline. They are *internally* valid (same training budget across all four) but may be *quantitatively* attenuated. The architectural critiques below stand independently — they identify defects that limit upside even with sufficient training — but the relative magnitudes might shift after a budget-matched re-run.

---

## Part 1 — DirectedGraphLSTM core design

Source: `experiments/training/train_graph_lstm.py` lines 98–294.

### 1.1 What it does

For each timestep t and each basin v:

```
h_v^t, c_v^t = LSTMCell(x_v^t, h_v^{t-1}, c_v^{t-1})     # per-basin LSTM update

For each upstream parent u of v (using h_u^{t-1}, the lag):
    m_uv = W_msg_edge([h_u^{t-1}, e_uv])                  # message
    
agg_v = aggregate({m_uv : u ∈ parents(v)})                # mean, attention, or gate
h_v^t = h_v^t + tanh(W_out · agg_v)                       # residual update, W_out zero-init
```

Output: `y_hat = head(dropout(h^seq_len))` — predict last day of each window.

### 1.2 Defects in the core design

**Defect 1.2.1 — Forward pass is a Python loop over seq_length=30.**
- Each timestep does an LSTMCell call + (if edges) the message-passing block, all in a Python `for t in range(seq_len):` loop.
- This is intrinsically slow vs `nn.LSTM` which fuses the per-step computation in cuDNN.
- *Why it matters now:* combined with the per-window batching this is what creates the gradient-step bottleneck. Each forward pass is expensive enough that batching 256 windows is the practical ceiling.
- *Why it might matter regardless:* `torch.compile` is supposed to fuse the loop, but our smoke-test results show no obvious speedup. The compile mode `reduce-overhead` is the wrong mode for a graph-break-heavy forward; should be `default` or `max-autotune`.

**Defect 1.2.2 — Loss only on the LAST day of each window (`predict_last_n: 1`).**
- 29 of 30 timesteps contribute zero gradient signal (they're warmup).
- This is identical to cudalstm config, so it's not a relative defect — but it's worth noting that we're paying the full forward-pass cost for one day of supervision.
- *Better practice:* `predict_last_n: 5` or larger gives 5× more supervision per window with negligible compute cost increase.

**Defect 1.2.3 — Message passing uses h from t−1, which is zeros at t=0.**
- First timestep: `h_u = h[parent_idx]` retrieves all zeros.
- So the first message-passing event is computing W_msg_edge([0, edge_features]) — pure edge-feature signal, nothing from the parent.
- For depth-1 basins, the upstream signal effectively starts at t=1. For depth-2 basins, useful upstream signal might require t≥3 (one hop per timestep).
- *Why it matters:* with seq_length=30 and Component-0 max depth = 4, this is fine — there's plenty of time. But it's an artifact and would matter for shorter sequences or deeper networks.

**Defect 1.2.4 — Residual structure has zero-init W_out, so the graph contribution starts at exactly zero.**
- Looks like a safe warm-start. *In practice* it means the gradient signal to W_out only exists if h_parent (post-LSTM) carries useful info. Early in training, h_parent is approximately random — gradient is noise.
- Net result: W_out tends to stay small for many epochs while the LSTM learns to "ignore" the graph contribution. The graph path is essentially a "vestigial appendage" that the model has to actively *learn to use*.
- *Better practice:* nonzero scaled init (e.g., 1/sqrt(hidden)/10 — small but not zero) gives the optimizer a chance to find a useful direction immediately.

**Defect 1.2.5 — Single linear `W_msg_edge` for the message function.**
- Linear in → linear out, no nonlinearity inside the message function itself.
- The pipeline is: linear → aggregation → linear → tanh. Just one nonlinearity per message-passing round, at the very end.
- *Better practice:* 2-layer MLP with ReLU/GELU inside the message function. Common in GraphSAGE, GIN, and most modern GNNs.

**Defect 1.2.6 — Single hop per timestep is a *hard* architectural limit.**
- The model has one message-passing layer per timestep. So a depth-3 basin receives signal from a depth-0 root only after 3 timesteps of lag.
- This is physically motivated (water takes time to flow downstream) but the *temporal* lag and the *graph-depth* lag are NOT the same thing. River velocities vary 10×; some flows reach the outlet in hours, others in weeks.
- *Better practice (and the literature consensus):* K-hop message passing per timestep, where K is a hyperparameter. Then *both* timestep-lag (from the LSTM) AND graph-depth-lag (from the K hops) carry information.

**Defect 1.2.7 — There is no skip connection between timesteps for the graph component.**
- The graph residual `h_new = h_new + tanh(W_out(m))` is *applied to* the LSTM hidden state, mixing them.
- This means in the next timestep, `h` is "the LSTM update + the graph correction" — entangled.
- *Better practice:* keep the LSTM trajectory clean by storing the graph correction separately and projecting only into the output prediction, OR run two separate hidden states (LSTM-h and graph-h) with a gating mechanism. Either preserves the inductive bias more cleanly.

### 1.3 What the core design got right

- **Physical lag is correct.** Using h^{t-1} for messages models the 1-day lag of upstream→downstream flow at typical CAMELS basin scales.
- **Mean aggregation as the default** (not sum) is the right call — sum-aggregation would blow up for high-in-degree basins.
- **Zero-init W_out** is a reasonable safety mechanism, even though it has the cold-start problem above.
- **Forget bias initialized to 3** matches the cudalstm baseline (this was a real audit finding — the architectures are matched on this front).

---

## Part 2 — The 5 topology features

Source: `experiments/training/train_graph_component0.py` lines 98–146, function `compute_topology_features`.

### 2.1 What they are

For each basin, 5 scalars are computed once at startup and appended to the static input vector:

| Idx | Feature | Computation | Range typical |
|---|---|---|---|
| 0 | graph depth | max over roots of `nx.shortest_path_length(G, root, b)` | 0–4 |
| 1 | in-degree | `G.in_degree(b)` | 0–15 (median 3, mean 4) |
| 2 | out-degree | `G.out_degree(b)` | 0–~3 |
| 3 | upstream count / n | `len(nx.ancestors(G, b)) / n_basins` | 0–~0.5 |
| 4 | log((upstream_area + own_area) / own_area) | log of relative-network-size scalar | varies |

Then z-normalized column-wise. Each basin gets a 5-vector appended to its static input.

### 2.2 Core defect: redundancy with basin one-hot encoding

NH `cudalstm` uses `use_basin_id_encoding: True`, which produces a **671-dimensional one-hot per basin** (671 = the full CAMELS basin list). Our graph trainer inherits this via the NH dataset pipeline (`x_one_hot` column in `sample["x_one_hot"]`).

**The one-hot encoding already perfectly identifies each basin.** The LSTM can learn an arbitrary per-basin response curve directly from the 671-dim one-hot. The 5 topology features are 5/681 ≈ **0.7% of the static input.**

Result: the topology features are *redundant signal in low-dimensional drag-along channels*. Even if the model wanted to use them, the basin one-hot already captures any per-basin information, and the topology features are noise from the LSTM's perspective.

**This is the structural reason G+T − G ≈ 0.** It's not that the features are bad — it's that the design choice to combine them with a 671-dim one-hot encoding makes them informationally redundant.

### 2.3 Other defects

**Defect 2.3.1 — Depth is computed wrong (minor bug).**
- Docstring says "longest path from any root." Code uses `nx.shortest_path_length`. For tree-shaped sub-networks these coincide; for DAGs (which Component 0 partly is — 116 of 183 basins have ≥ 2 parents) they don't.
- Effect: depth values may be slightly off for basins with multiple parents. Not catastrophic, but means the feature definition doesn't match its name.

**Defect 2.3.2 — Three of five features are low-cardinality integer-valued.**
- Depth ∈ {0,1,2,3,4} (5 levels), in-degree ∈ {0,1,…,15} (mostly 0–6), out-degree ∈ {0,1,2,3} (mostly 0–2).
- Z-normalizing them maps to a quasi-discrete continuous variable. This is suboptimal for LSTM input: the model has to learn that "depth = 0" should produce a fundamentally different behavior from "depth = 1", but it's just seeing a small numeric difference.
- *Better practice:* embed these as small categorical variables. `nn.Embedding(num_classes, d_embed)` for depth, in-degree, out-degree, each with d_embed=4 or 8. The LSTM then sees a 12–24-dim semantically-meaningful feature, not a 3-dim numeric blob.

**Defect 2.3.3 — Network-relative features depend on Component 0 specifically.**
- "upstream count / n_basins" uses n=183 because Component 0 has 183 basins. The same basin in a different network (or just a different cut of the network) would get different values.
- This makes the model overfit to Component 0's topology and won't transfer to other networks.
- *Better practice:* either normalize differently (e.g., by HUC-2 region size) or design features that are absolute properties of the basin (drainage area, mean elevation gradient, etc.).

**Defect 2.3.4 — `log((up_area + own_area) / own_area + 1e-6)`** has a numerical-eyesore in the formula.
- This is `log(1 + up_area / own_area + 1e-6)`. The `+1e-6` is meaningless because the argument is already ≥ 1.
- More importantly: the feature is monotone in `up_area / own_area`, which is just the *relative-upstream* feature with a log transform. It's well-defined but doesn't add much beyond "depth" and "upstream count" — three of the five features encode roughly the same thing (network position).

**Defect 2.3.5 — Features are static (constant in time) but the underlying graph effects are dynamic.**
- A "depth-3 basin in a 100-basin network" has different behavior in *wet vs dry season*, in *snowmelt-dominated vs rainfall-dominated regimes*, etc.
- A static feature can express the "label" but not its time-varying interpretation. The LSTM has to do all the lifting.
- *Better practice:* dynamic graph features (e.g., per-timestep aggregated upstream rainfall) would carry more signal — but this overlaps with what message passing is supposed to do.

### 2.4 What the topology features got right

- Z-normalization is correct (zero mean, unit std per column) and was applied consistently.
- The 5 features cover position (depth, network share), local structure (in/out-degree), and scale (relative upstream area) — a reasonable basis.
- Computing them once at startup rather than per-batch is the right efficiency call.

### 2.5 Diagnosis on why G+T = G

**With basin one-hot encoding ON: topology features are redundant ⇒ G+T − G ≈ 0 (confirmed).**

**Predicted behavior with basin one-hot encoding OFF (an experiment we haven't run): topology features should become non-redundant and provide a *real* lift.** This is testable cheaply (see `testing_framework_proposal.md` Step 2).

---

## Part 3 — Message passing mechanism

Source: `train_graph_lstm.py` lines 207–294, `load_graph_with_features` lines 300–329.

### 3.1 What it does (the "warm" / G+M / G+T+M variants)

1. **Edge features (z-normalized):** `[log(distance_km + 1), log(area_ratio), elev_drop]`
2. **Message function:** `msg = W_msg_edge(concat([h_parent_at_t-1, edge_features]))` — single linear layer producing a 64-dim message per edge.
3. **Aggregation:** mean over parents (default) — `agg = sum(msg over parents) / parent_count`.
4. **Residual:** `h_new += tanh(W_out · agg)` — W_out is `nn.Linear(64, 64, bias=False)`, zero-initialized.

### 3.2 Defects

**Defect 3.2.1 — Mean aggregation gives equal weight to a 1 km² parent and a 100 km² parent.**

The Component 0 graph has area-ratio (child/parent) min/median/max of **1.5 / 3.6 / 275**. A child basin with two parents — one contributing 10% of its drainage area, one contributing 60% — will have their messages averaged with equal weight. That is *physically wrong*.

A child's hydrological response is dominated by upstream basins in proportion to their area contribution. Mean aggregation throws this away.

*Better practice (any of these would help):*
- **Sum, not mean** — but sum needs degree normalization. Standard GCN uses `D^{-1/2} A D^{-1/2}` which IS approximately what we want here.
- **Area-weighted aggregation:** `m_v = sum_u (area_u / total_upstream_area_v) · msg_u`. Physically grounded.
- **Attention-based weighting** (the `attn` variant exists in code but wasn't tested in the factorial). Should be the *primary* approach; mean is too naive.

**Defect 3.2.2 — Edge feature is just a 3-vector tacked on; no normalization with respect to total network.**

The edge feature is `[log(dist), log(area_ratio), elev_drop]` z-normalized over all edges in the network. So a "typical" edge has 0-mean features. But the *physically interesting* edges (very long distance, very large area ratio, very high elev drop) sit out in the tails. Z-normalization compresses the tails.

Also, `area_ratio` is **child_area / parent_area** (always > 1 in the data, since downstream basins are larger than their tributaries). This is the *wrong direction* for physical intuition — we typically care about `parent_area / total_upstream_area_of_child`, which is the parent's *fractional contribution*. The current feature can't express that without seeing other parents.

*Better practice:* include `parent_area / sum_over_parents(parent_area)` as an edge feature — the parent's contribution fraction.

**Defect 3.2.3 — Single linear message function.**

`W_msg_edge: Linear(67 → 64)`. One matrix. No nonlinearity in the message function itself. The only nonlinearity in the message pathway is `tanh(W_out(m))`, which is *after* aggregation.

This means the model cannot express things like "if h_parent says 'high flow' AND edge feature says 'large area ratio' then send a strong message" — that's a logical-AND which needs nonlinearity *inside* the message function. Currently the only thing W_msg_edge can do is linearly combine h_parent dimensions with edge features.

*Better practice:* 2-layer MLP (`Linear(67, 64) → ReLU → Linear(64, 64)`). Doubles parameter count for that layer but is the GNN-literature default.

**Defect 3.2.4 — Information bottleneck through `tanh(W_out(m))`.**

After aggregation, the message is `tanh(W_out(m))`. The `tanh` saturates at ±1. Any signal that pushes W_out · m past ±2 gets squashed flat — gradient nearly zero.

Once W_out grows (and it has to, since it starts at zero), the saturation kicks in and limits the magnitude of the graph contribution. This is a self-limiting design.

*Better practice:* drop the `tanh`, replace with a learnable scale factor (a single scalar parameter that gets initialized to 0 and grows). Or use a ResNet-style scaled residual where the scale is a sigmoid of a learned parameter.

**Defect 3.2.5 — h_parent as the message is overkill.**

The message carries the *entire* 64-dim hidden state of the parent. h_parent encodes "everything the parent's LSTM knows" — local rainfall history, basin-specific responses, season — most of which is not useful to the child (the child has its own copy of regional rainfall, its own basin one-hot, its own season).

The child *actually needs* one of two things:
- The parent's predicted discharge (a single scalar)
- An "abstract upstream signal" — a few dimensions encoding "how much water is flowing past the parent right now"

Sending all 64 dimensions makes the model do a lot of extra work to extract those few useful dimensions. It also makes the message function harder to learn from limited data.

*Better practice:* either (a) explicitly route `discharge_predicted_parent` as the message, or (b) bottleneck the message through a 1-D-to-K-D learnable projection — explicitly making the message small.

**Defect 3.2.6 — There is no edge-level dropout.**

Edges are deterministic. The model can overfit to specific parent-child relationships. With 624 edges and limited training data, this is plausible.

*Better practice:* DropEdge (randomly drop a fraction of edges per training batch) — standard GNN regularization that improves generalization.

### 3.3 Available variants we didn't test

The code has `use_attention` and `use_sigmoid_gate` modes. The 5-condition factorial chose `warm` (mean aggregation + edge features) for G+M and G+T+M. The attention / sigmoid variants might give a real lift over mean aggregation but were left out of scope for this sweep.

The original 23-basin pilot showed `attention` and `sigmoid_gate` produced essentially identical results (correlation 0.994–0.999 with each other and with mean aggregation) — suggesting the *aggregation choice* is not the bottleneck; the *message function* is. This matches Defect 3.2.3 (single linear message function — there's not enough expressivity in the messages for the aggregation choice to matter).

### 3.4 What the message passing got right

- Edge directions are correctly upstream → downstream (parent → child).
- Edge features are reasonable physical quantities (distance, area ratio, elevation drop).
- Z-normalization of edge features is correct.
- The lag of t-1 is hydrologically motivated.

---

## Part 4 — Training pipeline confounds (recap + new)

Mostly covered in Part 0. Adding pieces that came up during the audit:

**Defect 4.1 — Best-checkpoint selection on the test set.**

In `train_graph_lstm.py` lines 635–643 (the older, standalone trainer), the best checkpoint is selected by evaluating on test data and saving the best. This is **test-set leakage** — the test NSE used for reporting is from a checkpoint selected by looking at the test set.

The production trainer (`train_graph_component0.py`) does **not** do this: it uses the final epoch's checkpoint. So our 5-condition factorial results are unaffected. But the code in `train_graph_lstm.py` is buggy and should be fixed before any future work.

**Defect 4.2 — No proper validation set.**

The NH cudalstm runs use `validate_every: 5` with a 10-basin validation sample — but the graph trainer does no validation at all. Final epoch is whatever it is.

If the loss is decreasing slowly (as it is — Part 0 / §2), it's possible the model is *still* learning useful things at epoch 30 *or* that it's started overfitting in some subtle way. Without a held-out validation set we can't tell which.

*Better practice:* hold out a basin-stratified validation set (e.g., 10% of basins entirely) and select epoch based on validation NSE. This also gives us a "best-epoch" signal that doesn't peek at test.

**Defect 4.3 — Random seed only affects model init, not data shuffling.**

The graph trainer sets `torch.manual_seed(args.seed)` and `np.random.seed(args.seed)`. But the data is loaded once (deterministic) and shuffled with `np.random.shuffle(indices)` — which uses the seeded RNG. So shuffling is reproducible per seed, but model-init and data-order are coupled to the same seed. Cross-seed variance therefore conflates two sources (init and order).

This is a minor issue but worth noting: the 0.017 cross-seed std might be a slight underestimate if init-only seed variation is smaller than init+order variation.

**Defect 4.4 — Compile mode is wrong for our forward pass.**

`torch.compile(model, mode='reduce-overhead', fullgraph=False)`. `reduce-overhead` is best for *graph-break-free* models; our forward has Python control flow (the `for t in range(seq_len):` loop with conditional message passing) which forces graph breaks. The right mode here is `default` or `max-autotune`, and `fullgraph=False` should stay false so the compile is allowed to give up on the loop.

Result in practice: compile probably gave us ~0 speedup. Worth verifying on Colab whether epochs are actually faster with `--use-compile`.

---

## Part 5 — Putting it together: why G+T+M < G

Recall the run results: G+T+M (full features) NSE = 0.586. G (no features) NSE = 0.609. The "full" model is *worse* than the no-features baseline by 0.011 NSE.

Possible explanations and what the audit tells us:

1. ❌ **"Topology features are hurting"** — no. G+T − G ≈ 0. Features are inert, not harmful.
2. ✅ **"Message passing is slightly hurting under-train"** — likely. G+M − G = −0.006. With the message-passing block in place, the model has *more parameters to train but the same training budget*, so the marginal effective training-per-parameter is worse. Plus W_out starting at zero means the path is hard to learn early.
3. ✅ **"The combination is sub-additive (correlated defects, not complementary)"** — the −0.009 interaction term + the −0.011 G+T+M − G says the same thing two ways: adding topology *and* messages doesn't help because they're trying to do the same job (encode "where am I in the graph"). Stacked, they steal each other's gradient signal.
4. ❌ **"Architecture is wrong for hydrology"** — no evidence for this. The basin-uniformity of the L − G gap argues against any specific architectural mismatch. The model is *plausibly* the right shape; it just needs more training and better tuning.

The fundamental story: **we built a model where the graph features and message passing both have to learn to override an already-strong basin one-hot encoding, with only ~420 gradient updates total, with zero-init residual paths, mean aggregation, and a single linear message function.** Of course it doesn't add value. None of those design choices were *wrong* individually, but together they make the graph signal pay a steep learning cost for no comparative advantage.

---

## Part 6 — Concrete improvement directions (prioritized)

These are paper-survival items. Listed in rough order of expected impact × cost.

### Tier 1 — Probably-must-fix (cheap, high upside)

| # | Change | Expected effect | Cost |
|---|---|---|---|
| **1.1** | **Increase gradient-step budget.** Drop batch_size to 32 (giving 114 steps/epoch instead of 14), or extend to 200 epochs at batch=256. Best: combine — batch=32 + 100 epochs ⇒ ~11,400 gradient steps ≈ 15% of NH's budget. | Eliminates the L − G gap. ETA NSE for G: 0.64–0.66. | One Colab session per variant (≈3 hr T4). |
| **1.2** | **Drop basin one-hot encoding when using topology features.** Train G+T without `use_basin_id_encoding`. If topology features substitute for the per-basin identity, this could give a real lift. | Tests whether topology features carry *additional* signal beyond basin-ID. | Same Colab session. |
| **1.3** | **Replace mean aggregation with area-weighted aggregation.** `m_v = sum_u (parent_area_u / total_parent_area_v) · msg_u`. Physically grounded. | Likely +0.005 to +0.020 NSE on basins with high-fan-in (~116 of 183). | Code-edit only; same compute. |
| **1.4** | **Make the message function nonlinear.** 2-layer MLP for `W_msg_edge`. | Lets messages express conjunctive features. | Code-edit only. |
| **1.5** | **Fix best-checkpoint selection to use a held-out validation set, not test.** Critical hygiene, even though our 5cond run didn't suffer from it. | Removes a confound for any future work. | Code-edit only. |

### Tier 2 — Architecture upgrades (moderate cost, possible real upside)

| # | Change | Rationale |
|---|---|---|
| **2.1** | **Replace single-hop-per-timestep with K-hop per timestep** (K=2 or 3). Stronger receptive field. | Standard GNN best practice; the model currently has very local connectivity. |
| **2.2** | **Route an interpretable scalar as the message** — explicitly the parent's predicted discharge (or runoff). Side-channel the full h_parent as auxiliary. | Easier learning problem; explicit physical meaning. |
| **2.3** | **Embed discrete topology features (depth, in/out-degree) instead of z-normalizing.** Use `nn.Embedding(max_value+1, d=4)` for each. | Avoids the "quasi-categorical numeric" pathology. |
| **2.4** | **Add DropEdge regularization.** Randomly drop 10–20% of edges per batch. | Standard GNN regularization; improves generalization on small graph datasets. |
| **2.5** | **Replace `tanh(W_out(m))` with a learnable scaled residual** — `scale * W_out(m)`, where `scale` is a single learnable parameter init at 0.01. | Removes the saturation bottleneck. |

### Tier 3 — Bigger redesigns (only if Tier 1+2 isn't enough)

| # | Change | Rationale |
|---|---|---|
| **3.1** | **Two-stream architecture.** Separate hidden-state trajectories for LSTM (local) and graph (upstream). Combine at output. | Disentangles the local and graph contributions; easier to inspect. |
| **3.2** | **Use a Transformer-style cross-attention from upstream to downstream** instead of LSTM message passing. The upstream pool is small (median 4 parents) so attention is cheap. | Modernizes the architecture; gives per-timestep adaptive weighting "for free." |
| **3.3** | **Predict-then-route.** Train an LSTM-only model to predict per-basin runoff (not discharge), then route runoff downstream with a known physical model (Muskingum, simple linear reservoir). | Fully physics-aware; very strong baseline for the hydrology audience. |

### Tier 4 — Diagnostic / paper-narrative

| # | Change | Rationale |
|---|---|---|
| **4.1** | **Run with `use_basin_id_encoding: False` baseline.** If cudalstm-without-one-hot is much worse than the graph variants, the one-hot is doing the work; if it's similar, the one-hot is doing nothing important. | Settles whether the paper should compare against "fair baseline" or "strong baseline." |
| **4.2** | **Probe what the message function actually learned** — gradient norms through W_out, hidden-state alignment between parent and child. Already partly built in `experiments/analysis/analyze_results.py`. | If the message path has near-zero gradient flow, it's dead-on-arrival and Tier-1 changes are needed before any others matter. |

---

## Part 7 — The paper-narrative recommendation

If the goal is "show our added features outperform a standard LSTM," the audit suggests three viable narratives:

**Narrative A — Train it properly, then make claims.**
Fix Tier-1 items 1.1, 1.3, 1.4. Re-run the 5-condition factorial. Plausible outcome: G+T+M reaches NSE ≈ 0.66, beating cudalstm by ~0.01. Small effect but defensible as a *clean* finding. *Risk:* might not actually outperform.

**Narrative B — Comparison with a fair baseline.**
Tier-4 item 4.1: re-run cudalstm *without* basin one-hot. Compare graph variants against that. The graph variants probably do beat cudalstm-without-one-hot, because the topology features and message passing then have something to contribute. *Caveat:* removes the "we beat the standard NH baseline" framing; replaces with "we beat a graph-feature-amenable baseline."

**Narrative C — Honest negative result + physics-aware redesign.**
Publish the 5-condition factorial as a negative result with the rigor it already has (multi-seed, paired contrasts, interaction term, all the right statistical machinery). Use it to motivate Tier-3 item 3.3 (predict-then-route) as the proposed *constructive* contribution. The story: "GNN-on-rivers with hand-designed features doesn't beat strong LSTM baselines; the path forward is hybrid physics-ML."

C is the safest paper but the smallest contribution. A and B are higher-upside but require more compute and risk a null result.

**My recommendation:** start with A. Cheapest, fastest, and even if A fails the run-time investment was small (one Colab session). If A fails, pivot to B + redesigned message passing. C is the last-resort fallback if both A and B come up empty.

---

## Appendix — Source pointers

| Concept | File / lines |
|---|---|
| `DirectedGraphLSTM` class | `experiments/training/train_graph_lstm.py:98-294` |
| Message passing forward | `experiments/training/train_graph_lstm.py:218-289` |
| Mean aggregation | `experiments/training/train_graph_lstm.py:280-285` |
| Attention variant (untested in factorial) | `experiments/training/train_graph_lstm.py:259-279` |
| Sigmoid-gate variant (untested in factorial) | `experiments/training/train_graph_lstm.py:237-258` |
| `compute_topology_features` (the 5 features) | `experiments/training/train_graph_component0.py:98-146` |
| `load_graph_with_features` (edge features) | `experiments/training/train_graph_lstm.py:300-329` |
| `train_epoch` (per-window batching) | `experiments/training/train_graph_lstm.py:431-467` |
| VARIANTS dict | `experiments/training/train_graph_component0.py:80-95` |
| Best-checkpoint test-leakage bug (older trainer only) | `experiments/training/train_graph_lstm.py:635-643` |
