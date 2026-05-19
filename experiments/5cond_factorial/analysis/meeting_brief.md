# Meeting Brief — 5-Condition Factorial Results & Open Questions

**For:** Tomorrow's meeting with the professor.
**Status:** Plain-English summary of what we ran, what we learned, what's unresolved, what to ask.

---

## 1. What we ran (the 5-condition factorial)

We trained 5 different model configurations on the 183-basin Component 0 network from CAMELS-US. 3 random seeds per configuration = **15 training runs total**. Each run was 30 epochs.

The 5 conditions:

| Name | What it is | What it tests |
|---|---|---|
| **L** | NeuralHydrology's standard `cudalstm` (one LSTM per basin, no graph) | The field-standard baseline we want to beat. |
| **G** | Our custom DirectedGraphLSTM, but with the graph "turned off" (no edges) | Architecture-matched control: same code as the graph variants, just without any graph signal. |
| **G+T** | G plus 5 hand-designed topology features (depth in the network, in-degree, etc.) | Does just *labeling* each basin with its position in the network help? |
| **G+M** | G plus message passing (each basin's LSTM also receives info from upstream basins) | Does sharing upstream→downstream information help? |
| **G+T+M** | Both topology features AND message passing | The full model — the one we want to beat L with. |

The paper's goal: **show that G+T+M outperforms L** (i.e., that our added graph features beat a standard LSTM).

## 2. What we found

The exact opposite of what we wanted. From best to worst (median NSE across basins, averaged across 3 seeds):

```
L      = 0.653   ← the simple LSTM baseline
G      = 0.609   ← our architecture with NO graph signal
G+T    = 0.605   ← + topology features
G+M    = 0.583   ← + message passing
G+T+M  = 0.586   ← both
```

Read this top-to-bottom: **every "advanced" feature makes things worse.** The simple baseline L beats our full graph model G+T+M by 0.067 NSE — a substantial gap in hydrology terms.

Cross-seed standard deviations are tiny (0.002–0.017), so the ordering is statistically robust. This isn't noise.

The detailed paired comparisons:
- **L beats G** by +0.050 NSE (paired across all 549 basin × seed comparisons). This is the *biggest* effect in the run.
- **Topology features alone (G+T vs G)** make essentially no difference.
- **Message passing alone (G+M vs G)** slightly hurts.
- **Combining T and M** is sub-additive — adding both helps less than either alone would predict.

## 3. The three things we've investigated about WHY this happened

The original A/B/C run last week showed L > C (the closest analog to today's G+T+M). At the time, the explanation was "the architectures differ — cudalstm and our DirectedGraphLSTM are not the same model, so we can't tell whether graph signal helps or whether the architecture itself is just worse." That's why we ran the 5-condition factorial: by including G (the architecture-matched control with no graph signal), we could separate "architecture difference" from "graph signal effect."

**Finding 1 — Architecture difference: still there.** Even with the graph turned off (G), our model is 0.050 NSE below L. So part of the gap is not about the graph at all — it's about how the two trainers/architectures *work*.

**Finding 2 — Today's follow-up: we don't fully know why L > G.** I dug into this today and tested two specific hypotheses:

- **First guess: maybe G's trainer just gets fewer training updates per epoch than L's trainer does.** Roughly true — L's trainer does ~2,610 gradient steps per epoch, ours does ~14 (because our trainer batches all 183 basins into each step instead of sampling them individually). So over 30 epochs, L gets 78,000 gradient updates and G gets 420.
- **The test:** I trained L with only 420 gradient updates (matching G's count) on my laptop today. Result: L_420 NSE = **0.502**, much *worse* than G's 0.609. So L doesn't beat G because of more steps — actually, given equal steps, G wins by a lot. The "step count" framing was misleading because each of G's steps processes 200× more data than each of L's steps.
- **What this means:** we *still* don't know why L > G. The two trainers see the same total amount of data over 30 epochs (~20 million basin-day examples), and L gets 0.05 NSE more out of it than G does. The mechanism could be: (a) NH's smaller batches give noisier gradients which can help training, or (b) NH samples randomly across basins each step which exposes the model differently, or (c) the cuDNN-optimized LSTM vs our Python-loop LSTM have some subtle numerical difference. We haven't isolated which.

**Finding 3 — Why are the "advanced" variants (G+T, G+M, G+T+M) WORSE than G?** This is the question the paper really hinges on. Two architectural reasons we identified from a code-level audit:

- **Topology features (G+T) are likely redundant.** NeuralHydrology includes a 671-dimensional "which basin am I" one-hot encoding in every input. The 5 hand-designed topology features are 5 out of 676 numbers — less than 1% of the static input. The model can already learn each basin's behavior from the one-hot; the topology features have nothing to add.
- **Message passing (G+M) makes the model harder to optimize.** The graph contribution starts at zero (we initialize the "graph weights" to zero so the model starts as a pure LSTM and has to *learn* to use the graph). With limited training budget, the graph component never gets fully exercised, and the extra parameters are mostly dead weight that hurts more than helps.

## 4. Bottom line in one paragraph

We built a graph-LSTM with three layers of "advanced features" (topology features, message passing, both) and tested whether they beat a standard LSTM baseline. **All three variants are worse than both the baseline AND the no-features control of the same architecture.** We have two leading explanations: (1) the basin one-hot encoding the baseline uses makes our hand-designed topology features informationally redundant, and (2) the message-passing component has design choices (mean aggregation, single-linear message function, zero-init residual) that compound to make it hard to learn under our trainer's gradient budget. There is also a residual architecture gap between our trainer and NeuralHydrology's that we haven't been able to fully attribute to any single cause yet.

## 5. What we still need to examine (in priority order)

1. **Does the L > G gap survive when our trainer runs with smaller batches?** If we drop batch size to 32 instead of 256, our trainer gets ~8× more gradient updates per epoch. If that closes the L > G gap, the gap was about "how the trainer mixes data" rather than anything architectural. *(~3 hours on T4 to test.)*

2. **Does the topology feature G+T help when the basin one-hot encoding is turned off?** If we train G and G+T without the one-hot encoding, and G+T then beats G, we've confirmed the redundancy hypothesis. The paper claim then becomes "graph features help when not competing against a 671-dim per-basin identifier." *(~6 hours on T4 to test.)*

3. **Where on L's learning curve does NSE = 0.609 (our G's level) sit?** L saves checkpoints every epoch; we can just evaluate model_epoch{1, 5, 10, 20, 30}.pt and see how many epochs (and gradient steps) L needed to reach the same NSE that G ends up at after 30 epochs. *(~5 minutes, very cheap.)*

4. **What if we re-design the message-passing mechanism?** The current design has several known weaknesses: mean aggregation ignores parent basin sizes (a 1 km² parent counts as much as a 100 km² parent), the message function is a single linear layer with no nonlinearity, and the "graph contribution" starts at exactly zero and has to grow from nothing. Tier-1 fixes from the architecture audit: replace mean with area-weighted aggregation, make the message function a 2-layer MLP, replace the saturating `tanh` residual with a learnable scaled residual.

5. **Is there a clean physics-aware alternative?** Instead of a learned message-passing graph, we could route upstream-predicted discharge downstream through a known physical model (e.g., Muskingum routing). This is a stronger inductive bias and might be the right baseline to compare against rather than the standard LSTM.

## 6. Questions to ask the professor

**On the headline result:**
1. We expected G+T+M > L; we got L > G+T+M by 0.07 NSE. Given that's the opposite of the paper's intended direction, what should the paper's framing become? Three options I see:
   - (a) Keep the original framing and aggressively engineer until we get G+T+M > L.
   - (b) Reframe as "we identified the conditions under which graph signal helps vs. hurts" — a study of the *mechanism* rather than a claim of superiority.
   - (c) Publish as a clean negative result and use it to motivate a hybrid physics-ML alternative (predict-then-route).
2. How much should we worry about the residual L > G gap (0.050 NSE between standard LSTM and our no-graph control)? Is it acceptable to attribute it to "implementation differences" if we can't pinpoint the mechanism, or do reviewers consistently demand this be cleanly explained?

**On the basin one-hot encoding question:**
3. NH's strong baseline uses a 671-dimensional basin-identity one-hot encoding. Our hand-designed topology features are 5/676 ≈ 0.7% of the static input. Is comparing against the one-hot-enabled L considered a "fair" comparison in this literature, or is the standard practice to evaluate graph methods against a "topology-amenable" baseline (one without per-basin identity)?
4. If we re-run without the one-hot, do you have an intuition for whether the graph variants would catch up to and possibly beat the no-one-hot L?

**On compute and scope:**
5. We have Colab Pro (~100 compute units/month). Each full sweep is ~30 GPU-hours. What's the practical compute ceiling for the paper — can we afford multiple sweeps (re-runs with redesigned architecture, larger seed counts, ablations) or do we need to be selective?
6. Is there an HPC option (university cluster) we should be using instead?

**On the 23-basin pilot:**
7. The earlier 23-basin Texas pilot showed +0.078 NSE for graph features over baseline. The Component-0 scaled run reverses this to −0.07. Should we treat the pilot as "motivating evidence for hypothesis exploration" only, or do you see scientific value in following up on what specifically differed? (One concrete difference: the pilot used warm-started weights from a pre-trained baseline; the 5cond runs were from-scratch.)

**On the architecture:**
8. The current message-passing implementation has known design weaknesses (mean aggregation ignoring parent area, single-linear message function, zero-init residual). Is fixing these now worth the time investment, or is there a more fundamental redesign you'd recommend (e.g., predict-then-route, transformer cross-attention, or sticking with the current design and just running longer)?
9. Are there any recent papers (Kirschstein 2024, Jiang 2025, or others) you'd point to as the right comparison architecture for this kind of work?

**On the testing protocol:**
10. We've been pre-registering each follow-up experiment (writing down hypothesis + success criterion before running, to prevent post-hoc rationalization). Is this overkill for a workshop submission, or is it the right level of rigor?

**On the paper timeline:**
11. Workshop deadlines and the realistic best-case scenario for the paper — what's the latest we'd want to lock the experimental design and start writing?

## 7. Recommended path into the meeting

If I had to commit to one direction before the meeting: **the cheapest informative next experiment is the one-hot ablation (Question 4).** It directly tests the most likely explanation for why G+T = G (topology features being redundant), it can run overnight on Colab, and the result either gives us a paper-narrative pivot (graph signal helps when comparison is fair) or eliminates a hypothesis and forces a redesign decision.

But that's a tactical answer. The strategic question — *which of the three paper framings (engineer it, reframe it, pivot to physics-ML)* — is what I'd most like the meeting to settle.

---

## Quick reference: how to read the files I've written

If the professor wants more detail:

- **`5cond_run_analysis.md`** — just the numbers and what they mean. No methodology, no plan. Read in 5 minutes.
- **`architecture_analysis.md`** — deep technical critique of every design choice. The long version of §3 above. Read in 20 minutes if you want all the engineering details.
- **`testing_framework_proposal.md`** — the proposed 6-step experimental protocol going forward, with compute estimates and decision diagram. Read in 10 minutes if you want to know the plan.
- **`experiments/5cond_factorial/preregistration_step1.md`** — the first follow-up experiment we ran (the matched-budget cudalstm test); has the pre-registration plus the result. Read in 5 minutes if you want to see what "pre-registration" means in practice.
- **`JOURNAL.md`** — running log of everything we've tried and found, with my reasoning. Skip unless you want forensic detail.

The most important file for the meeting is *this one* — `meeting_brief.md`. Everything else is supporting evidence.
