# Neural Hydrology Literature Review
## Topology-Aware Streamflow Prediction — Research Positioning Document
**Baker Lab, UCLA | Compiled April 2026**

---

> **Purpose:** Comprehensive deep-dive of all core literature for the DirectedGraph-LSTM project. Papers are grouped into two sets: (1) the four applied hydrology papers assigned for research positioning, and (2) the five core GNN theory papers from the project folder. Together these nine papers define the full intellectual context.

---

# PART I: Applied Hydrology Papers

---

## Paper 1 — Kirschstein & Sun (ICML 2024)
### "The Merit of River Network Topology for Neural Flood Forecasting"

**Citation:** Kirschstein, N. & Sun, Y. "The merit of river network topology for neural flood forecasting." *Proceedings of the 41st International Conference on Machine Learning*, pp. 24713–24725. PMLR 235, 2024.

---

### 1.1 What They Did

Kirschstein & Sun trained GNNs on the **LamaH-CE** dataset (859 gauges, Central Europe, hourly resolution) and compared forecasting performance under different adjacency definitions — physical upstream-downstream topology, learned edge weights, and no edges at all. The task was framed as an end-to-end single-model problem: unifying discharge prediction and spatial routing within a single GNN, rather than treating them as two-stage processes. Architectures tested: GCN (Kipf & Welling), GAT, and GCNII.

---

### 1.2 The Core Null Result

> "Even when edges were entirely removed, the GNN performed comparably to an MLP, with no tangible benefit from the inclusion of weighted edges that represent physical hydrological relationships."

The learned edge weights correlate with neither of the static physical definitions and exhibit no regular spatial pattern. Furthermore, the GNNs struggle to predict sudden, narrow discharge spikes — the high-frequency events most physically attributable to upstream propagation.

The study does not fully explain the underlying causes of this topological ineffectiveness, leaving the "why" as an open gap that Jiang et al. (ICML 2025) and the present research directly address.

---

### 1.3 Why the Null Result Happened — The Mechanistic Explanation

The failure is not random. Standard message-passing GNNs (GCN, GAT, GCNII) act as **low-pass filters** on the graph signal. The tree-like structure of river networks — high depth, low branching factor — creates two compounding problems:

1. **Over-squashing:** Nodes at high graph distance communicate through exponentially many paths converging at bottlenecks. The entry $(\hat{A}^r)_{is}$ decays exponentially for a binary tree, meaning upstream signals become invisible to downstream outlets after just a few GNN layers (see Topping et al., Section V.2 below).

2. **Directional insensitivity:** Standard message-passing yields similar performance whether edge directions are maintained, reversed, or randomly permuted. The low-pass filter suppresses exactly the high-frequency, directionally-asymmetric signals that upstream→downstream flow dynamics require (see Jiang et al., Section I.3 below).

Kirschstein & Sun exposed the failure without diagnosing the root cause. That diagnosis is the gap your work fills.

---

### 1.4 Dataset and Experimental Details

- **Dataset:** LamaH-CE (Klingler et al., 2021) — 859 catchments, Central Europe, hourly resolution, 34+ catchment attributes including topology data. Notably, this is the first major benchmarking dataset in the CAMELS family to include explicit river network topology.
- **Adjacency definitions tested:** (a) physical upstream-downstream edges, (b) learned adaptive edge weights, (c) no edges (MLP baseline).
- **Conclusion:** No adjacency definition produced measurable improvement over the no-edge baseline.

---

### 1.5 Positioning Against Your Work

Your DirectedGraph-LSTM is **not refuting the null result** — you are operationalizing the fix for the failure mode Kirschstein & Sun exposed. Specifically:

| Kirschstein & Sun Failure | Your Architectural Response |
|---|---|
| GNN treats forward and reverse topology equivalently | Directed edges with explicit parent→child ordering |
| Message passing acts as a low-pass filter | Temporal lagging of LSTM hidden states encodes high-frequency causal signals |
| No topology ≈ topology for prediction | Zero-initialized message weights ensure model cannot underperform baseline at initialization |
| Null result on static graph with scalar edge weights | Dynamic lagged hidden states carry physical memory of upstream conditions |

The key differentiator: where Kirschstein & Sun used GNNs that aggregate spatially across nodes, your model propagates **temporally** — the upstream LSTM hidden state $h_v^{(t-\tau)}$ is the message to downstream basin $u$ at time $t$, where $\tau$ encodes physical travel time. This temporal causal mechanism is absent in any of the GNN architectures they tested.

---

### 1.6 What the Paper Does Not Explain (Your Opening)

The paper explicitly notes it "does not fully explain the underlying causes of this topological ineffectiveness." This silence is the scientific gap that justifies your work's existence. A reviewer who asks "why should your approach work when Kirschstein & Sun showed topology doesn't help?" has a precise answer: the failure was **architectural** (low-pass filtering + directional insensitivity), not **topological** (the physical signal truly not existing). The signal exists; standard GNNs cannot access it.

---

## Paper 2 — Jiang, Wang, Zhu & He (ICML 2025)
### "Topology-aware Neural Flux Prediction Guided by Physics" (PhyNFP)

**Citation:** Jiang, H., Wang, J., Zhu, X., & He, Y. "Topology-aware neural flux prediction guided by physics." *Proceedings of the 42nd International Conference on Machine Learning*, PMLR 267, 2025.

---

### 2.1 Framing Against Kirschstein & Sun

Jiang et al. open by citing Kirschstein & Sun (2024) as establishing the core challenge:

> "GNNs often struggle in modeling physics-based flow dynamics due to their insensitivity to edge directions. In real systems, flow dynamics follow strict physical laws, where local and rapid changes — e.g., turbulent eddies, sharp flow transitions, or abrupt flux variations — only propagate in specific directions. Yet, GNNs typically yield similar performance whether the original edge directions are maintained, reversed, or randomly perturbed."

This directional insensitivity is diagnosed as the proximate cause of Kirschstein & Sun's null result.

---

### 2.2 The Core Theoretical Diagnosis: Low-Pass Filtering

The message-passing mechanism in standard GNNs implicitly acts as a **low-pass filter** — it captures low-frequency patterns (seasonal trends, climate zones) but suppresses high-frequency variations (rapid flow transitions, local surge events). Formally, for the GCN coupling with adjacency $A$:

$$h_i^{(l+1)} = U_l\!\left(h_i^{(l)},\ \sum_{j \in N_{\text{in}}[v_i]} M_l(h_i^{(l)}, h_j^{(l)}, e_{ji})\right)$$

As layers increase, the temporal gradient $\Delta x_i = \frac{1}{t}\sum_{s=1}^t (x_i^{[s]} - \mathbb{E}(x_i))^2$ diminishes for all nodes — high-frequency components are attenuated. The proof is via an **inverse problem construction**: reversing edge directions makes the task ill-posed (inferring upstream from downstream is non-unique and noise-amplifying). Standard GNNs show **similar loss curves** whether edges are forward, reversed, or undirected because they cannot distinguish the directional signal in the first place.

---

### 2.3 The Proposed Framework: PhyNFP

PhyNFP (Physics-guided Neural Flux Prediction) has two components:

#### 2.3.1 Explicit Local Directionality: Discretized Difference Matrices

Replace the standard adjacency matrix with matrices that encode **upstream-downstream gradient operators**. The key discretized update is:

$$\mu^{t+1} = \mu^t + \Delta t \frac{\partial \mu^t}{\partial x} \approx (I + \alpha \hat{D})\mu^t$$

where $\hat{D}$ is the discretized difference matrix with entries:

$$\hat{D}_{ij} = \begin{cases} 1 & \text{if } i = j \\ -1 & \text{if } j \text{ is the upstream node of } i \\ 0 & \text{otherwise} \end{cases}$$

Two enhanced difference matrices are constructed using edge features $e_{ij}$ (spatial distance $\Delta x$ and elevation difference $\Delta z$):

$$D_1 = \frac{1}{\Delta x}\hat{D}, \qquad D_2 = \frac{\Delta z}{\Delta x}\hat{D}$$

where $\Delta x = \varphi_1(e_{ij})$ and $\Delta z = \varphi_2(e_{ij})$ are learned MLP mappings. This encodes both propagation rate (distance) and gravitational effects (elevation gradient).

#### 2.3.2 Implicit Global Physics: Saint-Venant Regularization

Incorporate the 1D Saint-Venant (shallow water) equations as an implicit loss regularizer. The simplified momentum conservation equation is:

$$\frac{\partial u}{\partial t} + u \cdot \frac{\partial u}{\partial x} = -g\frac{\partial z}{\partial x}$$

Discretized in both time and space and incorporated into the GNN training loss, this constrains predictions to respect conservation of momentum. The regularization penalizes predictions inconsistent with the direction of steepest descent and flow inertia.

For traffic flow, the Aw-Rascle equations serve the analogous role, demonstrating that the framework generalizes across flow domains.

---

### 2.4 Key Results

| Metric | River dataset | Traffic dataset |
|---|---|---|
| Improvement over GNN competitors | **+31.6%** | +4.9% |
| Improvement in directional sensitivity | **+96.5%** | +79.9% |

The 31.6% improvement in the river dataset vs. 4.9% in traffic reflects the stronger physical directional constraints in river flow (water flows downhill; there is no reverse direction in steady state).

---

### 2.5 Positioning Against Your Work

Jiang et al. establish that **directionality encoded properly does improve flux prediction substantially**, resolving the Kirschstein & Sun null result. Your work is a **temporal-domain analog** of their spatial-domain argument:

| Jiang et al. | Your DirectedGraph-LSTM |
|---|---|
| Spatial difference operators encode directional gradients | Temporal lagging of hidden states encodes causal travel time |
| Saint-Venant PDE as explicit physics regularizer | LSTM cell state implicitly learns hydrological dynamics from data |
| Requires river geometry (channel distance, elevation) | Requires only directed graph topology (NHDPlus) |
| Operates on instantaneous flux state | Operates on recurrent temporal sequence |
| Architecture-level fix via modified adjacency | Architecture-level fix via directed message passing with lag |

The most important distinction: Jiang et al. require discretized PDEs and physical parameters (Manning coefficient, bed elevation) as explicit inputs. Your approach learns the temporal coupling implicitly through the LSTM dynamics, making it applicable without full hydrodynamic data. This is a genuine scientific trade-off — their model has stronger physics grounding; yours has weaker data requirements and generalizes to basins without detailed geometry.

---

### 2.6 The Falsifiable Prediction

Your project's depth-stratified NSE analysis is the discriminating experiment that Jiang et al. did not run. Their 31.6% average improvement masks how improvement distributes across network position. Your prediction:

> NSE improvement from the graph layer should scale with basin depth — near zero for headwaters (which receive no upstream messages), largest for the outlet basin (which aggregates the full network's upstream signal).

If this pattern holds, it is mechanistic evidence that the graph layer is doing physically meaningful upstream routing, not just acting as a regularizer. If improvement is uniform across depth, the mechanism is not working as intended.

---

## Paper 3 — Kratzert, Klotz, Shalev, Klambauer, Hochreiter & Nearing (HESS 2019)
### "Towards Learning Universal, Regional, and Local Hydrological Behaviors via Machine Learning Applied to Large-Sample Datasets" (EA-LSTM)

**Citation:** Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., & Nearing, G. "Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets." *Hydrology and Earth System Sciences*, 23, 5089–5110, 2019.

---

### 3.1 What They Established

By training a single LSTM model on 531 basins from the CAMELS dataset using meteorological time series data and static catchment attributes, the authors achieved two landmark results:

1. **Better than process-based models calibrated individually per basin.** A single regional LSTM outperforms VIC, mHM, and SAC-SMA when those models are calibrated to each basin separately — the strongest possible baseline.

2. **Cross-basin knowledge transfer via static attributes.** Conditioning the LSTM on static catchment attributes (elevation, aridity, soil type, land cover, etc.) allows a single model to differentiate hydrological behavior across hundreds of diverse basins.

---

### 3.2 The EA-LSTM Architecture

The Entity-Aware LSTM (EA-LSTM) modifies the standard LSTM by using static catchment attributes to **modulate the input gate**:

In a standard LSTM, all gates are computed purely from the dynamic input $x_t$ and previous hidden state $h_{t-1}$. The EA-LSTM replaces the input gate $i_t$ with:

$$i_t = \sigma(s_e)$$

where $s_e = f(\text{static attributes})$ is a learned embedding of the basin's static characteristics. This embedding acts as a **learned basin fingerprint** that modulates what dynamic signals the LSTM pays attention to — a basin with high aridity has different sensitivity to precipitation pulses than a humid basin.

The input gate controls what new information enters the cell state $c_t$. By conditioning it on static attributes, the model learns basin-specific rainfall-runoff relationships without separate calibration.

---

### 3.3 Key Numerical Results

From Table 2 of the paper:

| Model | Median NSE |
|---|---|
| SAC-SMA (individually calibrated) | 0.63 |
| VIC (individually calibrated) | 0.37 |
| LSTM + static concat | 0.69 |
| EA-LSTM (ensemble mean) | **0.73** |

Median NSE of ~0.73 on CAMELS is the **baseline against which your work competes**. Your 23-basin Texas network baseline of ~0.73 is consistent with this established benchmark, confirming your baseline is properly calibrated.

---

### 3.4 The Static Attribute Robustness Finding

Adding Gaussian noise to catchment attributes with variance equal to total input variance:
- Median NSE decreases from ~0.73 to ~0 (catastrophic failure)
- The 1st percentile NSE decreases from 0.13 to -5.87

Static features are most critical for low-NSE basins — precisely those with unusual hydrological behavior that is under-represented in the training distribution. This has a direct implication for your work: **graph topology is most valuable for the same basins where EA-LSTM static attributes struggle**, namely those whose behavior is driven by upstream routing rather than local basin characteristics.

---

### 3.5 The Learned Catchment Similarity

The EA-LSTM's embedding $s_e$ learns a high-dimensional basin similarity metric that corresponds well with prior hydrological knowledge: humid basins cluster together; arid basins form a separate cluster; snow-dominated basins form another. The model discovers the PUB (prediction in ungauged basins) taxonomy without being told it.

---

### 3.6 The Critical Mechanistic Distinction from Your Work

| EA-LSTM cross-basin sharing | DirectedGraph-LSTM sharing |
|---|---|
| Implicit: shared training on static attribute embeddings | Explicit: runtime message passing via upstream hidden states |
| Cross-basin information shared **at training time** via shared weights | Cross-basin information shared **at inference time** via graph edges |
| Effect: regularization + basin similarity learning | Effect: causal upstream routing + travel-time-aware dependencies |
| Cannot use information from yesterday's upstream discharge | Uses lagged upstream $h_v^{(t-\tau)}$ as explicit input |

This distinction is your strongest theoretical differentiator. EA-LSTM knows that basins are *similar to each other* (from static attributes). Your model additionally knows that basin A's *current state affects* basin B's *future state* (from directed graph edges). These are fundamentally different sources of inter-basin information.

---

### 3.7 Role in Your Research Narrative

The EA-LSTM is your **baseline architecture and performance floor**. You are not arguing that EA-LSTM is wrong; you are arguing that it leaves inter-basin temporal routing information unexploited. The EA-LSTM is the state-of-the-art treating basins as independent; your model extends it by adding the directed graph layer on top.

---

## Paper 4 — Nearing, Kratzert, Sampson, Pelissier, Klotz, Frame, Prieto & Gupta (WRR 2021)
### "What Role Does Hydrological Science Play in the Age of Machine Learning?"

**Citation:** Nearing, G.S., Kratzert, F., Sampson, A.K., Pelissier, C.S., Klotz, D., Frame, J.M., Prieto, C., & Gupta, H.V. "What role does hydrological science play in the age of machine learning?" *Water Resources Research*, 57(3), e2020WR028091, 2021.

---

### 4.1 The Central Argument

> "Recent experiments applying deep learning to rainfall-runoff simulation indicate that there is significantly more information in large-scale hydrological data sets than hydrologists have been able to translate into theory or models."

This is a call to action, not just an observation. The paper argues that the hydrological community holds "deeply subjective and non-evidence-based preferences for models based on a certain type of 'process understanding' that has historically not translated into accurate theory, models, or predictions." The evidence: a regional LSTM in ungauged basins outperforms individually-calibrated process-based models in gauged basins.

---

### 4.2 The PUB (Prediction in Ungauged Basins) Framing

The classical open problem in hydrology since the 2003 IAHS Decade on Predictions in Ungauged Basins (PUB): how do you predict streamflow where you have no direct observations?

The ML resolution: if a single LSTM trained on 531 gauged basins matches individually-calibrated process models on completely *ungauged* basins, then there must be **universal structure in precipitation-runoff dynamics** that transcends basin-specific calibration. This universal structure is what ML extracts from the data.

The implication: large hydrological datasets contain more than the information hydrologists have been able to encode. Process-based models represent a lower bound on information extraction, not an upper bound.

---

### 4.3 The Information-Theoretic Argument

The paper frames the ML vs. process-model comparison as a question about **information content**. If a discriminative ML model consistently outperforms a process-based model, the process-based model cannot be fully exploiting the information in the input-output data. This is logically independent of whether the ML model "understands" the physics — it is a statement about data efficiency.

This argument licenses the ML approach scientifically without requiring that LSTM cell states be interpretable as specific hydrological stores. The model can be a black box and still be scientifically valuable if it provides a performance ceiling that process models must explain.

---

### 4.4 What Nearing 2021 Does NOT Say

Nearing et al. are careful to not claim ML replaces process understanding. They argue for "developing a quantitative understanding of where and when hydrological process understanding is valuable in a modeling discipline increasingly dominated by machine learning." The paper is a call to rigorously test whether process knowledge improves ML models — not to abandon process knowledge.

---

### 4.5 Role in Your Research Narrative

Nearing 2021 is your **motivating argument for why the project is scientifically valuable**. The logical chain:

1. *Kratzert 2019:* Cross-basin training extracts universal structure — static attributes are enough to differentiate basins. Establishes the LSTM baseline.
2. *Nearing 2021:* This success implies more information exists in the data than current models exploit — **a call to action**.
3. *Your work:* River network topology is a specific, physically-grounded form of inter-basin information that independent LSTMs (even EA-LSTMs) cannot access by construction. This directly answers Nearing's call.

When writing your introduction, Nearing 2021 is the philosophical anchor: "More information exists; here is a specific form of that information (upstream routing); here is an architecture that accesses it."

---

### 4.6 The PUB Connection to Graph Topology

There is an underexplored connection: PUB performance is precisely where topology should matter most. An ungauged basin has no direct observations, so its predictions must come entirely from *similar basins* (EA-LSTM mechanism) and *connected basins* (your graph mechanism). For basins in a river network, the connected upstream basins provide the strongest physical constraint on expected discharge. Graph topology is therefore the natural complement to the EA-LSTM for improving PUB predictions.

---

# PART II: Core GNN Theory Papers

---

## Paper 5 — Kipf & Welling (ICLR 2017)
### "Semi-Supervised Classification with Graph Convolutional Networks"

**Citation:** Kipf, T.N. & Welling, M. "Semi-supervised classification with graph convolutional networks." *International Conference on Learning Representations*, 2017. arXiv:1609.02907.

---

### 5.1 The Central Contribution

Kipf & Welling derive a specific, computationally cheap graph propagation operator:

$$\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$$

from first principles in spectral graph theory, then show it enables competitive semi-supervised node classification. This operator is the common ancestor of every GNN architecture in your literature — all subsequent papers build on or modify it.

---

### 5.2 The Full Derivation Chain

#### Step 1: Spectral Graph Convolution

Define the graph Laplacian $L = I - D^{-1/2}AD^{-1/2} = U\Lambda U^\top$ (eigendecomposition). The graph Fourier transform of signal $x \in \mathbb{R}^n$ is $\hat{x} = U^\top x$. A spectral graph convolution with filter $g_\theta$ is:

$$g_\theta \star x = U g_\theta(\Lambda) U^\top x$$

**Problem:** Evaluating this requires $O(n^3)$ eigendecomposition and $O(n^2)$ matrix-vector multiplication. Useless at scale.

#### Step 2: Chebyshev Polynomial Approximation

Approximate $g_\theta(\Lambda)$ as a degree-$K$ Chebyshev expansion:

$$g_{\theta'}(\Lambda) \approx \sum_{k=0}^{K} \theta'_k T_k(\tilde{\Lambda}), \qquad \tilde{\Lambda} = \frac{2}{\lambda_{\max}}\Lambda - I$$

where $T_k$ is the $k$-th Chebyshev polynomial (recursively: $T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x)$, $T_0 = 1$, $T_1 = x$). The convolution becomes:

$$g_{\theta'} \star x \approx \sum_{k=0}^{K} \theta'_k T_k(\tilde{L})x, \qquad \tilde{L} = \frac{2}{\lambda_{\max}}L - I$$

Cost: $O(K|E|)$ via the recurrence. $K$-hop local filter (only mixes signals within $K$ hops).

#### Step 3: First-Order Approximation

Set $K=1$ and $\lambda_{\max} \approx 2$, so $\tilde{L} \approx L - I = -D^{-1/2}AD^{-1/2}$:

$$g_\theta \star x \approx \theta_0 x + \theta_1(-D^{-1/2}AD^{-1/2})x$$

Set $\theta = \theta_0 = -\theta_1$ (single parameter to reduce overfitting):

$$g_\theta \star x \approx \theta(I + D^{-1/2}AD^{-1/2})x$$

#### Step 4: The Renormalization Trick

The operator $I + D^{-1/2}AD^{-1/2}$ has eigenvalues in $[0, 2]$. Repeated application causes numerical instability (exploding/vanishing values). Replace:

$$I + D^{-1/2}AD^{-1/2} \;\longrightarrow\; \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} =: \hat{A}$$

where $\tilde{A} = A + I_N$ (add self-loops) and $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$. Eigenvalues now in $[0, 1]$.

#### The GCN Layer (Final Form)

Generalize to multi-feature input $X \in \mathbb{R}^{n \times C}$ with $F$ filters:

$$H^{(l+1)} = \sigma\!\left(\underbrace{\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}}_{\hat{A}}\; H^{(l)}\; W^{(l)}\right)$$

where $W^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$ are learnable weights. For two-layer semi-supervised classification:

$$Z = \text{softmax}\!\left(\hat{A}\;\text{ReLU}\!\left(\hat{A} X W^{(0)}\right) W^{(1)}\right)$$

Trained by cross-entropy over labeled nodes only; full graph structure used in forward pass.

---

### 5.3 Interpretation of $\hat{A}$ Entries

The $(i,j)$ entry of $\hat{A}$ is:

$$\hat{A}_{ij} = \frac{\tilde{A}_{ij}}{\sqrt{\tilde{d}_i \tilde{d}_j}}$$

Higher-degree nodes contribute with smaller weight: hub nodes (or highly-connected river confluences) have their signals diluted. Self-loops ensure $\hat{A}_{vv} = 1/\tilde{d}_v > 0$, so each node retains a weighted copy of its own representation.

---

### 5.4 The WL-Algorithm Connection (Appendix A)

The GCN is a differentiable, parameterized generalization of the 1-dim Weisfeiler-Lehman (WL-1) graph isomorphism algorithm:

$$h_i^{(l+1)} = \sigma\!\left(\sum_{j \in \mathcal{N}_i} \frac{1}{c_{ij}} h_j^{(l)} W^{(l)}\right)$$

with normalization $c_{ij} = \sqrt{d_i d_j}$. The WL-analogy implies: even an *untrained* GCN with random weights produces meaningful node embeddings that reflect graph structure. This is the formal reason why graph structure helps even before learning.

---

### 5.5 Known Limitations (Explicitly Flagged)

1. **Undirected graphs only.** The framework does not naturally support directed edges or edge features. The workaround (bipartite graph representation) is a hack. This is the gap Jiang et al. (2025) closes.

2. **Depth pathology (over-smoothing).** Appendix B shows test accuracy peaks at 2-3 layers and degrades sharply beyond 7 layers without residual connections. Exponential convergence of Dirichlet energy to zero — all node features become identical. GraphCON addresses this.

3. **Scalar edge weights only.** All edges treated with scalar weights from degree normalization. Neural Sheaf Diffusion addresses this by allowing per-edge linear transformation matrices.

---

### 5.6 Relevance to Your Work

$\hat{A}$ is the common ancestor. Your DirectedGraph-LSTM does not use $\hat{A}$ — you bypass the undirected-graph assumption entirely by passing directed LSTM hidden states. But the spectral derivation above explains *why* all downstream GNN pathologies occur: because $\hat{A}$ is a low-pass filter derived from an undirected, uniform-weight Laplacian. Every limitation (over-squashing, over-smoothing, directional insensitivity) flows from these design choices.

---

## Paper 6 — Topping, Di Giovanni, Chamberlain, Dong & Bronstein (ICLR 2022)
### "Understanding Over-Squashing and Bottlenecks on Graphs via Curvature"

**Citation:** Topping, J., Di Giovanni, F., Chamberlain, B.P., Dong, X., & Bronstein, M.M. "Understanding over-squashing and bottlenecks on graphs via curvature." *International Conference on Learning Representations*, 2022. arXiv:2111.14522.

---

### 6.1 The Problem Being Solved

GNNs empirically fail at tasks requiring long-range node interactions. Prior work attributed this vaguely to "bottlenecks." This paper provides the first rigorous geometric theory: **negatively curved edges are the precise cause of over-squashing**, and curvature-based graph surgery is the principled fix.

---

### 6.2 The Jacobian Sensitivity Bound

**Lemma 1 (Jacobian Sensitivity):** For a generic MPNN with bounded derivatives $|\nabla\phi_\ell| \leq \alpha$ and $|\nabla\psi_\ell| \leq \beta$, and nodes $i$, $s$ at graph distance $r+1$:

$$\left|\frac{\partial h_i^{(r+1)}}{\partial x_s}\right| \leq (\alpha\beta)^{r+1} \left(\hat{A}^{r+1}\right)_{is}$$

**This is the central result:** topology, not learning, controls long-range sensitivity. If $(\hat{A}^r)_{is}$ is tiny, **no weight configuration can route signal from $s$ to $i$ effectively**. No amount of learning can compensate for bad graph topology.

When the subgraph induced on $B_{r+1}(i)$ is a binary tree: $(\hat{A}^{r+1})_{is} = 2^{-1} \cdot 3^{-r}$, giving exponential decay in $r$. This is exactly the tree structure of river networks.

**Hydrological implication:** For a basin $v$ located $r$ hops downstream of headwater basin $u$, the maximum influence of $u$'s precipitation on $v$'s discharge prediction decays as $\sim 3^{-r}$. After 5 hops in a binary tributary tree, the influence is less than $0.5\%$ of what it would be in a fully-connected graph.

---

### 6.3 Balanced Forman Curvature

**Definition 1 (Balanced Forman Curvature):** For edge $i \sim j$ in a simple unweighted graph, with degrees $d_i \leq d_j$, let:
- $|\#\Delta(i,j)|$ = number of triangles containing edge $i \sim j$
- $|^\square_i(i,j)|$ = nodes $k \in S_1(i) \setminus S_1(j)$ forming a 4-cycle at $i \sim j$ (no internal diagonal)
- $\gamma_{\max}(i,j)$ = maximal 4-cycle degeneracy factor

**If $\min\{d_i, d_j\} = 1$ (leaf endpoint rule):** $\text{Ric}(i,j) = 0$.

**Otherwise:**

$$\text{Ric}(i,j) = \frac{2}{d_i} + \frac{2}{d_j} - 2 + \frac{2|\#\Delta(i,j)|}{\max(d_i,d_j)} + \frac{|\#\Delta(i,j)|}{\min(d_i,d_j)} + \frac{(\gamma_{\max})^{-1}}{\max(d_i,d_j)}\left(|^\square_i| + |^\square_j|\right)$$

Note: $\text{Ric}(i,j) > -2$ always.

**Geometric intuition via three regimes:**
- **Clique (positive curvature):** Many triangles → $|\#\Delta|$ large → $\text{Ric} > 0$. Like a sphere: geodesics converge. Information has many redundant paths.
- **Grid (zero curvature):** 4-cycles but no triangles → $\text{Ric} = 0$. Like flat Euclidean space. Balanced mixing.
- **Tree (negative curvature):** No triangles, no 4-cycles → $\text{Ric} = 2/d_i + 2/d_j - 2$. Like hyperbolic space: geodesics diverge. Bottleneck.

**Theorem 2 (Lower Bound):** $\kappa(i,j) \geq \text{Ric}(i,j)$ where $\kappa$ is the Ollivier curvature. Balanced Forman curvature is a sharp lower bound on Ollivier curvature, computationally cheaper ($O(|E|d_{\max}^2)$ vs. $O(|E|d_{\max}^3)$).

---

### 6.4 The Main Theorem: Negative Curvature ↔ Over-Squashing

**Theorem 4:** If $\text{Ric}(i,j) \leq -2 + \delta$ for small $\delta$ (with technical conditions on $\delta$ relative to $d_i, d_j$), then there exists a large set $Q_j \subset S_2(i)$ with $|Q_j| > \delta^{-1}$ such that:

$$\frac{1}{|Q_j|}\sum_{k \in Q_j} \left|\frac{\partial h_k^{(\ell_0+2)}}{\partial h_i^{(\ell_0)}}\right| < (\alpha\beta)^2 \delta^{1/4}$$

Negatively curved edges force information to bottleneck through a small number of nodes, squashing the gradients from exponentially many 2-hop neighbors. The more negative the curvature, the worse the bottleneck.

---

### 6.5 Cheeger Constant and Spectral Gap Connection

**Definition (Cheeger Constant):**

$$h_G = \min_{S \subset V} \frac{|\partial S|}{\min\{\text{vol}(S), \text{vol}(V \setminus S)\}}$$

where $\partial S = \{(i,j) \in E : i \in S, j \notin S\}$ and $\text{vol}(S) = \sum_{i \in S} d_i$.

The Cheeger inequality:

$$2h_G \geq \lambda_1 \geq \frac{h_G^2}{2}$$

where $\lambda_1$ is the spectral gap of the normalized Laplacian.

Small $h_G$ (bottleneck) $\Rightarrow$ small $\lambda_1$ (slow mixing) $\Rightarrow$ small $(\hat{A}^r)_{is}$ for inter-community pairs.

**Proposition 5:** If $\text{Ric}(i,j) \geq k > 0$ for all edges, then $\lambda_1/2 \geq h_G \geq k/2$. Positive curvature everywhere gives a lower bound on both the spectral gap and the Cheeger constant.

---

### 6.6 Why Diffusion-Based Rewiring Fails for Bottlenecks

**Theorem 6 (Fundamental limitation of PPR rewiring):** For any subset $S \subset V$ with $\text{vol}(S) \leq \text{vol}(G)/2$, the new Cheeger constant after PPR rewiring satisfies:

$$h_{S,\alpha} \leq \frac{1-\alpha}{\alpha} \cdot \frac{d_{\text{avg}}(S)}{d_{\min}(S)} \cdot h_S$$

The new Cheeger constant is **bounded by a constant times the old one**. DIGL/PPR-based rewiring cannot improve the bottleneck by more than a constant factor. This is because diffusion-based rewiring prioritizes short-distance nodes (intra-community), not the inter-community bottleneck edges that cause over-squashing.

---

### 6.7 SDRF: Stochastic Discrete Ricci Flow

```
Algorithm 1 (SDRF):
Input: graph G, temperature τ, max iterations, optional upper bound C+

Repeat:
  1. Find edge (i*, j*) with minimal Ric(i*, j*)
  2. For k ∈ B₁(i*), l ∈ B₁(j*): compute improvement Δ Ric from adding k~l
  3. Sample (k, l) with probability ∝ softmax(τ · ΔRic); add edge k~l to G
  4. Remove edge with maximal Ric if Ric > C+ (optional)
Until convergence or max iterations reached
```

SDRF adds local 3- or 4-cycles around the most negatively curved edge, reducing the bottleneck surgically without disrupting the overall degree distribution. Compared to DIGL:
- DIGL adds 300–8000% edges (massive densification)
- SDRF adds ~1–8% edges, Wasserstein distance to original degree distribution ≈ 0

---

### 6.8 Application to River Networks

For the 3-node chain graph (node 1 — node 2 — node 3):
- $d_1 = 1$, $d_2 = 2$, $d_3 = 1$
- Edge $(1,2)$: $\min\{d_1, d_2\} = 1$ → $\text{Ric}(1,2) = 0$ (leaf endpoint rule)
- Edge $(2,3)$: $\min\{d_2, d_3\} = 1$ → $\text{Ric}(2,3) = 0$ (leaf endpoint rule)
- $(\hat{A}^2)_{13} = \hat{A}_{12}\hat{A}_{23} = \frac{1}{\sqrt{6}} \cdot \frac{1}{\sqrt{6}} = \frac{1}{6}$

After SDRF adds edge $(1,3)$: $(\hat{A}'^2)_{13} = \frac{1}{3}$. Two-hop influence doubles.

**Key insight for your project:** Most river network edges have $\text{Ric} = 0$ because headwater nodes have degree 1 (leaf endpoint rule). The bottleneck analysis is most relevant for **interior confluence nodes** where multiple tributaries merge (degree $\geq 3$). Your diagnostic should focus on these interior edges when computing curvature on the 23-basin Texas network.

---

### 6.9 Relevance to Your Work

This paper provides the theoretical vocabulary for why GNNs fail on river networks and why your temporal-lag mechanism is a principled alternative to topological rewiring:

| Over-squashing problem | Your solution |
|---|---|
| $(\hat{A}^r)_{is}$ exponentially small for long paths | Explicit directed message with learned lag weight — not filtered by graph powers |
| Bottleneck cannot be fixed by learning | Temporal lag bypasses graph distance entirely |
| SDRF rewires topology to add shortcuts | Your model creates temporal shortcuts via the LSTM's memory |
| Curvature is a pre-processing diagnostic | Curvature maps identify which basins most need upstream signal — predicts where your model should improve most |

---

## Paper 7 — Rusch, Chamberlain, Rowbottom, Mishra & Bronstein (ICML 2022)
### "Graph-Coupled Oscillator Networks (GraphCON)"

**Citation:** Rusch, T.K., Chamberlain, B.P., Rowbottom, J., Mishra, S., & Bronstein, M.M. "Graph-coupled oscillator networks." *Proceedings of the 39th International Conference on Machine Learning*, PMLR 162, 2022. arXiv:2202.02296.

---

### 7.1 The Problem Being Solved

Standard GNNs suffer from **over-smoothing** in deep architectures. Formally (Definition 3.2):

$$E(X^n) \leq C_1 e^{-C_2 n}$$

The Dirichlet energy $E(X) = \frac{1}{v}\sum_{i \sim j}\|X_i - X_j\|^2$ converges exponentially to zero as the number of layers $n$ increases. All node features become identical — the model cannot distinguish any basin from any other after sufficient depth. This prevents building expressive deep GNNs.

**GCN as first-order Euler integration:** The GCN layer $H^{(k+1)} = \sigma(\hat{A}H^{(k)}W)$ can be read as a discrete-time Euler step of the heat equation $\dot{X} = -L_{\text{rw}}X$ (with $\sigma = \text{id}$, $W = I$):

$$H^{(k+1)} - H^{(k)} = (\hat{A} - I)H^{(k)} = -L_{\text{rw}}H^{(k)}$$

Solutions converge exponentially to the kernel of $L_{\text{rw}}$ (constant vectors per connected component). This is over-smoothing by construction.

---

### 7.2 The GraphCON ODE

GraphCON replaces the layer stack with a second-order damped oscillator system:

$$X''(t) = \sigma(F_\theta(X, t)) - \gamma X(t) - \alpha X'(t)$$

In state-space form with auxiliary velocity variable $Y = X'$:

$$\begin{cases} Y' = \sigma(F_\theta(X, t)) - \gamma X - \alpha Y \\ X' = Y \end{cases}$$

**Physical interpretation:**
- Each node $i$ is a **mass on a spring**: $X_i(t)$ is position, $Y_i(t)$ is velocity
- $F_\theta$ is the inter-node coupling (GNN message passing = spring force between connected masses)
- $\gamma > 0$: restoring force coefficient (spring stiffness — prevents drift to zero)
- $\alpha > 0$: damping coefficient (friction — prevents unbounded oscillation)
- $\sigma$: nonlinearity applied to the net force

**Hydrological analogy:** Each basin is a mass with inertia (slow drainage, baseflow recession = damping). The coupling between basins is upstream discharge routing. The oscillator dynamics preserve basin-specific signatures that would be washed out by pure diffusion.

---

### 7.3 The IMEX Discretization

Implicit-explicit symplectic Euler scheme with step size $\Delta t$:

$$Y^{n+1} = Y^n + \Delta t\left[\sigma(F_\theta(X^n, t_n)) - \gamma X^n - \alpha Y^n\right]$$
$$X^{n+1} = X^n + \Delta t\, Y^{n+1}$$

This is an $N$-layer GNN where each layer carries a position state $X^n$ and a velocity state $Y^n$.

---

### 7.4 Recovery of Standard GNNs at Steady State

At a fixed point $Y^* = 0$ (velocity zero), equation for $Y'$ gives:

$$0 = \sigma(F_\theta(X^*)) - \gamma X^* \implies X^* = \frac{\Delta t}{\gamma}\sigma(F_\theta(X^*))$$

Fixed-point iteration:

$$X^{n+1} = \frac{\Delta t}{\gamma}\sigma(F_\theta(X^n))$$

This is exactly the canonical GNN layer (3) up to the scaling $\Delta t/\gamma$. **Standard GNNs are the steady-state iteration of GraphCON dynamics.** GraphCON subsumes all standard GNNs as special cases.

---

### 7.5 The Over-Smoothing Avoidance Proof

**Proposition 3.3:** Over-smoothing (Definition 3.2) occurs for the ODE $\Leftrightarrow$ the hidden states $(X^*, 0) = (c, 0)$ are exponentially stable fixed points of the ODE, for some constant $c$.

**Proposition 3.4 (Key result — requires $\sigma = \text{ReLU}$, $\alpha \geq 1/2$):** For any $c \in \mathbb{R}^m$ with non-negative entries, $(c, 0)$ is a fixed point of (8) but is **NOT exponentially stable**. Small perturbations grow rather than decay.

**Proof sketch:** The linearized system around $(c, 0)$ produces an energy identity with three terms $T_1 + T_2 + T_3$:
- $T_1$: dissipative term — initial perturbations decay as $e^{-2\alpha t}$
- $T_2$: production term — perturbations grow algebraically as $O(\epsilon^2)(1 - e^{-2\alpha t})$
- $T_3$: asymmetry term — indefinite sign, proportional to asymmetry of $\hat{A}$

For $\alpha \geq 1/2$, the combined $T_2 + T_3 \geq 0$ (net production), so perturbations cannot decay exponentially. The fixed point $(c, 0)$ is unstable, preventing over-smoothing by construction.

**⚠️ Critical caveat:** The proof requires $\sigma = \text{ReLU}$ and $\alpha \geq 1/2$. With identity activation $\sigma = \text{id}$ and no damping $\alpha = 0$, the uncoupled system gives pure oscillations $X_i(t) = X_i(0)\cos(t) + Y_i(0)\sin(t)$ — neither over-smoothing nor converging, but potentially unstable in the nonlinear case. In practice you must use $\alpha \geq 1/2$ and ReLU.

---

### 7.6 Gradient Bounds

**Proposition 3.5 (No exploding gradients):** For GraphCON-GCN, the gradient satisfies:

$$\left|\frac{\partial J}{\partial w_k^\ell}\right| \leq \frac{\beta' \hat{D} \Delta t (1 + \Gamma N \Delta t)}{v}\left(\max_i(|X_i^0| + |Y_i^0|) + \max_i|X_i| + \beta\sqrt{N\Delta t}\right)^2$$

With $\Delta t \sim N^{-1}$: the bound is **independent of $N$**. Even for fixed $\Delta t$: growth is at most quadratic in $N$, not exponential.

**Proposition 3.6 (No vanishing gradients):** To leading order in $\Delta t$:

$$\frac{\partial J}{\partial w_k^\ell} = \frac{2\Delta t^2}{v}\sum_{j \in \mathcal{N}_k} \frac{\sigma'(C^{\ell-1}_j) X^{\ell-1}_j (X^N_j - X_j)}{\sqrt{d_j d_k}} + O(\Delta t^3)$$

The gradient is **independent of $N$ to leading order** — it cannot vanish by increasing depth.

---

### 7.7 Key Empirical Results

| Dataset | GCN | GraphCON-GCN |
|---|---|---|
| Texas (H=0.11) | 55.1% | **85.4%** |
| Wisconsin (H=0.21) | 51.8% | **87.8%** |
| Cornell (H=0.30) | 60.5% | **84.3%** |
| ZINC MAE (20 layers) | 0.489 | **0.214** |

For the ZINC regression task, GCN accuracy *worsens* with depth (5→20 layers: MAE 0.442→0.489); GraphCON *improves* monotonically (MAE 0.241→0.214). This is the direct demonstration of over-smoothing vs. over-smoothing prevention.

---

### 7.8 Relevance to Your Work

GraphCON is a *wrapper* around any GNN coupling function $F_\theta$. It is not the architecture you're implementing, but the analysis is relevant in two ways:

1. **Your LSTM already implements the oscillator mechanics implicitly.** The LSTM cell state $c_t$ is analogous to position $X$; the hidden state $h_t$ is analogous to velocity $Y$. The forget gate provides implicit damping; the input gate provides the restoring force. The oscillator dynamics that GraphCON adds explicitly to static GNNs are natively present in the LSTM recurrence.

2. **If you layer GCN-style aggregation on top of LSTM hidden states** in future work, you'll need the GraphCON wrapper to prevent over-smoothing of the aggregated representations. For your 23-basin network, depth is not the bottleneck (shallow network), but this matters for scaling.

---

## Paper 8 — Bodnar, Di Giovanni, Chamberlain, Liò & Bronstein (NeurIPS 2022)
### "Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing in GNNs"

**Citation:** Bodnar, C., Di Giovanni, F., Chamberlain, B.P., Liò, P., & Bronstein, M.M. "Neural sheaf diffusion: A topological perspective on heterophily and oversmoothing in GNNs." *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. arXiv:2202.04579.

---

### 8.1 The Central Idea

GCN implicitly assumes a **trivial sheaf** — scalar edge weights with identity transport. This forces diffusion to converge to a single constant value per connected component (over-smoothing) and makes separation impossible in heterophilic graphs. Equipping the graph with a richer **cellular sheaf** generalizes the propagation operator and fixes both problems simultaneously.

---

### 8.2 Cellular Sheaves: The Core Machinery

**Definition 1 (Cellular Sheaf):** A cellular sheaf $\mathcal{F}$ over graph $G = (V, E)$ assigns:
- A vector space (stalk) $\mathcal{F}(v) \cong \mathbb{R}^d$ to each node $v$
- A vector space $\mathcal{F}(e) \cong \mathbb{R}^{d_e}$ to each edge $e$
- A linear **restriction map** $\mathcal{F}_{v \trianglelefteq e} : \mathcal{F}(v) \to \mathcal{F}(e)$ for each incident node-edge pair $(v, e)$

The space of 0-cochains is $\mathcal{C}^0(G; \mathcal{F}) := \bigoplus_{v \in V} \mathcal{F}(v)$.

**Definition 2 (Sheaf Laplacian):** The sheaf Laplacian $\mathcal{L}_\mathcal{F} : \mathcal{C}^0(G; \mathcal{F}) \to \mathcal{C}^0(G; \mathcal{F})$ acts node-wise as:

$$(\mathcal{L}_\mathcal{F}\, x)_v := \sum_{v \trianglelefteq e} \mathcal{F}_{v \trianglelefteq e}^\top \!\left(\mathcal{F}_{v \trianglelefteq e}\, x_v - \mathcal{F}_{u \trianglelefteq e}\, x_u\right)$$

The normalized sheaf Laplacian $\Delta_\mathcal{F} = D^{-1/2}\mathcal{L}_\mathcal{F} D^{-1/2}$ where $D$ is the block-diagonal degree matrix.

**GCN is the trivial-sheaf special case:** Set $d=1$, $\mathcal{F}_{v \trianglelefteq e} = 1$ for all $(v,e)$. Then $\mathcal{L}_\mathcal{F} = L$ (standard Laplacian) and:

$$I - \Delta_\mathcal{F} = I - \Delta_0 = D^{-1/2}AD^{-1/2} = \hat{A}$$

GCN is Neural Sheaf Diffusion with the most impoverished possible sheaf.

---

### 8.3 Sheaf Diffusion as the Generalized Model

The continuous sheaf diffusion equation:

$$\dot{X}(t) = -\Delta_\mathcal{F}\, X(t), \qquad X(0) = X$$

As $t \to \infty$: $X(t) \to \text{proj}_{\ker(\Delta_\mathcal{F})} X(0)$ (projection onto harmonic space $H^0(G; \mathcal{F}) = \ker(\mathcal{L}_\mathcal{F})$).

For the trivial sheaf: $\ker(\mathcal{L}) = \{\text{constant vectors}\}$ — over-smoothing inevitable.  
For richer sheaves: $\ker(\mathcal{L}_\mathcal{F})$ can be high-dimensional and class-structured — over-smoothing avoided.

---

### 8.4 The Hierarchy of Expressive Power

This is the most important theoretical content. A strict hierarchy of what different sheaf classes can and cannot do in the infinite-time diffusion limit:

#### Class 1: Symmetric Invertible ($H^1_{\text{sym}}$)
$\mathcal{F}_{v \trianglelefteq e} = \mathcal{F}_{u \trianglelefteq e}$ for all incident $(u,v,e)$. Equivalent to weighted graph Laplacians with positive weights. Includes standard GCN.

**Proposition 8:** Can separate two classes if homophilic (each node has at least one same-class neighbor).

**Proposition 9 (Fundamental limitation):** For a connected bipartite graph $G = (A, B, E)$ with $|A| = |B|$ as two classes, $H^1_{\text{sym}}$ **cannot linearly separate the classes for any initial conditions**. Proof: the only harmonic eigenvector is $y_v = \sqrt{d_v}$ (same sign for both partitions). Diffusion converges to a constant — cannot distinguish classes.

**This is the GCN failure mode for heterophilic graphs.**

#### Class 2: Non-Symmetric Invertible ($H^1$)
Allows $\mathcal{F}_{v \trianglelefteq e} \neq \mathcal{F}_{u \trianglelefteq e}$, including sign changes.

**Proposition 10 (Key result):** For the sheaf with $\mathcal{F}_{v \trianglelefteq e} = -\alpha_e$ (class A nodes) and $\mathcal{F}_{u \trianglelefteq e} = +\alpha_e$ (class B nodes):

The harmonic eigenvector is:
$$y_v = \begin{cases} +\sqrt{\sum_{e \ni v}\alpha_e^2} & v \in A \\ -\sqrt{\sum_{e \ni v}\alpha_e^2} & v \in B \end{cases}$$

Classes diffuse to **opposite signs** — perfectly linearly separable for almost all initial conditions.

**This explains why negatively weighted edges work for heterophily:** the transport map $-\mathcal{F}^\top_{v \trianglelefteq e}\mathcal{F}_{u \trianglelefteq e} = -\alpha_e^2 < 0$ forces disagreement between partitions in the diffusion limit, preserving class structure instead of collapsing it.

#### Class 3: Diagonal Invertible ($H^d_{\text{diag}}$, $d \geq C$)
Restriction maps are $d \times d$ diagonal matrices. Can be interpreted as $d$ independent 1D sheaves.

**Proposition 12:** For $d \geq C$ classes, $H^d_{\text{diag}}$ has linear separation power. Uses a "one-vs-all" construction: the $i$-th diagonal dimension separates class $i$ from all others.

#### Class 4: Orthogonal ($H^d_{\text{orth}}$)
Restriction maps are orthogonal matrices $\mathcal{F}_{v \trianglelefteq e} \in O(d)$.

**Proposition 13:** For $C \leq 2d$ classes, $H^d_{\text{orth}}$ has linear separation power. The proof uses rotation matrices from quaternion algebra — for $d=2$, up to 4 classes using complex unit representations; for $d=4$, up to 8 classes using unit quaternion representations.

---

### 8.5 Energy Monotonicity Theorems

**Theorem 15 (For $H^1_+$, positive transport):** With $\sigma = \text{(Leaky)ReLU}$:

$$E_\mathcal{F}(Y) \leq \lambda^*\|W_1\|_2^2 \|W_2^\top\|_2^2 E_\mathcal{F}(X)$$

where $\lambda^* = \max_i(\lambda_i - 1)^2 \leq 1$. Energy monotonically decreases — **over-smoothing is guaranteed** for this sheaf class.

**Theorem 16:** Same bound holds for $H^d_{\text{orth,sym}}$.

**Proposition 17 (The escape route):** For sheaves outside $H^d_{\text{sym}}$, there exist $W_1$ with $\|W_1\|_2 < \epsilon$ (arbitrarily small norm) such that $E_\mathcal{F}((I \otimes W_1)x) > E_\mathcal{F}(x)$. Non-symmetric sheaves can **increase** Dirichlet energy, giving the model control over asymptotic behavior. This is how Neural Sheaf Diffusion avoids over-smoothing: by allowing non-symmetric restriction maps.

---

### 8.6 Spectral Properties of the Harmonic Space

**Proposition 3 (Cheeger-type bound):** For a discrete $O(d)$-bundle, with path-comparison measure $r = \max_{\gamma, \gamma'} \|P^\gamma_{v \to u} - P^{\gamma'}_{v \to u}\|$:

$$\lambda_0^\mathcal{F} \leq \frac{r^2}{2}$$

The spectral gap of the sheaf Laplacian is controlled by how path-dependent the transport is. Path-independent transport $\Rightarrow$ non-trivial harmonic space $\Rightarrow$ $\lambda_0^\mathcal{F} = 0$.

**Lemma 6:** For a connected $O(d)$-bundle, $\dim(H^0) \leq d$, with equality iff transport is path-independent.

---

### 8.7 Practical Implementation: Learning Sheaves from Data

The restriction maps are learned via a parametric function $\Phi$:

$$\mathcal{F}_{v \trianglelefteq e := (v,u)} = \Phi(x_v, x_u) = \sigma(V[x_v \| x_u])$$

followed by reshaping. Three parametrizations:
- **Diagonal:** $\mathcal{F}_{v \trianglelefteq e}$ is diagonal $d \times d$. Cheap, $O(|E|c)$ message passing.
- **Orthogonal:** $\mathcal{F}_{v \trianglelefteq e} \in O(d)$ via Householder reflections. Better theoretical properties, controlled parameterization.
- **General:** Full $d \times d$ matrix. Maximum flexibility but SVD normalization challenges.

**Proposition 18 (Universal sheaf learning):** If all edge-endpoint feature pairs $(x_v, x_u)$ are distinct and $\Phi$ is an MLP with sufficient capacity, $\Phi$ can learn any sheaf over the graph. This is the universal approximation guarantee for sheaf learning.

---

### 8.8 Relevance to Your Work

Neural Sheaf Diffusion is the most theoretically sophisticated architecture in your literature. Its direct relevance:

1. **Physical interpretation is exact for hydrology.** The restriction map $\mathcal{F}_{v \trianglelefteq e}$ encodes how upstream basin $v$'s state is *transformed* before contributing to the shared edge space. In hydrology this transformation is physically motivated: scaling by relative drainage area, lagging by travel time, attenuating by channel losses, sign-flipping for contrasting hydrological regimes. Your LSTM's learned message weights perform a *scalar* version of this; sheaf diffusion generalizes it to a full linear transformation matrix.

2. **Your directed architecture implicitly implements a special case of $H^1$** (non-symmetric sheaves). Because upstream→downstream directionality means $\mathcal{F}_{v \trianglelefteq e} \neq \mathcal{F}_{u \trianglelefteq e}$ by construction, you're already in the class that Proposition 10 proves is capable of heterophilic separation. You're implementing a rank-1 (scalar) instance of a non-symmetric sheaf.

3. **The heterophily result matters for adjacent basins with contrasting hydrology.** Standard symmetric GCN would push their representations together. Your model preserves directional asymmetry by design.

---

## Paper 9 — Baker Lab MPNN Report (March 2026)
### "Mathematical Connectivity of MPNNs Across GCN, Over-Squashing, GraphCON, and Neural Sheaf Diffusion"

**Source:** Internal research progress report, Neural Hydrology Project, Baker Lab UCLA, March 10, 2026.

---

### 9.1 Purpose of This Document

This is an internal synthesis report providing the mathematical unification of Papers 5–8. It serves as the connective tissue for your related-work section. The key insight: **all four papers are targeted modifications of a single canonical MPNN template**, and the central mathematical object unifying all of them is $\hat{A}^r$ — the $r$-th power of the propagation matrix.

---

### 9.2 The Canonical MPNN Template

**Definition 2.1 (MPNN Layer, Node Form):** At layer $k$, every node $v$ updates:

$$m_v^{(k)} = \text{Aggregate}\!\left(\left\{\psi^{(k)}(h_v^{(k)}, h_u^{(k)}) : u \in \mathcal{N}(v)\right\}\right)$$
$$h_v^{(k+1)} = \phi^{(k)}\!\left(h_v^{(k)}, m_v^{(k)}\right)$$

**Matrix Form (for linear $\psi$, sum aggregation, linear $\phi$):**

$$H^{(k+1)} = \sigma\!\left(\hat{A}\, H^{(k)}\, W^{(k)}\right) \tag{*}$$

Each paper modifies exactly one component of $(*)$:

| Paper | Modifies | What changes |
|---|---|---|
| **Kipf & Welling** | $\hat{A}$ | Derives $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ from spectral theory |
| **Over-squashing** | Analysis of $\hat{A}^r$ | Shows topology controls long-range sensitivity; curvature identifies bottlenecks |
| **GraphCON** | Depth composition rule | Replaces iterated $\sigma(\hat{A}HW)$ with second-order damped oscillator ODE |
| **Neural Sheaf** | $\hat{A}$ structure | Replaces scalar edge weights with per-edge linear maps via sheaf Laplacian $\Delta_\mathcal{F}$ |

---

### 9.3 The Central Mathematical Object: $\hat{A}^r$

Every pathology and every fix across all four papers can be expressed in terms of $\hat{A}^r$:

- **Over-squashing** studies $(\hat{A}^r)_{is}$ directly as the sensitivity bound (Lemma 1 of Topping et al.)
- **Over-smoothing** (GraphCON) is about repeated application of $\hat{A}$ driving $H$ into the kernel of $L$ (eigenspace with eigenvalue 1)
- **Sheaf diffusion** replaces $\hat{A}$ with $I - \Delta_\mathcal{F}$, expanding the kernel and thus the long-time limit
- **SDRF rewiring** changes $\hat{A}$ to improve entries of $\hat{A}^r$ across bottleneck edges

---

### 9.4 The Derivation Connectivity Roadmap

```
Canonical MPNN
H^(k+1) = σ(Â H^k W^k)
        ↙               ↘
GCN: Â = D̃^{-1/2}Ã D̃^{-1/2}    Diffusion view
(from Chebyshev approx)          Ẋ = -Δ₀X
        ↙           ↘                  ↓
Over-squashing      GraphCON       Neural Sheaf
|∂h^r_i/∂x_s|  X'' = σ(Fθ)      Ẋ = -Δ_F X
≤ (αβ)^r (Â^r)_is  - γX - αẊ    ker(Δ_F) richer
        ↓           ↓             → no over-smoothing
Cheeger/curvature  IMEX scheme
h_G, Ric(i,j)     steady state
        ↓          = standard GNN
SDRF rewiring
improves (Â^r)_is
```

---

### 9.5 The 3-Node Chain: Worked Example Unifying All Papers

The chain graph $1 - 2 - 3$ demonstrates all four frameworks on a single concrete object.

**Adjacency and propagation matrix:**

$$A = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}, \quad \hat{A} = \begin{pmatrix} 1/2 & 1/\sqrt{6} & 0 \\ 1/\sqrt{6} & 1/3 & 1/\sqrt{6} \\ 0 & 1/\sqrt{6} & 1/2 \end{pmatrix}$$

**Two-hop influence (over-squashing paper):**

$$(\hat{A}^2)_{13} = \hat{A}_{12}\hat{A}_{23} = \frac{1}{\sqrt{6}} \cdot \frac{1}{\sqrt{6}} = \frac{1}{6}$$

Lemma 1 bound: $|\partial h_1^{(2)}/\partial x_3| \leq 1/6$.

**After SDRF adds edge $(1,3)$:**

$$\hat{A}' = \frac{1}{3}\begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix}, \qquad (\hat{A}'^2)_{13} = \frac{1}{3}$$

Two-hop influence doubles — numerical proof of rewiring's effect.

**Curvature on chain:** $d_1 = 1$, $d_2 = 2$, $d_3 = 1$. Both edges have $\text{Ric} = 0$ (leaf endpoint rule). Interior edges with $d \geq 2$ at both endpoints would have negative curvature in larger networks.

**GraphCON first step** (with $\sigma = \text{id}$, $\gamma = 1$, $\alpha = 0$, $X^0 = [1, 0, 2]^\top$, $Y^0 = \mathbf{0}$):

$$Y^1 = \hat{A}X^0 - X^0 = [0.5, 1.2247, 1.0]^\top - [1, 0, 2]^\top = [-0.5, 1.2247, -1.0]^\top$$
$$X^1 = X^0 + Y^1 = [0.5, 1.2247, 1.0]^\top$$

Recovers the GCN output exactly at first step. Differences emerge at $n \geq 2$ via velocity $Y^n$ carrying momentum.

**Sheaf diffusion with negative transport on edge $(2,3)$:** Setting $\mathcal{F}_{2 \trianglelefteq (2,3)} = -1$:

$$(\mathcal{L}_\mathcal{F} x)_2 = (x_2 - (-1)x_3) + (x_2 - x_1) = (x_2 + x_3) + (x_2 - x_1)$$

The sheaf Laplacian enforces $x_2 = -x_3$ at equilibrium (heterophilic separation) rather than $x_2 = x_3$ (over-smoothing to constant).

---

### 9.6 Testable Hypotheses for Your River Basin Network

These five hypotheses apply the four papers' results to your 23-basin Texas network. Each is falsifiable and measurable:

1. **Jacobian decay along path depth.** For GCN-type layers on the basin graph, $|\partial h_v^{(r)}/\partial x_u^{(0)}|$ follows the Lemma 1 bound. Measurable as loss of headwater NSE contribution vs. outlet NSE contribution — headwaters should contribute less signal at equal depth.

2. **Curvature identifies confluence bottlenecks.** High-degree confluence nodes (degree $\geq 3$, non-leaf) have negative Balanced Forman curvature on merging edges. These edges have high betweenness centrality and predict over-squashing severity.

3. **SDRF rewiring improves outlet NSE.** Adding shortcut edges at the most negatively curved confluence (if any exist after accounting for leaf endpoint rule) increases $(\hat{A}^r)_{is}$ for headwater-outlet pairs and measurably improves NSE for the outlet basin.

4. **GraphCON wrapper stabilizes depth.** A GraphCON wrapper around your GCN coupling stabilizes Dirichlet energy across model depth; vanilla GCN shows exponential $E(H^{(k)}) \to 0$. Measurable as basin-feature diversity at output layer as depth increases.

5. **NSE improvement scales with basin depth.** Your primary falsifiable prediction: DirectedGraph-LSTM should show near-zero NSE improvement at headwaters and maximum improvement at the outlet. Uniform improvement would suggest regularization artifact, not causal routing.

---

# PART III: Cross-Paper Synthesis and Research Positioning

---

## Master Positioning Table: All Nine Papers

| Paper | What it establishes | Your project's relationship |
|---|---|---|
| **Kratzert 2019 (EA-LSTM)** | Regional LSTM baseline ~0.73 NSE; static attributes differentiate basins | Your baseline architecture; model does *similar* basins, yours does *connected* basins |
| **Nearing 2021 (WRR)** | More information in hydro data than models exploit; ML outperforms PBMs at PUB | Philosophical motivation: topology is the unexploited information |
| **Kirschstein & Sun 2024 (ICML)** | Null result: naïve GNNs on river topology $\approx$ MLP | Defines the problem; you fix the architectural cause |
| **Jiang et al. 2025 (ICML)** | Directional GNNs with physics regularization +31.6%; low-pass filtering is the cause | Parallel solution in spatial domain; your solution is in temporal domain |
| **Kipf & Welling 2017 (ICLR)** | $\hat{A}$ from spectral theory; GCN layer; over-smoothing at depth | Foundation; all pathologies derive from undirected scalar-weight $\hat{A}$ |
| **Topping et al. 2022 (ICLR)** | Negative curvature $\Rightarrow$ over-squashing; SDRF rewiring as fix | Diagnostic toolkit for your network; temporal lags bypass topological bottlenecks |
| **Rusch et al. 2022 (ICML)** | Second-order ODE prevents over-smoothing; standard GNNs are steady states | LSTM cell state implicitly implements oscillator mechanics |
| **Bodnar et al. 2022 (NeurIPS)** | Non-symmetric sheaves handle heterophily; GCN is trivial-sheaf special case | Your directed edges implement a scalar non-symmetric sheaf; physically motivated restriction maps |
| **MPNN Report 2026 (internal)** | $\hat{A}^r$ unifies all four theory papers; derivation roadmap | Provides vocabulary and roadmap for your related-work section |

---

## The Four-Level Ablation Structure

The relationship between the EA-LSTM baseline and your DirectedGraph-LSTM can be framed as a four-level ablation:

| Level | Architecture | Cross-basin information | What's new |
|---|---|---|---|
| 0 | Single-basin LSTM | None | — |
| 1 | EA-LSTM (Kratzert 2019) | Static attribute similarity | Implicit, training-time only |
| 2 | DirectedGraph-LSTM (your work) | Directed upstream routing | Explicit, runtime, causal |
| 3 | DirectedGraph-LSTM + sheaf restriction maps | Directed + physically parameterized | Full physical transport encoding |

Each level adds a strictly more specific form of inter-basin information. Your contribution is Level 2 → demonstrating that *runtime causal routing* provides a measurable NSE benefit beyond *training-time static similarity*.

---

## The Mechanistic Contrast: Spatial vs. Temporal Directionality

| Jiang et al. (spatial) | Your work (temporal) |
|---|---|
| Encodes direction via discretized gradient operators on $A$ | Encodes direction via lagged LSTM hidden states $h_v^{(t-\tau)}$ |
| Requires explicit geometry (channel distance, elevation) | Requires only directed graph topology (NHDPlus) |
| Physics encoded as PDE regularizer in loss | Physics encoded implicitly in LSTM learned dynamics |
| Fixes instantaneous spatial aggregation | Fixes temporal causal propagation delay |
| Addresses Kirschstein & Sun's spatial architecture | Addresses Kirschstein & Sun's temporal architecture |

Both are valid, complementary solutions. The key scientific claim for your work: **the temporal causal mechanism is both sufficient to produce improvement AND interpretable as physical travel time** — the lag parameter $\tau$ has a direct physical meaning (routing delay between basins) that the spatial difference operators lack.

---

## The Primary Falsifiable Prediction

> **NSE improvement from the DirectedGraph-LSTM graph layer should scale monotonically with basin depth — near zero for headwater basins (depth 0, no upstream neighbors), and maximal for the outlet basin (maximum depth, full network upstream).**

This prediction distinguishes three possible outcomes:

| Observed pattern | Interpretation |
|---|---|
| NSE improvement $\propto$ depth | Graph layer is doing physically correct upstream routing ✓ |
| Uniform NSE improvement | Graph layer is acting as a regularizer; no causal routing |
| Negative correlation | Graph layer is corrupting predictions; initialization or training issue |

The depth-stratified analysis is the discriminating experiment that neither Kirschstein & Sun nor Jiang et al. ran. It is the minimum experiment required to establish that your model is doing what you claim.

---

*Document compiled from primary literature: 9 papers, full text reviewed. All equations transcribed from source material.*