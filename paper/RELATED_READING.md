# Related Reading — structural & stylistic observations from the field

Purpose: calibrate our draft's *conventions and register* against accepted papers in this exact
subfield. This is NOT a findings summary (see `research_papers.md` for that) — it records how these
papers *structure sections, report compute, present results, and phrase claims*, so our draft reads
as in-community. Sources fetched and read this session (full text where open-access); quotes are
verbatim from the fetched text.

---

## The papers read (full-text structure obtained)

1. **Kratzert et al. 2019** (HESS, open access) — the multi-basin / EA-LSTM CAMELS paper. Our L
   baseline instantiates this paradigm.
2. **Kirschstein & Sun 2024** (ICML, arXiv:2405.19836 HTML) — the GNN-topology null we explain.
3. **Jiang et al. 2025** (ICML, arXiv:2506.05676; PMLR 267:27670--27684) — the physics-guided
   directional fix we operationalize.

---

## Observation 1 — Compute / GPU / wall-clock time: **none of them report it**

The direct answer to "should we report GPU clock time?": **No.** All three papers report *what* was
trained (model count, seed count, dataset size) but give **no GPU spec, no wall-clock, no compute
cost**. Examples:
- Kratzert 2019: reports "48 different trained LSTM-type models", "8 random seeds" — no hardware/time.
- Kirschstein 2024: no compute reporting anywhere.
- Jiang 2025: no compute reporting anywhere.

**Action for our draft:** the ML-conference report *wants* a compute note, but this subfield does not
carry one in the body. Resolution: keep our compute statement minimal and factual (a single stock
model, tens of minutes/run, few dozen runs) as a *reproducibility* signal, not a hardware benchmark.
Do NOT expand it into a compute-resources subsection — that would read as out-of-register here. A
one-line "each run is a single cudalstm, ~tens of minutes on one GPU" is enough and matches the norm
of reporting scale without hardware.

## Observation 2 — Section structure: Methods and Experiments are SEPARATE, Experiments has an
"Experimental Setup" subsection

All three separate the model/formalism (Methods/Methodology) from the empirical work (Experiments):
- Kratzert 2019: `Methods` (2.1--2.6, with **2.6 Experimental setup** as a subsection) then `Results`.
- Kirschstein 2024: `Methodology` (3.1 Data, 3.2 Task) then `Experiments` (**4.1 Experimental Setup**,
  4.2--4.5 the comparisons).
- Jiang 2025: `Proposed Approach` then `Experiments`.

**Validates our split:** Methodology = model + feature + metrics definitions; a separate section for
the empirical protocol. Kirschstein's "4.1 Experimental Setup" is the direct analogue of our
`Experimental Protocol`. Our structure is in-convention. Minor: consider renaming to match the field
term "Experimental Setup" if the venue skews hydrology, or keep "Experimental Protocol" (fine for ML).

## Observation 3 — Results presentation: tables + CDFs, medians across basins, mean±std over seeds

- Kratzert 2019: empirical **CDFs of NSE across basins** (Figs 3--5), plus tables of mean/median NSE
  with ranges; "median NSE" is the emphasized summary. Ensemble variability as "0.67±0.006 (0.71)".
- Kirschstein 2024: tables of NSE **mean and std across cross-validation folds**; network-viz figures.
- Jiang 2025: tables of MSE + directional-sensitivity scores; perturbation-response figures.

**Action for our Results:** our per-basin median Δ + mean±std over seeds is exactly the field norm.
**A CDF of per-basin ΔNSE (or NSE) is the canonical field figure** and would strengthen our Results
(Kratzert's Figs 3--5 are CDFs). Strongly consider one CDF figure. Emphasize **median across basins**
as the headline summary (we already do).

## Observation 4 — How the NULL is phrased (Kirschstein) — blunt, quotable, and structurally mirrors us

Kirschstein's null is stated plainly and repeatedly:
> "model performance shows almost no sensitivity to the choice of graph topology."
> "The river graph topology makes no difference. Even when the model is allowed to learn an optimal
> edge weight assignment, it does not manage to outperform the baseline."
> "the impact of river topology is negligible."

Their **experimental setup is a systematic sweep: 6 adjacency definitions × 3 edge orientations
(downstream / upstream / bidirected) × GNN variants**. This is the *direct analogue* of our
topology-specificity + directionality controls (real vs reversed vs random edges). We can position
precisely: they swept adjacency/orientation on *learned GNNs* and found nothing; we sweep the same
degrees of freedom on a *fixed 1-hop flow feature* and find the real-edge, upstream signal survives.

**Action:** in Related Work / Discussion, quote their null verbatim and note our controls mirror
their sweep — this is the tightest possible positioning, in their own words.

## Observation 5 — Register / contribution phrasing (calibration)

- Kratzert 2019: "we were able to significantly improve performance compared to a set of several
  different hydrological benchmark models"; "a single 'universal' deep learning model can learn both
  regionally consistent and location-specific hydrologic behaviors." Formal, modest hedging.
- Jiang 2025: motivation-before-equation, "moderate-to-heavy formalism", **no formal theorems** —
  derivation/adaptation of PDE frameworks, not theorem-proof. Intuition precedes each equation set.

**Action for our Methodology:** our register matches (formal, modest, equation-with-prose, no
theorems). Jiang confirms that a methodology can be equation-heavy *without theorems* and still land
at ICML — validates our "equations surrounded by prose, matched to claim strength" approach. Do not
add theorems we don't have.

## Observation 6 — Limitations

- Kratzert 2019 has a `Discussion and conclusion` with honest limitation framing: "Treating catchment
  attributes as static is a strong assumption... which is not necessarily reflected in the real world."
- Kirschstein/Jiang fold limitations into Conclusion rather than a dedicated section.

**Action:** a `Discussion`/`Limitations` with honest scope (regional, heuristic edges, modest effect)
is in-convention. A dedicated Limitations section is safe (the ML report wants it); folding into
Discussion is also acceptable in this subfield.

---

## Net actions for the draft (prioritized)

1. **Keep the compute statement minimal** — one line, scale not hardware. Do NOT add a compute
   subsection. (Field norm: none of the three report compute.)
2. **Add a CDF figure** of per-basin NSE / ΔNSE in Results — the canonical field figure (Kratzert).
3. **Position against Kirschstein in their own words** — quote the null; note our controls mirror
   their adjacency×orientation sweep but on a fixed flow feature.
4. **Keep the Methodology equation-heavy but theorem-free** — Jiang validates this lands at ICML.
5. **Section split is correct** — Methods/Experimental-Setup separation matches all three.
6. Consider renaming `Experimental Protocol` → `Experimental Setup` (field term) — optional.

## Bib correction found this session
- Jiang 2025: verified PMLR **267:27670--27684** (the `[verify]` page flag in references.md — now
  confirmed). arXiv:2506.05676. Public code at github.com/HaoyangJiang-WM/PhysicsNFP.

## Not yet read (available if a deeper pass is wanted)
- MC-LSTM (Hoedt 2021, arXiv:2101.05186), Nearing 2021 (WRR), the GNN-theory set (MagNet, Topping,
  Rusch, Bodnar). These inform Related Work depth but are less load-bearing for section *structure*.
  Fetch on demand.
