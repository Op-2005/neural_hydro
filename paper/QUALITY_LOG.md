# Paper Quality Log — iterative section QC

Each entry = a revision-workflow pass on a section before the next one is built, so the paper
grows on a verified foundation. Method: ml-paper-writer revision workflow (extract claims → check
evidence → hunt gaps/overclaims → align → revise). Every number re-verified against
`analysis/*.md` at check time.

---

## 2026-08-01 — Abstract (pre-Methodology QC)

**Fact-integrity audit — all numbers verified against source (PASS):**
- oracle +0.038 / p=2.6e-17 (PAPER_TABLE) ✓
- realizable +0.025 (PAPER_TABLE: +0.0253) ✓
- survives shuffled null (SIGNIFICANCE: realizable−null +0.017, p=2.3e-12) ✓
- holds in log-NSE (METRIC_HONESTY: realizable log-NSE +0.027) ✓
- depth gradient 0.002→0.020→0.031→0.044, zero at headwaters (PAPER_TABLE Table 3) ✓
- persists under k=2 pruning (MECHANISM: k=2 realizable +0.025 ≈ full, 3-seed) ✓
- 55–72% oracle recovery (current_implementation §6) ✓
No hallucinated numbers; no internal inconsistencies.

**Gap found (narrative, not factual):** the abstract omitted the strongest 3-seed mechanism
result — **topology-specificity** (forward vs random rewire, +0.034, p=2.3e-14). The abstract
predated the 3-seed mechanism runs. The depth gradient it cited is a *correlational* signature;
topology-specificity is the *causal*, stronger, more reviewer-proof claim and the cleaner mirror
of the Kirschstein null.

**Revision applied:** folded topology-specificity into the "mechanistic" sentence (+0.034,
p=2×10⁻¹⁴, three seeds); kept every prior verified claim; kept directionality OUT (per /crs scope:
weak preference, not a headline). Compiles clean.

**Consistency checks:** abstract's contribution sentence matches SKELETON.md contribution sentence;
terminology consistent ("dynamic upstream-flow signal", "structure-as-flow vs structure-as-label").

**Minor flags (defer):** "match physics-based models" needs kratzert2019 cite in Intro (not
abstract); abstract ~250 content-words — fine for arXiv/journal, check page-limit at venue-fit.

**Verdict:** abstract is a verified foundation. Cleared to build Methodology onto it.

---

## Terminology register (enforce across all sections)
- "dynamic upstream-flow signal" (NOT "upstream feature" / "flow input" interchangeably)
- "static topological descriptor" / "structure-as-label" vs "structure-as-flow"
- "controlled ablation" (the design); "realizable" = predicted-Q deployable model; "oracle" =
  observed-Q upper bound; "null" = shuffled-in-time control.
- "topology-specificity" (real edges ≫ random) vs "directionality" (forward vs reversed — weak).
- L = baseline; L+upQ = oracle; L+upQ\_pred = realizable; L+upQ\_shuf = null.

---

## How the project docs feed the writing (provenance, not content)

**`JOURNAL.md` — the decision-provenance layer (INPUT to writing, not content).** Used to set each
claim's honesty ceiling and catch temporal gaps between decisions and drafted sections. Concrete
examples this project:
- Directionality scoped to "mild aggregate preference, not a headline" — set by the 2026-07-29
  journal entry (3-seed downgrade). The abstract respects that ceiling.
- The pre-Methodology QC caught the missing topology-specificity result *because* the journal
  records the 3-seed mechanism runs landed AFTER the abstract was drafted.
- **Stays OUT of the paper:** the DirectedGraphLSTM failure, ruled-out confounds, the falsified
  lag0 pre-registration. They shaped what we may claim; they are not paper content. The ONE
  negative result that IS reported is the static-topology null — because it is load-bearing (it is
  half of the "structure-as-flow not structure-as-label" contrast).

**`ml-conference-acceptance-criteria-report.md` — a floor-check, not a target.** We clear its
non-negotiables (claims↔evidence, limitations section, reproducibility + compute statement,
significance-via-insight) and then write to the narrative. We do NOT overfit it — its own text
flags rubric-driven, content-poor prose as a rejection signal. The one line we lean on: insight
beats SOTA (permission to lead with the mechanism, not the modest +0.025).

---

## 2026-08-01 — Methodology section (drafted end-to-end) — 4-axis evaluation

Drafted §Methodology (setup/notation, base model, study network, controlled ablation, structural
signal, deployable two-stage, evaluation) with 10 numbered equations + 2 tables. Compiles clean
(6pp). Grounded against code (feature aggregation, cudalstm, edge rule, metric formulas all read
from source this session). ml-math-rigor + ml-paper-writer applied.

### Axis 1 — RIGOR (math-rigor audits)
- **Notation audit: PASS.** Every symbol introduced before use; no collisions. u_i(t) named in
  §ablation (Eq. input), fully defined in §feature (Eq. feature) — forward-reference signposted
  ("defined in §feature"), definition follows immediately. Clean.
- **Correctness audit: PASS, verified vs source.** NSE, log-NSE (eps=1e-3·max(mean,1e-6)), KGE
  (beta=s̄/ō, gamma=CV ratio) all match analyze_metric_honesty.py exactly. Eq. feature dimensional
  check stated (km²·mm/d ÷ km² = mm/d). Edge cases named (headwaters ∅→0 in Eq. feature + Eq. edge).
- **Match-formalism-to-claim: PASS.** LSTM cited not re-derived (no padding). Eq. feature explicitly
  "fixed, directed, single-hop precomputation... not message passing" — honesty ceiling in the math.
- **Flow audit: PASS.** Section opens with a roadmap; each subsection's purpose stated; equations
  surrounded by prose ("Two properties matter...", "central to our claims..."). No formula stands alone.

### Axis 2 — NARRATIVE (story-building)
- The section has a spine: one-variable-changes design → the model it changes → the graph it uses →
  the signal (the contribution) → deployable form → how we measure. Each subsection sets up the next.
- The contribution is front-loaded (§ablation Eq. input = "only the last coordinate changes") and the
  honesty statement (§feature = "not message passing") plants the Kirschstein-contrast the Results pay off.
- Reads as argument, not spec-dump. GOOD.

### Axis 3 — INTEGRITY / scientific standards
- Every constant traces to source (config table byte-identical to disk; metric formulas verified).
- No leakage claim is precise (upstream + lag≥1; own discharge never enters). Stated at Eq. feature.
- Heuristic edges flagged honestly IN the methods (§network), not hidden. Pre-registration stated.
- No overclaim: the feature is described at exactly its true depth. Directionality NOT claimed here
  (it's a control condition in Table 2, results scope it). Honesty ceiling held.

### Axis 4 — PROTOCOL / acceptance-report floor
- Reproducibility: full config table (Table 1), split, seeds, metric defs — an expert could
  reproduce. Clears the report's reproducibility non-negotiable.
- Soundness: controlled design + significance protocol (paired Wilcoxon, 3 seeds) stated.
- Clarity: notation fixed once, consistent terms (matches QUALITY_LOG terminology register).

### Minor flags (defer / for later passes)
- [ ] "predict last 1 day" in Table 1 vs "next-day discharge" in prose — consistent, but confirm
      predict_last_n=1 phrasing reads cleanly to a hydrologist.
- [ ] Eq. model uses w,b for the head; ensure no symbol clash if Results introduces weights (none yet).
- [ ] compute/hardware statement (report wants it) — belongs in a Reproducibility para or appendix,
      not Methods. TODO when we add the checklist.
- [ ] Figure 1 (the static-null vs dynamic-gain contrast) is referenced conceptually; not yet drawn.

### Verdict
Methodology is rigorous, reads as narrative, integrity-clean, clears the reproducibility floor.
Cleared as a foundation for the Results section. No overclaims; every number source-verified.

---

## 2026-08-01 (later) — Methodology multi-pass rigor audit (crs-unleashed)

Re-read the drafted Methodology as a hostile reviewer, 3 gated passes.

### Pass A — CORRECTNESS (caught a real error)
- **One-hot dimension error found + fixed.** Table 1 said "671-dim basin one-hot"; verified against
  code (`basedataset.py:208-209`, `num_classes=len(id_to_int)`, id_to_int built from TRAIN basins)
  that the true dimension is |V|=183 (the study basins), not 671. Fixed Table 1 to "$|V|$-dim
  ($|V|=183$)", consistent with §setup's $\{0,1\}^{|V|}$. The old current_implementation.md "671"
  was wrong; a fabricated number would have shipped. This is why the pass exists.
- All other equations re-verified vs source: NSE/logNSE(eps)/KGE/feature/edge — match code. No
  further correctness issues.

### Pass B — AI-SLOP + REDUNDANCY
- Prose semicolons: ~7 → **0** (converted to periods / restructured; math `\;` + table cells excluded).
- Clause em-dashes: 0 (only correct en-dashes for ranges/names remain).
- Cut the filler roadmap sentence (over-signposting a 1.5pp section).
- De-duplicated "byte-identical" (was stated 3×: §model, §ablation, Table caption → now once each,
  complementary not repeated).
- Tightened the trailing editorial clause ("...not the machinery around it" kept but as a clean
  colon-clause, not "and it is what lets us read...").
- "what the signal can buy" → "bounds the achievable gain".

### Pass C — REVIEWER REREAD (precision)
- **Leakage sentence rewritten.** Was "The lag τ≥1, together with upstream discharge..." — conflated
  two arguments and re-introduced τ≥1 after Eq. fixed τ=1. Now two clean claims: (1) uses upstream
  basins so i's own discharge never enters; (2) lagged so no same-day info. Precise.

### On "what work we build on" (CRS stance)
The grounding is correct and honestly stated. §model cites all three layers we build on: the
foundational LSTM-for-rainfall-runoff (kratzert2018), the multi-basin LSTM paradigm our L baseline
IS (kratzert2019), and the specific stock cudalstm model from the NeuralHydrology model zoo we run
(kratzert2022joss). This is the right grounding — we cite both the paradigm and the software, and
frame our novelty as the ablation FINDING + deployable feature, not a new architecture. No change
needed; the "existing model we build off" is explicit and correctly credited.

### Skill update
Embedded an "AI-slop tells" subsection into ml-paper-writer (semicolon/em-dash overuse, long
multi-clause sentences, filler roadmaps, trailing editorial clauses, synonym drift, over-formal
jargon) with the read-aloud test. Applies to every future prose pass.

**Verdict:** Methodology now correctness-clean (real error fixed), slop-free (0 semicolons/em-dashes/
hedges), reviewer-precise. Compiles clean (6pp). Stronger foundation than the first draft.

---

## 2026-08-01 (later) — Experimental Protocol section (new) + a process finding

**Final Methodology reread:** clean, no material issues. Per instruction, did not manufacture edits.
One structural fix made: split §Evaluation → metric *definitions* stay in Methodology (§Skill
metrics), the comparison *protocol* moved to the new section (no duplication).

**New §Experimental Protocol** (carries the rigor apparatus, not re-listed hyperparameters):
single-variable design verified by config-diff, pre-registration, paired multi-seed comparison,
reproducibility + compute. Iterative audit (same as Methodology):
- **Correctness (verified at write time):** config-diff byte-identical claim TRUE (4 headline
  configs share an identical core-config hash). Pre-reg examples accurate: null control registered
  ≤+0.01, actual null Δ +0.004; lag0 falsification real (lag0 +0.023 < lag1 +0.027). Compute
  "tens of minutes" VERIFIED from output.log (epoch1 00:48 → epoch30 01:29 ≈ 41 min/run).
- **AI-slop:** prose semicolons → 0; em-dashes 0; "intensifier" grep hits were false positives
  ("eVERY"). Clean.
- **Honesty:** the "bespoke graph model cannot offer this control" line references our abandoned
  approach WITHOUT narrating it — failed work stays out. No overclaim.

**PROCESS FINDING (user-raised, important).** The rechecks keep catching errors (671-dim; soft
compute claim) *after* writing, not before. Root cause: verifying at review time, and trusting
DERIVED docs (current_implementation.md said 671) instead of primary source (code builds 183).
Fix baked into both skills: **write-time verification + classify-intent-before-writing** — every
verifiable fact traces to its primary source opened THIS session before it goes on the page;
summaries/READMEs/drafts are not verification; unverifiable → `[verify]`, don't write it. The
review pass is now a backstop that should find nothing.

**Venue-fit flag:** "Code and configurations are released" is true (public repo) but the repo URL
deanonymizes the author — anonymize for a double-blind submission.

---

## 2026-08-02 — Results section (drafted end-to-end) + figures + full protocol audit

Drafted §Results (4 subsections, each answering a question; interpret-don't-paste) with 2
data-driven figures + Table 1. All accumulated protocol applied.

### Write-time verification (the discipline that caught an error)
Every number pulled from its primary source AS written. This caught a real inconsistency:
- **Recovery %**: abstract said "55--72%", I was about to write "55--70%". Neither is stated
  cleanly in a doc — it's derived (realizable Δ / oracle Δ per seed). Computed from source:
  54/72/62% → true range **54--72%**. Fixed BOTH abstract and Results to 54--72% (internal
  consistency). The abstract's "55" was itself slightly wrong. Caught at write time, not review.
- All other numbers cross-checked vs source: static 2x2 (+0.002/p0.67, +0.006/p0.28, +0.016/p3e-8),
  realizable−null (+0.017, p2e-12, CI [+0.011,+0.022]), depth (0.002/0.020/0.031/0.044),
  forward−random (+0.034, p2.3e-14), k2 (+0.025 vs full +0.026). All match. No hallucinations.

### AI-slop: 0 prose semicolons, 0 clause em-dashes, 0 hedges (after fixing 3 semicolons).

### Honesty ceiling held
- Directionality reported as "a mild, aggregate-level effect rather than a firm claim... the robust
  statement is topology specificity, not strict directionality." Not overclaimed.
- Static 2x2 +0.006/p0.28 not dressed up; the one-hot +0.016 used to EXPLAIN the null (redundancy).
- Basin-set precision: full 183 for headline deltas, connected for graph controls, STATED each time
  (Table 1 delta +0.038 all-basin vs mechanism +0.046 connected — kept distinct, not conflated).

### Field calibration applied (from RELATED_READING)
- Median-across-basins headline throughout. Kirschstein null QUOTED verbatim ("almost no
  sensitivity to the choice of graph topology") and positioned as our mirror. CDF figure added
  (Fig cdf, the canonical field figure). No compute padding in Results.

### Figures (generated from stored metrics, verified faithful)
- fig_cdf_nse.pdf: per-basin NSE CDF, L/realizable/oracle (seed 11).
- fig_depth.pdf: realizable ΔNSE by depth, pooled 3 seeds. Medians match PAPER_TABLE exactly.

**Verdict:** Results is number-verified (one cross-doc inconsistency fixed), slop-free, honesty-
clean, field-calibrated. Compiles (9pp) with 2 figures + Table 1.

---

## 2026-08-03 — Introduction (drafted end-to-end) + audit

Drafted §Introduction (6-move structure: LSTM context → GNN-null tension → the static/dynamic
distinction → the controlled ablation → 3 contributions bulleted → honest scope). Full protocol.

### Write-time number verification (all 4 load-bearing numbers vs primary source)
- static topology "at most +0.006, not significant" -> RESULTS_2X2 without-one-hot +0.0057 ✓
- oracle "+0.038, p=2.6e-17" -> PAPER_TABLE +0.0378, p=2.6e-17 ✓
- topology-specificity "+0.034, p=2.3e-14" -> MECHANISM_MULTISEED ✓
- realizable "+0.025, 54-72%" -> PAPER_TABLE +0.0253; recovery 54/72/62% ✓
All identical to abstract/Results (internal consistency held).

### Honesty ceiling
- Directionality: 0 mentions in the intro (correctly scoped OUT of the headline — it's a mild
  aggregate effect, reported only in Results).
- Scope stated explicitly: "effect sizes are modest and the study is regional rather than a national
  benchmark, so we make no state-of-the-art claim." Leads with the insight, not the number.
- Kirschstein null QUOTED verbatim ("no adjacency definition produced measurable improvement",
  "almost no sensitivity to the choice of graph topology") — grounded, not paraphrased.

### AI-slop: 0 prose semicolons, 0 clause em-dashes, 0 hedges.

### Citations verified: kratzert2018/2019 (LSTM paradigm), kirschstein2024 (the null) — claims match
research_papers.md. No fabricated cites.

**Verdict:** Introduction number-verified, honesty-clean (directionality out, scope stated), slop-
free, leads with the contrast-as-explanation. Aligned with abstract + Results. Compiles (10pp).

---

## 2026-08-04 — Full top-to-bottom audit (reviewer + CRS) + completed Related Work/Discussion/Limitations/Conclusion

### Full-paper audit (both hats)
- **Formatting:** 0 undefined refs/citations, 0 overfull hboxes, all cross-refs resolve, both
  figures exist+referenced. Fixed the one real issue: 4 clause-em-dashes in the ABSTRACT (written
  before the em-dash standard was tightened) → now 0 paper-wide. The 3 "unreferenced" equations
  (depth/KGE/realizable) are definitional, referenced by concept 6/4/14× — correct, not a flag.
- **Math strength:** definitional (weighted-mean feature + standard metrics), theorem-free — CORRECT
  for an empirical ablation; Jiang 2025 confirms theorem-free lands at ICML. Over-formalizing would
  be padding. No proofs needed or claimed.
- **Experimental methodology:** strongest axis — byte-identical config verified by diff, pre-reg with
  reported falsification, paired multi-seed + non-parametric significance, every claim 3-seed.
- **Figures:** CDF + depth solid (field-standard). Fig 1 (core-idea diagram) STILL MISSING — the one
  real figure gap; queued.
- **Integrity/narrative/language:** clean paper-wide. 0 semicolons, 0 em-dashes, 0 hedges.

### New sections drafted (completeness pass)
- **Related Work:** 3 paragraphs (DL-for-streamflow / GNN-on-topology / why-message-passing-struggles).
  Kirschstein positioned as our mirror; Jiang as the direction we reach differently; GNN-theory refs
  (topping/rusch/bodnar over-squashing + low-pass) ground the "why GNNs fail" point. Context+difference,
  not a dump.
- **Discussion:** resolves the null (label inert / flow specific), connects Kirschstein↔Jiang, states
  the transferable principle as a HYPOTHESIS this instantiates (not overclaimed to other domains).
- **Limitations:** all 5 real caveats verified present — regional/no-SOTA, inferred edges, modest
  effect, weak directionality, 3-seed. Honest, not apologetic.
- **Conclusion:** contribution restated + national-scale/NHDPlus forward pointer.

All 3 audited: 0 slop, all citations in bib, honesty ceiling held (directionality scoped down,
transferable principle hedged). Compiles clean (12pp).

**Verdict:** paper is now COMPLETE in prose (all sections drafted). Remaining: Fig 1 (diagram),
author/venue metadata (user), a final full read-through pre-submission.
