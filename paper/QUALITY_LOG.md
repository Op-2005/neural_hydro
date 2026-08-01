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
