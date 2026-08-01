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
