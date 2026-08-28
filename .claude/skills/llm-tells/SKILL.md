---
name: llm-tells
description: Detect and remove the stylistic fingerprints of LLM-generated prose — colon "setup:payoff" rhythm, em-dash and semicolon overuse, uniform sentence length, sentence-initial connectives, buzzword clusters, signposting, and relentless restatement. Invoke for "does this sound like AI", "make this sound human", a writing-quality pass, or before submitting anything written with model assistance.
---

# Removing LLM tells from prose

Full catalogue of 24 tells in `references/reference.md`. Load it only when you need a tell's
rationale; the measurements below are the working tool.

## Measure before editing

Impressions are unreliable here. Every tell in the catalogue is countable, and this project has
twice "fixed" prose that measurement showed was already fine. Run these against the target first.

```bash
# prose em dashes (exclude table cells and comments)
grep -n ' --- ' FILE | grep -v '&' | grep -v '^\s*[0-9]*:%'
# prose semicolons
grep -c '[a-z]; [a-z]' FILE
# sentence-initial connectives
grep -cE '(^|\. )(Moreover|Furthermore|Additionally|Consequently|Overall|Notably|Importantly)' FILE
```

For sentence-length variance, colon density, and repeated openings, strip LaTeX/markup first, then
compute. Targets for informational prose:

| Metric | Healthy | Tell |
|---|---|---|
| Sentence-length stdev | > 10 | < 7 means uniform cadence |
| Sentences under 8 words | 5–15% | ~0% means no rhythm relief |
| Colon-bearing sentences | < 8% | > 12% is the "setup: payoff" fingerprint |
| Paragraphs stacking 2+ colons | rare | the specific pattern to hunt |
| Prose em dashes | ~0 | any cluster reads as machine-written |
| Same 2-word sentence opener | ≤ 3 | ≥ 5 is visible |

## The tells that actually fire in practice

Ranked by how often they survive other review passes:

1. **Colon rhythm.** The single most durable tell, because each instance looks fine alone. Fix by
   converting some to commas, some to a plain relative clause, some to two sentences. Vary the
   repair; converting all of them the same way just trades one pattern for another.
2. **Em dashes as the default aside.** Prefer a comma, a parenthesis, or a period.
3. **Semicolons as clause glue.** Reserve for genuinely parallel structure.
4. **Repeated sentence openers.** "We therefore…" five times reads as a template.
5. **Restatement closers.** A sentence whose only job is to rephrase the paragraph it ends.

## Academic register — where the catalogue does NOT apply

The source catalogue targets blogs and essays. Applying it wholesale to a paper causes damage.
**Ignore these for academic prose:**

- *Sentence fragments for rhythm* (#6) — wrong register entirely.
- *First person "I argue"* (#14) — a multi-author paper uses "we"; that is correct, not a tell.
- *Contractions* (#20) — papers should not use them.
- *Loose typography* (#24) — papers should be typeset cleanly.
- *Evaluative language, "surprisingly"/"worryingly"* (#16) — conflicts with claim discipline. A
  result's significance is carried by its statistics, not by an adverb.
- *Deliberate hedge removal* (#13) — a hedge protecting a scoped claim is doing evidentiary work.
  Cut a hedge only after checking what it protects. This project cut one and had to restore it.

**Do apply:** #1 lexical richness, #2 stock frames, #5 sentence rhythm, #7 connectives, #10
signposting, #11 restatement, #17 concrete specifics, #21 em dashes, #22 semicolons, #23 colons.

## Rules

- **Measure, fix, re-measure.** Report the before and after count.
- **Do not fix what the measurement says is clean.** Prose passes on this project have twice
  removed content that was carrying evidence, reading it as style.
- **Never introduce errors to look human.** The goal is varied and specific, not sloppy.
- **A number, name, or date beats any stylistic edit.** The strongest anti-tell is concrete content
  that could not have been generated from a pattern.
