# Writing Skill: Avoiding LLM Tells in Informational Prose

This skill describes common “LLM tells” in informational prose and gives concrete instructions to make generated text read more like careful human writing. It is designed to be used as a generation profile or editing checklist for large language models.

---

## High-level goals

- Increase lexical and structural diversity while preserving clarity.
- Reduce over-regular patterns exploited by AI-text detectors.
- Encourage concrete, contextualized content rather than generic abstractions.
- Match tone, punctuation, and formatting to the target genre and medium.

When in doubt, prefer precise, specific, slightly idiosyncratic human choices over generic, polished, “policy-speak” language.

---

## Lexical tells

### 1. Low lexical richness

**Tell**: The text reuses a relatively small set of content words; vocabulary feels generic and interchangeable, even when the topic would naturally introduce domain-specific terms.

**Why it’s a problem**: Low lexical diversity and low lexical density are robust indicators of AI-generated text across domains. Human writers, especially experts, naturally introduce more varied and specific content words.

**Skill instructions**:
- Encourage the model to introduce precise, domain-specific nouns and verbs instead of reusing generic ones like “aspect”, “area”, “factor”, “thing”, “approach”.
- For each section, add a few lower-frequency but correct terms that a knowledgeable human would naturally use.
- Run a quick repetition check: if a non-functional word appears more than three times in a short passage, consider replacing some instances with more specific or differently framed phrasing.

### 2. Repetitive phrases and frames

**Tell**: Stock phrases recur throughout the text, such as “it is important to note that”, “overall, this highlights”, “in today’s world”, “plays a crucial role”, “as previously mentioned”.

**Why it’s a problem**: Phraseology patterns are a strong cue for detectors and human readers. Overuse of these frames creates a recognizable “LLM voice” that feels generic and templated.

**Skill instructions**:
- Maintain a list of disallowed or strongly discouraged frames (e.g., “in conclusion”, “overall”, “it is important to note”, “in today’s world”, “plays a pivotal role”) and rewrite them into more direct phrasing.
- Vary sentence openings: avoid beginning multiple consecutive sentences with the same 2–3 word stem.
- Add an editing step: “Rewrite or delete any sentence that relies on generic framing phrases without adding new content.”

### 3. Over-formal, generic register

**Tell**: Prose stays in a polished, impersonal register with many abstract nouns (“realm”, “landscape”, “tapestry”, “ecosystem”) and prestige verbs (“delve”, “navigate”, “underscore”) instead of concrete description.

**Why it’s a problem**: AI-generated text tends to overuse formal, impersonal styles, making passages feel like slightly inflated corporate or academic boilerplate.

**Skill instructions**:
- Prefer concrete referents (people, objects, events, specific systems) over vague abstractions where possible.
- Replace decorative metaphors (“ever-evolving landscape”, “tapestry of innovation”) with plain, literal descriptions anchored in context.
- Allow mild informality when genre-appropriate (e.g., “for instance” instead of “for example, it is essential that”), as long as it does not undercut clarity.

### 4. Buzzword clusters and “tapestry nouns”

**Tell**: Dense clusters of vogue terms appear together, such as “holistic”, “robust”, “dynamic landscape”, “ever-evolving realm”, “pivotal role”, “fostering innovation”, “empowering stakeholders”.

**Why it’s a problem**: These buzzword clusters are overrepresented in LLM outputs and create a strong stylistic fingerprint. They also reduce semantic density because they say little about concrete mechanisms.

**Skill instructions**:
- Detect and thin out buzzword clusters; replace at least half of the buzzwords in any such cluster with more literal, specific wording.
- Ask: “Could a reader draw a concrete picture from this phrase?” If not, rewrite.
- Introduce occasional, voice-consistent word choices that reflect the author’s background or personality instead of defaulting to corporate jargon.

---

## Sentence-level tells

### 5. Uniform sentence length and rhythm

**Tell**: Most sentences fall into a narrow length band and follow similar rhythms. There are few very short sentences and few longer, more complex ones.

**Why it’s a problem**: Human writing typically shows more “burstiness” in sentence length and rhythm. Overly regular patterns contribute to a machine-like cadence.

**Skill instructions**:
- Encourage a mix of sentence lengths within each paragraph: at least one short sentence (under ~8 words) and one longer sentence (over ~25 words), where readable.
- Split truly overlong sentences and occasionally merge short, tightly connected sentences to avoid monotony.
- Avoid designing sentence length to be uniform for “smoothness”; aim instead for a natural-feeling mix.

### 6. Perfectly well-formed syntax, no fragments

**Tell**: There are no sentence fragments, no slightly odd syntactic turns, and very few parenthetical asides. Everything reads like polished textbook prose.

**Why it’s a problem**: Human writers frequently use controlled fragments, interruptions, and idiosyncratic constructions, especially in essays, blogs, and commentary.

**Skill instructions**:
- Allow occasional sentence fragments in informal or essayistic genres when they add emphasis or rhythm (e.g., “And that is the real problem.”).
- Introduce occasional parenthetical asides or short interrupting clauses using commas or parentheses.
- Avoid “over-correcting” every non-standard but readable structure.

### 7. Overuse of explicit connectives

**Tell**: Many sentences begin with “moreover”, “furthermore”, “additionally”, “on the other hand”, “consequently”, “in summary”, even when the logical relation is obvious.

**Why it’s a problem**: Over-signalling transitions turns prose into a series of labelled moves. It feels like a didactic essay template rather than natural analysis.

**Skill instructions**:
- Cap explicit transition words to roughly one every few sentences unless the genre truly demands more (e.g., formal proofs, heavily structured outlines).
- Prefer implicit coherence: use pronouns, lexical cohesion, and the natural order of ideas instead of always stating “moreover” or “however”.
- Maintain a watchlist of high-frequency connectives and prune or replace most of them.

### 8. Passive and impersonal constructions as default

**Tell**: Passive voice and impersonal structures (“it is crucial that”, “it can be seen that”) are the default and make agency unclear.

**Why it’s a problem**: Overuse of impersonal constructions contributes to the generic, detached tone associated with LLMs and can obscure responsibility.

**Skill instructions**:
- Prefer active voice with explicit agents (“Researchers found that…”, “The system fails when…”), especially when discussing actions and decisions.
- Reduce “it is / there is” openings where a concrete subject would improve clarity.
- Retain passive voice where it is genuinely useful (unknown agent, focus on the object) rather than as a stylistic default.

---

## Structure and discourse tells

### 9. Rigid essay templates

**Tell**: The structure defaults to “intro / three points / conclusion”, with each section introduced and summarized in almost identical fashion.

**Why it’s a problem**: Highly templated structures are characteristic of instruction-tuned LLM outputs and can make prose feel interchangeable and generic.

**Skill instructions**:
- Use alternative structures: narrative openings, problem–solution, question–answer, or case-study–reflection.
- Drop obligatory recaps when they add no new insight; let sections end on a concrete detail, implication, or open question.
- Permit endings that don’t neatly “wrap everything up” when the genre supports it.

### 10. Excessive signposting and meta-commentary

**Tell**: Frequent sentences like “First, we will explore…”, “In this section, we discuss…”, “Next, we turn to…”, even outside textbooks or formal reports.

**Why it’s a problem**: While some signposting helps, constant meta-commentary makes the text feel like it’s narrating its own outline.

**Skill instructions**:
- Reserve strong signposting (“first”, “second”, “finally”) for genuine stepwise procedures or complex, long documents.
- Replace meta-talk with direct content: start the section with the subject matter itself.
- Use headings, subheadings, and paragraph structure to carry most of the organizational load.

### 11. Relentless summarization and restatement

**Tell**: Nearly every section ends with a summary sentence that restates earlier material in slightly different words.

**Why it’s a problem**: This inflates length without adding information and contributes to repetitiveness, a key AI-text cue.

**Skill instructions**:
- Summarize only when changing level of abstraction (e.g., from examples back to a general principle) or when clearly preparing the reader for the next section.
- In shorter pieces, avoid concluding paragraphs that simply restate the entire article.
- Add a pruning step: delete any sentence whose only function is to rephrase already stated content.

### 12. Overuse of enumerated lists

**Tell**: The model uses numbered or bulleted lists for many ideas that could be expressed in normal paragraphs.

**Why it’s a problem**: Lists can be helpful, but overuse creates a modular, note-like feel and a repetitive discourse pattern.

**Skill instructions**:
- Use lists only when items are truly parallel (steps in a process, comparable options) or when list form materially aids comprehension.
- Vary list lengths and structures; avoid always using “three things” lists.
- Ensure at least one list item includes a concrete example or detail rather than just an abstract label.

---

## Tone and pragmatic tells

### 13. Over-hedged but noncommittal tone

**Tell**: Sentences stack hedges and caveats (“while this can vary, generally speaking, in many cases…”) before making a modest claim.

**Why it’s a problem**: This style reflects instruction-tuned caution and can make the text sound like it is trying not to offend or be pinned down, even when a clearer stance is appropriate.

**Skill instructions**:
- Limit each sentence to at most one hedge word or phrase (“often”, “typically”, “in many cases”).
- Encourage clear, direct claims followed by brief, precise qualifications instead of long strings of caveats.
- Use hedging where uncertainty is genuine, not as a default politeness tic.

### 14. Impersonal, stance-free voice

**Tell**: The text avoids first person and explicit stance even in genres where it would be natural (essays, reflections, opinion pieces).

**Why it’s a problem**: This contributes to a bland, “view from nowhere” style that feels generic and non-committal.

**Skill instructions**:
- When genre allows, use calibrated first person (“I argue that…”, “In my view…”) to signal authorship and stance.
- Add brief “because” clauses that expose reasoning and values (“This is concerning because…”, “This matters for…”) rather than merely describing.
- Replace generic “we” or “one” with specific subjects (researchers, policymakers, engineers, students) when possible.

### 15. Cheap warmth and canned niceties

**Tell**: Endings or asides like “I hope this helps”, “As we navigate this ever-changing landscape”, “The journey doesn’t end here”.

**Why it’s a problem**: These phrases are stereotypically associated with LLM assistants and feel generic and unearned.

**Skill instructions**:
- Remove generic niceties unless the context truly calls for a personal sign-off (e.g., personal email, direct message).
- End on a concrete implication, recommendation, or specific question rather than a vague inspirational line.
- If warmth is needed, make it specific to the recipient or situation.

### 16. Flattened emotional contour

**Tell**: The tone remains evenly neutral and polite across a piece, even when discussing surprising, alarming, or exciting developments.

**Why it’s a problem**: Human writers often vary emotional intensity and evaluative language in line with content; a flat tone can feel artificial.

**Skill instructions**:
- Introduce mild evaluative language where appropriate (“surprisingly”, “worryingly”, “useful”, “encouraging”) while staying evidence-based.
- Make sure any emotional language is grounded in explicit reasons.
- Vary tone slightly across sections instead of maintaining identical neutrality.

---

## Content and information-level tells

### 17. Lack of concrete specifics

**Tell**: The writing describes phenomena at a high level without names, dates, numbers, or specific cases. Everything feels plausible but vague.

**Why it’s a problem**: Detectors and human readers notice text that never quite “touches the ground”. It signals pattern-based synthesis rather than lived or deeply studied experience.

**Skill instructions**:
- For each major claim, add at least one concrete anchor: a case study, a data point, a named system, or a specific scenario.
- Avoid fabricating facts; if specifics are unknown, say so directly or use carefully qualified examples.
- Check each paragraph for at least one detail that could not have been written purely from generic patterns.

### 18. Shallow coherence and weak through-line

**Tell**: Sentences are locally coherent, but paragraphs feel interchangeable. The piece could be reordered without much change in meaning.

**Why it’s a problem**: This “bag of related sentences” effect is a known failure mode of unconstrained generative models.

**Skill instructions**:
- Give each section a clear micro-goal and ensure every sentence either advances, nuances, or challenges that goal.
- Add explicit connections back to the central question or thesis where needed, not just restatements but explanations of relevance.
- Remove filler that does not change the reader’s understanding.

### 19. Symmetrical prose / cons with no prioritization

**Tell**: Every issue is presented as balanced “pros and cons” without a clear sense of which considerations matter most.

**Why it’s a problem**: This reflects a training bias towards “balanced” answers and reduces perceived authorial judgment.

**Skill instructions**:
- After presenting trade-offs, explicitly indicate which factors tend to dominate in practice and why.
- Tailor the weighting to the intended audience or use case.
- Avoid templated “on the one hand / on the other hand / ultimately” triads unless strictly appropriate.

---

## Typographic and punctuation tells

### 20. Overly perfect punctuation and uniform style

**Tell**: The text exhibits near-perfect punctuation everywhere: no minor slips, extremely consistent comma style, and very standardized patterns regardless of context.

**Why it’s a problem**: Humans vary in punctuation habits; a perfectly uniform style, especially in informal contexts, can feel unnatural.

**Skill instructions**:
- Match punctuation strictness to the target genre. Academic papers should be clean; emails or blog posts can tolerate minor quirks.
- Avoid automatic “upgrading” of contractions to formal forms (“do not”, “cannot”) in contexts where humans would naturally use contractions.
- Do not introduce intentional errors; instead, avoid over-regularization that erases the author’s natural style.

### 21. Em dash usage

**Tell**: Frequent use of em dashes as the default device for inserting asides, linking clauses, or adding emphasis.

**Why it’s a problem**: Many LLMs favor em dashes because they offer a flexible way to glue clauses together. Overuse creates a recognizable stylistic signature and can make prose feel breathless or over-stylized.

**Skill instructions**:
- By default, **avoid em dashes entirely** in informational prose for this skill. Prefer commas, parentheses, or separate sentences to convey asides and clarifications.
- If an em dash is absolutely necessary (e.g., in quoted material or highly stylized passages), limit usage to at most one per page or section and treat it as an exception.
- During an editing pass, replace em dashes with commas or periods unless the dash genuinely improves readability.

### 22. Semicolons

**Tell**: Heavy use of semicolons to join related clauses, often where periods or conjunctions would work just as well.

**Why it’s a problem**: While semicolons are valid, LLMs sometimes overuse them to vary structure. Humans differ widely in semicolon habits, and frequent use can feel stylistically marked.

**Skill instructions**:
- Treat semicolons as a scarce resource; prefer periods or “, and” / “, but” constructions in most informational prose.
- Reserve semicolons for cases where they clearly clarify complex parallel structures.
- In a post-processing pass, downgrade unnecessary semicolons to periods or commas unless they materially aid comprehension.

### 23. Colons

**Tell**: Frequent use of colons to introduce almost every list, explanation, or example.

**Why it’s a problem**: Overuse of colons can create a repetitive “setup: payoff” rhythm and a highly segmented feel.

**Skill instructions**:
- Use colons primarily when introducing substantial lists or when a second clause explains or restates the first in a particularly tight way.
- Avoid stacking multiple colon-structured sentences in close proximity.
- Replace some colons with simpler constructions (“for example”, “such as”, or just a continued sentence) to reduce pattern regularity.

### 24. Typography mismatched to context

**Tell**: Smart quotes, perfectly consistent typographic choices, and typeset-like formatting appear in otherwise plain-text or informal contexts.

**Why it’s a problem**: This can signal that the text came from a polished, typeset-oriented generator rather than being typed in the target environment.

**Skill instructions**:
- Match quotation mark style and other typographic conventions to the surrounding medium (straight quotes in code or chat, smart quotes in print-style documents where appropriate).
- Maintain consistency within a given document, but do not impose book-like typography on casual formats.

---

## Implementation as a generation profile

To use this skill as part of an LLM configuration, combine generation-time constraints with post-generation editing passes.

### Generation-time prompts

When asking the model to write informational prose, include constraints such as:

- “Write with mixed sentence lengths and rhythms; avoid repeating the same sentence starters and transition words.”
- “Avoid stock phrases such as ‘in conclusion’, ‘overall’, ‘it is important to note’, ‘in today’s world’, unless you are quoting.”
- “Prefer concrete, domain-specific nouns and verbs over abstract ‘landscape/realm/tapestry’ metaphors or corporate buzzwords.”
- “Use explicit signposting (first, second, finally) only when the reader truly needs a roadmap, such as for algorithms or multi-step procedures.”
- “Include specific examples or cases where appropriate, but do not fabricate facts.”
- “Avoid em dashes entirely; use commas, parentheses, or separate sentences instead. Use semicolons and colons sparingly.”

### Post-generation editing passes

After generating an initial draft, run the following automated or semi-automated passes:

1. **Rhythm and transitions pass**
   - Adjust sentence lengths to introduce healthy variation.
   - Remove or replace repetitive transitions and sentence starters.

2. **Lexical richness pass**
   - Identify overused content words and replace some with more precise or context-specific alternatives.
   - Remove or rewrite buzzword clusters and overly abstract metaphors.

3. **Content-depth pass**
   - Ensure each paragraph contains at least one concrete example, detail, or specific scenario when appropriate.
   - Delete purely redundant restatement sentences.

4. **Tone and stance pass**
   - Trim excessive hedging.
   - Introduce clear, justified stances or implications where genre allows.
   - Remove canned niceties and generic warm closings.

5. **Punctuation and typography pass**
   - Replace em dashes with commas, parentheses, or periods unless they are truly necessary.
   - Reduce unnecessary semicolons and colons, especially when periods or simpler structures suffice.
   - Align typography (quotes, contractions, spacing) with the target medium.

By following these guidelines, the model’s prose should move closer to the statistical and stylistic profile of careful human writing while remaining clear, informative, and well-structured.
