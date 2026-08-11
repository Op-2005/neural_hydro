---
description: Run the full adversarial review stack on the paper (prose, math, technical panel, cross-vendor, citations, verification) and return one ranked findings table.
---

Read `.claude/skills/review/SKILL.md` and follow it exactly.

Arguments (optional): $ARGUMENTS — a section name to scope the review to, a file path to
review instead of `paper/main.tex`, or `--no-crossvendor` to skip pass 4.

With no arguments, review the whole paper at `paper/main.tex`.
