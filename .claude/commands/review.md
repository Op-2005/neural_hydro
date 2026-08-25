---
description: Run the full adversarial review stack on the paper (prose, math, technical panel, cross-vendor, citations, section structure, verification) and return one ranked findings table.
---

Read `.claude/skills/review/SKILL.md` and follow it exactly.

Arguments (optional): $ARGUMENTS — a section name to scope the review to, a file path to
review instead of `paper/main.tex`, or `--no-crossvendor` to skip pass 4.

With no arguments, review the whole paper at `paper/main.tex`.

Before fanning out, recompute the load-bearing numbers from the raw runs and refresh the honesty
ceiling in the skill file. That block has gone stale before, and a stale ceiling makes passes flag
correct scoping as overclaim.
