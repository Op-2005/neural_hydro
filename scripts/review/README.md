# Review tooling

Supporting files for the `review` skill (`.claude/skills/review/SKILL.md`), the top-level
adversarial review stack for `paper/main.tex`.

## Files

| File | Purpose |
|---|---|
| `setup_crossvendor.sh` | Installs [poldrack/ai-peer-review](https://github.com/poldrack/ai-peer-review) and merges this project's prompts. Idempotent; never overwrites API keys. |
| `crossvendor_prompts.json` | Domain-adapted prompts replacing the tool's neuroscience defaults — scientific ML, GNNs, spatiotemporal forecasting, hydrology. |

## Usage

```bash
./setup_crossvendor.sh --check              # readiness report
./setup_crossvendor.sh                      # install tool + prompts
./setup_crossvendor.sh --install-prompts    # refresh prompts only
```

Then in Claude Code, just say **`review`**.

## Why the prompts are replaced

The upstream tool ships neuroscience/brain-imaging prompts. Ours are rewritten for this
paper's actual failure surface: graph operators and adjacency conventions, temporal leakage in
time-series features, spatial leakage between upstream and downstream basins, paired-vs-unpaired
statistics, and whether correlational routing results are stated in causal language.

They also instruct reviewers to treat items missing from extracted PDF text as *clarification
requests* rather than assert the work was not done — the dominant false-positive mode when a
model sees mangled equations and tables.

## Editing the prompts

The tool substitutes placeholders via `str.format()`, so **literal JSON braces in the prompts
must be doubled** (`{{` / `}}`). Single braces that are not real placeholders raise `KeyError`
at runtime. After editing, verify:

```bash
python3 -c "
import json; p=json.load(open('scripts/review/crossvendor_prompts.json'))['prompts']
p['review'].format(paper_text='x')
p['metareview'].format(reviews_text='x')
p['concerns_extraction'].format(model_names='a', first_model='a', second_model='b',
                                meta_review_text='x', model_mapping='m')
print('prompts format cleanly')"
```

## Cost note

The cross-vendor leg sends the full paper to each selected model. Three models on a ~12-page
paper is a handful of API calls, but it is not free and it is a per-vendor charge. The in-house
passes cost nothing extra.
