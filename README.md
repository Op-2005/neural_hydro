# Neural Hydrology — does a streamflow LSTM actually use the river network?

An experimental codebase built on [NeuralHydrology](https://github.com/neuralhydrology/neuralhydrology)
(Kratzert et al., JKU Linz) that asks a single question on CAMELS-US: **when a rainfall–runoff
model improves after being given a river network, what about that network is it actually using?**

## The motivation

Predictive models of physical systems are routinely built on top of a known physical network —
atoms bonded, power lines connected, rivers draining into one another. When such a model gets
better, the gain is usually read as evidence that the model exploited the physical relation the
network encodes. That reading is rarely tested directly, because isolating the network's
contribution means varying the graph while holding the model and its entire training procedure
fixed.

That is what this repository does. Every condition is the **stock `cudalstm`** with a
byte-identical configuration — same hyperparameters, same split, same seeds — and the only thing
that changes is the spatial information appended to the input. So any movement in skill is
attributable to that information and nothing else.

## What we found

**Encoding decides whether the network is usable at all.** Given as a static label — graph
depth, upstream count, upstream area — it is inert: the shared LSTM can already memorize any
per-basin constant, so a fixed descriptor adds nothing. Given as a daily signal — the discharge
of neighbouring basins from yesterday — it helps, and scrambling the dates removes the gain, so
it is real timing information rather than added model capacity.

**But the useful property is proximity, not connectivity.** Swap every true upstream neighbour
for a non-upstream basin at about the same distance and the gain is unchanged. Push the
substitutes further away and it decays. Simply selecting the two nearest gauges, ignoring the
river network entirely, beats the network outright — because the network's own rules pick
neighbours roughly twice as far away. So a performance gain from a physical graph is not, by
itself, evidence that the model learned the physics that graph encodes.

Two scope bounds we state plainly: the nearest-gauge advantage **disappears** once neighbour
discharge has to be predicted rather than observed, so this is a finding about where the
information lives rather than a better forecasting method; and the reliable gain sits **below
median flow**, where upstream precipitation does not reproduce it, leaving catchment storage as
a candidate explanation we do not claim to have identified.

**Scope.** A regional subset of CAMELS-US in the eastern United States, and a river network
**inferred heuristically** from drainage area, elevation and a distance cap — not surveyed
hydrography. Conclusions are about this graph and this daily-forecasting setup.

Effect sizes are what a single input coordinate produces, and they are modest. Every reported
value, with both tails, is in `experiments/topology_ablation/analysis/VERIFIED_NUMBERS.md`,
which is generated from the raw run metrics rather than written by hand.

## How the study is built

- **One variable at a time.** Stock `cudalstm`, byte-identical configs, verified by direct diff
  (`experiments/topology_ablation/analysis/COMPLIANCE.md`).
- **Pre-registration.** Every experiment has a `preregistration_*.md` written *before* the run,
  stating the hypothesis and the falsification criterion. Falsified predictions are reported as
  falsified — several are.
- **Three seeds (11, 13, 17), judged by the weakest.** Significance is never pooled across
  seeds; for a null, the weakest seed is the one least able to reject.
- **Controls, not just conditions.** A shuffled-input capacity control, an oracle upper bound, a
  distance-matched substitution, a random rewire, nearest-gauge selectors, and a no-ML routing
  baseline.
- **Numbers are generated, never transcribed.** `verify_all_numbers.py` regenerates
  `analysis/VERIFIED_NUMBERS.md` directly from the raw run metrics.

## What this is

A research codebase, not a finished product: experiment code, configurations, pre-registrations,
and per-result analysis tables.

It is also a **chronological record**, which means it contains results whose interpretation our
own later controls falsified — most notably a routing mechanism we believed in through July 2026
and retired that August, and a depth gradient that turned out to follow from how the input was
defined rather than from hydrology. Those files are kept because the measurements are correct and
the sequence is the honest history. Each analysis directory's README marks what still stands and
what does not, so read it before quoting any single result file.

## Where to start

**`experiments/topology_ablation/`** is the current study. Its README covers the conditions, the
statistical protocol, and how to reproduce any result; `analysis/VERIFIED_NUMBERS.md` inside it is
the generated single source of truth for every reported number, and the `preregistration_*.md`
files are the hypotheses as written before each run.

`topology_analysis/` builds the basin sets and edge lists the conditions consume. `runs/` holds
the raw outputs and per-run notes. `neuralhydrology/` is the upstream framework, unmodified.
Earlier work is kept for provenance in `experiments/5cond_factorial/` and
`experiments/local_subgraphs/` — its results were discarded as confounded, and the diagnosis is
why the current study is designed the way it is. Each directory has its own README.

## Reproducing

```bash
# train one condition
python neuralhydrology/nh_run.py train --config-file experiments/topology_ablation/configs/L_component0_seed11.yaml

# regenerate every reported number from the runs on disk
python experiments/topology_ablation/verify_all_numbers.py
```

Run outputs land in
`runs/topology_ablation/component0/<condition>_component0_seed<s>/test/model_epoch030/test_metrics.csv`,
which is the ground truth for every figure and table.

## Credits

Built on NeuralHydrology:

```bibtex
@article{kratzert2022joss,
  title = {NeuralHydrology --- A Python library for Deep Learning research in hydrology},
  author = {Frederik Kratzert and Martin Gauch and Grey Nearing and Daniel Klotz},
  journal = {Journal of Open Source Software},
  year = {2022},
  volume = {7}, number = {71}, pages = {4050},
  doi = {10.21105/joss.04050},
}
```

CAMELS-US dataset: Newman et al. (2015), Addor et al. (2017).
