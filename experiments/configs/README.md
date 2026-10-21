# configs — NH configs for the early baselines

NeuralHydrology YAML files, one per run, for the **pre-ablation** baselines: the 23-basin Texas
pilot (weak and strong variants), the held-out-basin PUB setup, and the first scaled Component-0
baseline. Basin lists come from `../basin_lists/`.

Train one with:

```bash
python neuralhydrology/nh_run.py train --config-file experiments/configs/<file>.yaml
```

The **current study generates its own configs** under `topology_ablation/configs/`, written by its
runners so that every condition is byte-identical apart from the input being tested. Nothing here is
used by the paper.

The distinction worth knowing: the *strong* pilot baseline enables the basin one-hot identity, the
*weak* one does not. That gap is why the current study deliberately keeps the one-hot on — it is the
strongest competitor to any static descriptor, so a fair test has to beat it.
