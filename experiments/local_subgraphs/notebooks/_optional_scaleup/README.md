# Optional scale-up notebook — NOT needed for the local batch

The local-subgraph batch runs on CPU in ~15-30 min — use
`experiments/local_subgraphs/run_all_local.sh`, not this notebook.

This Colab notebook exists only for the eventual LARGE scale-up: once a winning
configuration is identified at local scale, re-run it on a bigger basin set
(or many subgraphs at once) where GPU speed actually matters. Colab compute
units are reserved for that, not for the fast local iteration loop.

To use it for scale-up: open in Colab, set T4 GPU, Run all. It reuses the same
sweep + analysis scripts as the local runner.
