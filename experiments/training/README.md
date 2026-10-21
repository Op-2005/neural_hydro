# training — the custom graph-LSTM (abandoned architecture)

The `DirectedGraphLSTM` and its ablation variants: a per-timestep loop over an `nn.LSTMCell` with
directed message passing from each basin's upstream parents. Variants cover edge features, an
attention weighting, a sigmoid gate, a frozen-LSTM mode, warm-starting, and basin-identity on/off.

**This architecture was abandoned, and the reason is the most important thing in this folder.**
Comparisons against a plain multi-basin LSTM turned out to be uninterpretable: the custom trainer
processed every basin per step, so it took orders of magnitude fewer gradient updates than the
tuned `cudalstm` and was still undertrained when training ended. Two further confounds compounded
it — the basin one-hot already lets the model memorise anything basin-specific, and freshly
initialised input weights for added features are noise an undertrained model never learns to
suppress.

The post-mortem is `../5cond_factorial/analysis/architecture_analysis.md`. The current study's
design follows directly from it: use the stock trainer, change only the input, and all three
confounds disappear at once.

The plain-LSTM baselines never needed anything here — they run through
`neuralhydrology/nh_run.py` with a config from `../configs/`.
