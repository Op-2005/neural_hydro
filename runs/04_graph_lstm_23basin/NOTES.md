# Run 04 — Graph-LSTM v1 (historical, undertrained)

**Model.** Early version of the DirectedGraph-LSTM. 23-basin Texas network.
10 epochs. No warm-start; no edge features.

**Result.** Median test NSE 0.329.

**Why it matters.** First attempt at the graph model. Undertrained and without
edge features / warm-start. Demonstrated the forward pass worked end-to-end
but produced a worse result than the non-graph baseline — which drove the
design changes in run 06: warm-start from a trained LSTM, add edge features,
train longer.

**Status.** Historical. Superseded by run 06 and the ablations 07–11. Kept
for completeness; do not cite its NSE.
