#!/usr/bin/env bash
# Local-first entry point for the local-subgraph batch.
#
# These subgraphs are SMALL (13-23 basins) — each 3-condition × 3-seed sweep
# runs in ~15-30 min on a laptop CPU. No GPU / Colab needed. Colab is reserved
# for the LARGE scale-up runs only (see notebooks/_optional_scaleup/).
#
# This is the methodology step toward the paper: establish whether graph signal
# reappears at small/local scale (the meeting hypothesis: 183 basins is too
# large for the LSTM to benefit from graph structure).
#
# Usage:
#   bash experiments/local_subgraphs/run_all_local.sh
#   bash experiments/local_subgraphs/run_all_local.sh sg_northeast   # one subgraph
#
# Idempotent: completed runs skip. Safe to re-run.

set -e
cd "$(dirname "$0")/../.."   # repo root

PY="${PY:-/Applications/anaconda3/envs/nh/bin/python}"
SEEDS="${SEEDS:-11 13 17}"
CONDITIONS="${CONDITIONS:-L G G_T_M}"
DEVICE="${DEVICE:-cpu}"

SUBGRAPHS=("$@")
if [ ${#SUBGRAPHS[@]} -eq 0 ]; then
  SUBGRAPHS=(sg_midatlantic sg_ohio sg_tennessee sg_southeast sg_northeast sg_texas_pilot)
fi

# 1. (Re)build subgraphs if basin lists are missing.
if [ ! -f experiments/local_subgraphs/basin_lists/subgraph_manifest.csv ]; then
  echo "Building local subgraphs..."
  "$PY" experiments/local_subgraphs/build_local_subgraphs.py
fi

# 2. Sweep each subgraph.
for sg in "${SUBGRAPHS[@]}"; do
  echo ""
  echo "############ SUBGRAPH $sg ############"
  "$PY" experiments/local_subgraphs/run_subgraph_sweep.py \
    --subgraph "$sg" --seeds $SEEDS --conditions $CONDITIONS --device "$DEVICE"
done

# 3. Analyze — produces the loss-distribution invariant table.
echo ""
echo "############ ANALYSIS ############"
"$PY" experiments/local_subgraphs/analyze_subgraphs.py

echo ""
echo "Done. Invariant + contrasts at experiments/local_subgraphs/analysis/INVARIANT.md"
