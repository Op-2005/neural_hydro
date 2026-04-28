#!/bin/bash
# Multi-seed E0 sweep: 5 seeds × Probe A + Probe B on run-05 strong baseline.
# Cheap (~3 min total). Pre-registered in JOURNAL.md 2026-04-24
# next-sessions plan.

set -e
cd "$(dirname "$0")/../.."
PYTHON=/Applications/anaconda3/envs/nh/bin/python
SEEDS=(11 13 17 19 23 42)

for seed in "${SEEDS[@]}"; do
  echo "=== seed=${seed} ==="
  $PYTHON experiments/probes/e0_self_stabilization.py \
    --baseline-dir runs/05_lstm_23basin_strong_baseline \
    --sigma 0.5 \
    --probe-b-mode t-1 \
    --seed "${seed}" \
    --out-suffix "seed${seed}" 2>&1 | tail -7
done
echo "ALL SEEDS DONE"
