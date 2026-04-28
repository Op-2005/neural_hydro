#!/bin/bash
# Multi-seed E0.5 sweep launcher. Trains 60-epoch strong baseline at 5 different
# seeds, sequentially. Approx 25 min per seed × 5 seeds = ~2 hrs total.
# Outputs go to runs/lstm_strong_60ep_seed{N}_*/.
# Pre-registered in JOURNAL.md 2026-04-24 next-sessions plan.

set -e
cd "$(dirname "$0")/../.."

PYTHON=/Applications/anaconda3/envs/nh/bin/python
TEMPLATE=experiments/configs/lstm_study_network_strong_60ep_template.yaml
SEEDS=(11 13 17 19 23)

mkdir -p runs/_multiseed_e0_5
for seed in "${SEEDS[@]}"; do
  cfg=runs/_multiseed_e0_5/cfg_seed${seed}.yaml
  sed "s/{seed}/${seed}/g" "$TEMPLATE" > "$cfg"
  echo "=========================================================="
  echo "Multi-seed E0.5 — seed=${seed}  ($(date))"
  echo "Config: $cfg"
  echo "=========================================================="
  $PYTHON neuralhydrology/nh_run.py train --config-file "$cfg"
  echo "=========================================================="
  echo "Seed ${seed} done at $(date)"
  echo "=========================================================="
done

echo "ALL SEEDS COMPLETE"
