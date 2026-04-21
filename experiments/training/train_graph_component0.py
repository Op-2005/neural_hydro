"""Scaled graph-LSTM trainer for Component 0 (183 basins).

This is the HYPOTHESIS.md §6 Step-3 script: train baseline-warm-started
DirectedGraph-LSTM variants on the 183-basin eastern-US network that has
meaningful n-per-depth-stratum (51 at depth 2, 16 at depth 3).

Intentionally parameterized via CLI so we can sweep seeds/variants without
editing code, which is the repo-hygiene fix flagged in the earlier review.

Three variants to run, each × multiple seeds:
  * --variant frozen     : graph+edges, LSTM+head frozen (clean test of clause i/iii)
  * --variant warm       : graph+edges, full finetune (for comparison to prior 23-basin result)
  * --variant gcn_lowpass: mean-aggregation only, no edge features, no direction —
                           the low-pass stand-in required by §4 item 3
                           (over-squashing control / Kirschstein-style null baseline)

Smoke-test mode (--epochs 2) reports wall-clock per epoch so we can set a
realistic compute budget for the full sweep.

Usage:
    /Applications/anaconda3/envs/nh/bin/python experiments/train_graph_component0.py \\
        --variant frozen --seed 42 --epochs 15
"""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Repo root is three levels up from experiments/training/<script>.py.
# Second insert keeps experiments/training on sys.path so we can import
# train_graph_lstm (same directory).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.evaluation.utils import load_basin_id_encoding
from neuralhydrology.utils.config import Config

from train_graph_lstm import (
    DirectedGraphLSTM,
    load_graph_with_features,
    load_basin_data,
    warm_start_from_baseline,
    train_epoch,
    evaluate,
)

ROOT = Path(__file__).parent.parent.parent
BASIN_FILE = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_basins.txt"
EDGE_FILE = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_edges.csv"

HIDDEN_SIZE = 64
DROPOUT = 0.4
LR = 1e-3
SEQ_LENGTH = 30
BATCH_SIZE = 256

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
LOGGER = logging.getLogger(__name__)
DEVICE = torch.device("cpu")


def find_component0_baseline():
    """Find the most recent successful Component 0 baseline run."""
    cands = sorted(ROOT.glob("runs/lstm_component0_baseline_*"))
    for d in reversed(cands):
        if list(d.glob("model_epoch*.pt")):
            return d
    return None


VARIANTS = {
    # Ablation dimensions: use_edge_features, use_diff, frozen, attn, sigate, gcn-style
    "warm":         dict(edge_feat=True,  diff=False, frozen=False, attn=False, sigate=False),
    "frozen":       dict(edge_feat=True,  diff=False, frozen=True,  attn=False, sigate=False),
    "gcn_lowpass":  dict(edge_feat=False, diff=False, frozen=False, attn=False, sigate=False),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--basin-file", default=str(BASIN_FILE))
    parser.add_argument("--edge-file", default=str(EDGE_FILE))
    parser.add_argument("--baseline-run", default=None,
                        help="Path to NH baseline run (auto-detected if omitted)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 2 epochs and report per-epoch wall-clock")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--tag", default="",
                        help="Extra tag for the run directory name")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 2

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    variant = VARIANTS[args.variant]
    basin_file = Path(args.basin_file)
    edge_file = Path(args.edge_file)

    # Resolve baseline
    if args.baseline_run:
        baseline_run = Path(args.baseline_run)
    else:
        baseline_run = find_component0_baseline()
    if baseline_run is None:
        LOGGER.error("No Component 0 baseline run found. Train it first with "
                      "lstm_component0_baseline.yaml")
        sys.exit(1)
    LOGGER.info(f"Baseline run: {baseline_run}")

    timestamp = datetime.now().strftime("%d%m_%H%M%S")
    tag = f"c0_{args.variant}_seed{args.seed}"
    if args.smoke_test:
        tag += "_SMOKE"
    if args.tag:
        tag += f"_{args.tag}"
    run_dir = ROOT / f"runs/graph_{tag}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Output directory: {run_dir}")

    basin_ids = [l.strip() for l in open(basin_file) if l.strip()]
    n_basins = len(basin_ids)
    edges = load_graph_with_features(edge_file, basin_ids)
    LOGGER.info(f"Basins: {n_basins}   Edges: {len(edges)}")

    cfg = Config(baseline_run / "config.yml")
    scaler = load_scaler(baseline_run)

    id_to_int = load_basin_id_encoding(baseline_run) if cfg.use_basin_id_encoding else {}

    LOGGER.info("Loading train-period data...")
    x_d_train, x_s, y_train = load_basin_data(cfg, scaler, basin_ids, "train", id_to_int)
    LOGGER.info(f"Train tensor shape: {x_d_train.shape}   Static dim: {x_s.shape[1]}")
    LOGGER.info("Loading test-period data...")
    x_d_test, _, y_test = load_basin_data(cfg, scaler, basin_ids, "test", id_to_int)
    LOGGER.info(f"Test tensor shape: {x_d_test.shape}")

    n_dyn = x_d_train.shape[3]
    input_size = n_dyn + x_s.shape[1]

    model = DirectedGraphLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        edges=edges,
        n_basins=n_basins,
        n_targets=1,
        dropout=DROPOUT,
        initial_forget_bias=cfg.initial_forget_bias,
        use_edge_features=variant["edge_feat"],
        use_diff_term=variant["diff"],
        use_attention=variant["attn"],
        use_sigmoid_gate=variant["sigate"],
    ).to(DEVICE)
    LOGGER.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if not args.no_warm_start:
        ckpts = sorted(baseline_run.glob("model_epoch*.pt"))
        if ckpts:
            LOGGER.info(f"Warm-starting from: {ckpts[-1]}")
            warm_start_from_baseline(model, ckpts[-1])
        else:
            LOGGER.warning("No baseline checkpoint to warm-start from.")

    if variant["frozen"]:
        for p in model.lstm_cell.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = False
        LOGGER.info("Frozen LSTM + head; only W_msg_edge and W_out will train")

    trainable = [p for p in model.parameters() if p.requires_grad]
    LOGGER.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,} of "
                 f"{sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam(trainable, lr=LR)

    # Pre-training eval (sanity: should ~match baseline if warm-start worked)
    LOGGER.info("Pre-training test NSE (sanity check):")
    pre = evaluate(model, x_d_test, x_s, y_test, basin_ids)
    pre_med = float(np.nanmedian(list(pre.values())))
    LOGGER.info(f"  median NSE = {pre_med:.3f}")

    epoch_times = []
    loss_history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, x_d_train, x_s, y_train, optimizer, BATCH_SIZE)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        loss_history.append(loss)
        LOGGER.info(f"Epoch {epoch}/{args.epochs}  loss={loss:.5f}  "
                     f"wall={elapsed:.1f}s")
        torch.save(model.state_dict(), run_dir / f"model_epoch{epoch:03d}.pt")

    # Final eval
    final = evaluate(model, x_d_test, x_s, y_test, basin_ids)
    final_med = float(np.nanmedian(list(final.values())))
    final_mean = float(np.nanmean(list(final.values())))
    LOGGER.info(f"Final median NSE: {final_med:.3f}   mean: {final_mean:.3f}")

    pd.DataFrame([{"basin": b, "NSE": v} for b, v in final.items()]).to_csv(
        run_dir / "test_metrics.csv", index=False)

    run_config = {
        "variant": args.variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "n_basins": n_basins,
        "n_edges": len(edges),
        "baseline_run": str(baseline_run),
        "smoke_test": bool(args.smoke_test),
        "pre_training_median_nse": pre_med,
        "final_median_nse": final_med,
        "final_mean_nse": final_mean,
        "loss_history": loss_history,
        "epoch_wall_seconds": epoch_times,
        "mean_epoch_wall_seconds": float(np.mean(epoch_times)),
        "input_size": input_size,
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    LOGGER.info("=" * 70)
    LOGGER.info(f"DONE. variant={args.variant}  seed={args.seed}  median_nse={final_med:.3f}")
    LOGGER.info(f"Mean epoch wall-clock: {np.mean(epoch_times):.1f}s  "
                 f"({np.mean(epoch_times)/60:.1f} min)")
    if args.smoke_test:
        full_epochs = 15
        projected = full_epochs * np.mean(epoch_times)
        LOGGER.info(f"SMOKE-TEST PROJECTION: full run ({full_epochs} epochs) = "
                     f"{projected:.0f}s = {projected/60:.1f} min")
        LOGGER.info(f"  For 3 seeds × 3 variants = 9 runs: "
                     f"{9 * projected / 3600:.1f} hours total CPU")
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()
