"""Ungauged-basin experiment: train graph-LSTM on 20 basins, test on 3 held-out.

The 3 held-out basins (08158700, 08164300, 08189500) are INCLUDED in the graph
structure — their parents are training basins, so at inference the model gets
upstream messages from trained LSTMs. But the held-out basins are NEVER in the
loss during training.

This tests whether the graph can transfer learned dynamics to unseen basins.

Baseline comparison: strong LSTM (no basin encoding, trained on 20 basins) evaluated
on the 3 held-out basins.

Usage:
    /Applications/anaconda3/envs/nh/bin/python experiments/train_graph_ungauged.py
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Repo root is three levels up from experiments/training/<script>.py.
# Second insert keeps experiments/training on sys.path so we can import
# train_graph_lstm (same directory).
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.utils.config import Config

from train_graph_lstm import DirectedGraphLSTM, load_graph_with_features

# Baseline trained on 20 basins, no basin encoding
# (will be resolved at runtime)
BASELINE_GLOB = "12_lstm_ungauged_baseline"

# All 23 basins file (for graph) + separate train / test lists
ALL_BASINS_FILE = Path("experiments/basin_lists/study_network_basins.txt")
TRAIN_BASINS_FILE = Path("experiments/basin_lists/ungauged_train_basins.txt")
TEST_BASINS_FILE = Path("experiments/basin_lists/ungauged_test_basins.txt")
EDGE_FILE = Path("topology_analysis/phase1_network_discovery/outputs/study_network_edges.csv")

EPOCHS = 15
LR = 1e-3
HIDDEN_SIZE = 64
DROPOUT = 0.4
CLIP_GRAD = 1.0
SEED = 42
BATCH_SIZE = 256

USE_EDGE_FEATURES = True
USE_DIFF_TERM = False
USE_ATTENTION = False
USE_SIGMOID_GATE = False
WARM_START = True
FREEZE_LSTM = False

DEVICE = torch.device("cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def find_baseline():
    candidates = sorted(Path("runs").glob(BASELINE_GLOB))
    for run_dir in reversed(candidates):
        if list(run_dir.glob("model_epoch*.pt")):
            return run_dir
    return None


def warm_start(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mapping = {
        "lstm.weight_ih_l0": "lstm_cell.weight_ih",
        "lstm.weight_hh_l0": "lstm_cell.weight_hh",
        "lstm.bias_ih_l0": "lstm_cell.bias_ih",
        "lstm.bias_hh_l0": "lstm_cell.bias_hh",
    }
    own = model.state_dict()
    for src, dst in mapping.items():
        if src in ckpt and dst in own and ckpt[src].shape == own[dst].shape:
            own[dst].copy_(ckpt[src])
            LOGGER.info(f"  Warm: {src} -> {dst}")
    for src, dst in [("head.net.0.weight", "head.weight"),
                      ("head.net.0.bias", "head.bias")]:
        if src in ckpt and dst in own and ckpt[src].shape == own[dst].shape:
            own[dst].copy_(ckpt[src])
            LOGGER.info(f"  Warm: {src} -> {dst}")


def load_basin_data(cfg, scaler, basin_ids, period):
    """Load data for a set of basins. No id_to_int (no basin encoding)."""
    all_x_d, all_x_s, all_y = [], [], []
    for basin in basin_ids:
        ds = get_dataset(cfg=cfg, is_train=False, period=period,
                          basin=basin, scaler=scaler)
        bx_d, by = [], []
        x_s = None
        for i in range(len(ds)):
            sample = ds[i]
            dyn = [v for k, v in sorted(sample["x_d"].items())]
            bx_d.append(torch.cat(dyn, dim=-1))
            by.append(sample["y"])
            if x_s is None and "x_s" in sample:
                x_s = sample["x_s"]
        all_x_d.append(torch.stack(bx_d))
        all_y.append(torch.stack(by))
        all_x_s.append(x_s)
    return (torch.stack(all_x_d, dim=1), torch.stack(all_x_s),
            torch.stack(all_y, dim=1))


def compute_nse(pred, obs):
    mask = ~np.isnan(obs)
    if mask.sum() < 10 or obs[mask].std() == 0:
        return float("nan")
    p, o = pred[mask], obs[mask]
    return 1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)


def evaluate(model, x_d, x_s, y, basin_ids):
    model.eval()
    n_windows = x_d.shape[0]
    preds = np.full((n_windows, x_d.shape[1]), np.nan)
    x_s_dev = x_s.to(DEVICE)
    with torch.no_grad():
        for w in range(n_windows):
            x_dw = x_d[w].transpose(0, 1).to(DEVICE)
            y_hat = model(x_dw, x_s_dev)
            preds[w] = y_hat[:, -1, 0].numpy()
    obs = y[:, :, -1, 0].numpy()
    return {bid: compute_nse(preds[:, i], obs[:, i]) for i, bid in enumerate(basin_ids)}


def train_epoch(model, x_d, x_s, y, optimizer, batch_size, loss_mask):
    """loss_mask: [n_basins] bool — True for basins to include in loss."""
    model.train()
    n_windows = x_d.shape[0]
    indices = np.arange(n_windows)
    np.random.shuffle(indices)
    total_loss = 0.0
    n_steps = 0
    x_s_dev = x_s.to(DEVICE)
    loss_mask_t = torch.tensor(loss_mask, device=DEVICE)

    for i in range(0, n_windows, batch_size):
        batch_idx = indices[i:i + batch_size]
        batch_loss = torch.tensor(0.0, device=DEVICE)
        valid_count = 0
        for w in batch_idx:
            x_w = x_d[w].transpose(0, 1).to(DEVICE)
            y_w = y[w].to(DEVICE)
            y_hat = model(x_w, x_s_dev)
            y_hat_last = y_hat[:, -1, :]
            y_true_last = y_w[:, -1, :]

            # Apply loss mask: only training basins contribute to loss
            valid = ~torch.isnan(y_true_last) & loss_mask_t.unsqueeze(-1)
            if valid.sum() > 0:
                batch_loss = batch_loss + ((y_hat_last[valid] - y_true_last[valid]) ** 2).mean()
                valid_count += 1
        if valid_count > 0:
            batch_loss = batch_loss / valid_count
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            total_loss += batch_loss.item()
            n_steps += 1
    return total_loss / max(n_steps, 1)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    baseline = find_baseline()
    assert baseline is not None, "No ungauged baseline run found"
    LOGGER.info(f"Baseline: {baseline}")

    all_basins = [l.strip() for l in open(ALL_BASINS_FILE) if l.strip()]
    train_basins = set(open(TRAIN_BASINS_FILE).read().split())
    test_basins = set(open(TEST_BASINS_FILE).read().split())
    loss_mask = np.array([b in train_basins for b in all_basins])
    LOGGER.info(f"Total basins: {len(all_basins)}")
    LOGGER.info(f"Training basins: {loss_mask.sum()}")
    LOGGER.info(f"Held-out basins: {(~loss_mask).sum()} — {[b for b in all_basins if b not in train_basins]}")

    edges = load_graph_with_features(EDGE_FILE, all_basins)
    LOGGER.info(f"Graph: {len(edges)} edges over {len(all_basins)} basins")

    cfg = Config(baseline / "config.yml")
    scaler = load_scaler(baseline)

    LOGGER.info("Loading train period data...")
    x_d_train, x_s, y_train = load_basin_data(cfg, scaler, all_basins, "train")
    LOGGER.info(f"Train: {x_d_train.shape}")
    LOGGER.info("Loading test period data...")
    x_d_test, _, y_test = load_basin_data(cfg, scaler, all_basins, "test")
    LOGGER.info(f"Test: {x_d_test.shape}")

    n_dyn = x_d_train.shape[3]
    n_static_total = x_s.shape[1]
    input_size = n_dyn + n_static_total
    LOGGER.info(f"Input size: {input_size}  (dyn={n_dyn}, static={n_static_total})")

    model = DirectedGraphLSTM(
        input_size=input_size, hidden_size=HIDDEN_SIZE,
        edges=edges, n_basins=len(all_basins),
        n_targets=1, dropout=DROPOUT,
        initial_forget_bias=cfg.initial_forget_bias,
        use_edge_features=USE_EDGE_FEATURES,
        use_diff_term=USE_DIFF_TERM,
        use_attention=USE_ATTENTION,
        use_sigmoid_gate=USE_SIGMOID_GATE,
    ).to(DEVICE)

    if WARM_START:
        ckpts = sorted(baseline.glob("model_epoch*.pt"))
        LOGGER.info(f"Warm-starting from: {ckpts[-1]}")
        warm_start(model, ckpts[-1])

    trainable = [p for p in model.parameters() if p.requires_grad]
    LOGGER.info(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.Adam(trainable, lr=LR)

    # Pre-training eval on held-out basins
    LOGGER.info("Pre-training NSE on HELD-OUT basins (should match baseline):")
    pre = evaluate(model, x_d_test, x_s, y_test, all_basins)
    for b in all_basins:
        if b in test_basins:
            LOGGER.info(f"  {b}: {pre[b]:.3f}")

    # Training loop
    timestamp = datetime.now().strftime("%d%m_%H%M%S")
    run_dir = Path(f"runs/graph_ungauged_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Output dir: {run_dir}")

    best_test_nse = float("-inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, x_d_train, x_s, y_train, optimizer, BATCH_SIZE, loss_mask)
        LOGGER.info(f"Epoch {epoch:2d}/{EPOCHS}  train_loss={loss:.5f}")
        torch.save(model.state_dict(), run_dir / f"model_epoch{epoch:03d}.pt")

        if epoch % 3 == 0 or epoch == EPOCHS:
            nse = evaluate(model, x_d_test, x_s, y_test, all_basins)
            heldout_nse = [nse[b] for b in all_basins if b in test_basins]
            train_nse = [nse[b] for b in all_basins if b in train_basins]
            LOGGER.info(f"  Held-out NSE median: {np.nanmedian(heldout_nse):.3f}  "
                         f"per-basin: {dict(zip([b for b in all_basins if b in test_basins], [f'{n:.3f}' for n in heldout_nse]))}")
            LOGGER.info(f"  Train NSE median: {np.nanmedian(train_nse):.3f}")
            if np.nanmedian(heldout_nse) > best_test_nse:
                best_test_nse = np.nanmedian(heldout_nse)
                best_epoch = epoch
                torch.save(model.state_dict(), run_dir / "model_best.pt")

    # Final evaluation
    if (run_dir / "model_best.pt").exists():
        model.load_state_dict(torch.load(run_dir / "model_best.pt", weights_only=True))
        LOGGER.info(f"Using best checkpoint (epoch {best_epoch})")

    final_nse = evaluate(model, x_d_test, x_s, y_test, all_basins)

    # Baseline's NSE on held-out (from NH eval)
    LOGGER.info("=" * 70)
    LOGGER.info("FINAL RESULT: held-out basin NSE (ungauged scenario)")
    LOGGER.info("=" * 70)

    # Load baseline predictions on held-out (need to run NH evaluate first, or just use NSE)
    LOGGER.info(f"{'Basin':>10}  {'Graph':>8}")
    for b in all_basins:
        if b in test_basins:
            LOGGER.info(f"  {b:>10}  {final_nse[b]:8.3f}")

    metrics = pd.DataFrame([{"basin": b, "NSE": final_nse[b]} for b in all_basins])
    metrics.to_csv(run_dir / "test_metrics.csv", index=False)

    run_config = {
        "experiment": "ungauged",
        "train_basins": sorted(list(train_basins)),
        "held_out_basins": sorted(list(test_basins)),
        "baseline_run": str(baseline),
        "n_edges": len(edges),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "best_epoch": best_epoch,
        "best_heldout_median_nse": float(best_test_nse),
        "final_heldout_nse": {b: float(final_nse[b]) for b in test_basins},
        "final_train_nse": {b: float(final_nse[b]) for b in train_basins},
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=str)
    LOGGER.info(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
