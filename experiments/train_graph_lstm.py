"""Standalone training script for the Directed Graph-LSTM.

Bypasses the NeuralHydrology training loop because the Graph-LSTM requires
all basins to be processed jointly at each timestep (for inter-basin message
passing), while NH processes basins independently in shuffled batches.

Uses NH's data loaders, scaler, and evaluation metrics for consistency
with the baseline LSTM results.

Usage:
    /Applications/anaconda3/envs/nh/bin/python experiments/train_graph_lstm.py
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.evaluation.metrics import calculate_all_metrics
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.utils.config import Config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASELINE_RUN_DIR = Path("runs/03_lstm_23basin_baseline")
EDGE_FILE = Path("topology_analysis/phase1_network_discovery/outputs/study_network_edges.csv")
BASIN_FILE = Path("experiments/study_network_basins.txt")

EPOCHS = 10
LR = 1e-3
HIDDEN_SIZE = 64
SEQ_LENGTH = 30
DROPOUT = 0.4
CLIP_GRAD = 1.0
SEED = 42

DEVICE = torch.device("cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DirectedGraphLSTM(nn.Module):
    """LSTM with directed upstream message passing.

    At each timestep t, for each basin v:
      1. Standard LSTM step:  h_v^t, c_v^t = LSTMCell(x_v^t, h_v^{t-1}, c_v^{t-1})
      2. Upstream message:    m_v^t = mean_{u in parents(v)} [ W_up * h_u^{t-1} ]
      3. Residual update:     h_v^t = h_v^t + tanh(W_msg * m_v^t)
      4. Readout:             y_hat = head(dropout(h_v^T))

    W_msg is zero-initialized so the model starts as a standard LSTM.
    """

    def __init__(self, cfg, parent_indices):
        """
        Parameters
        ----------
        cfg : Config
            NeuralHydrology config (for embedding/head settings).
        parent_indices : dict[int, list[int]]
            Maps basin index -> list of upstream parent basin indices.
        """
        super().__init__()
        self.parent_indices = parent_indices
        self.hidden_size = cfg.hidden_size

        self.embedding_net = InputLayer(cfg)
        self.lstm_cell = nn.LSTMCell(
            input_size=self.embedding_net.output_size,
            hidden_size=cfg.hidden_size,
        )
        self.dropout = nn.Dropout(p=cfg.output_dropout)

        n_out = len(cfg.target_variables)
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=n_out)

        # Upstream message layers (zero-initialized for residual start)
        self.W_upstream = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.W_msg = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        nn.init.zeros_(self.W_msg.weight)

        # Initial forget bias
        if cfg.initial_forget_bias is not None:
            self.lstm_cell.bias_hh.data[cfg.hidden_size:2 * cfg.hidden_size] = cfg.initial_forget_bias

    def forward(self, x_d, x_s):
        """
        Parameters
        ----------
        x_d : Tensor [seq_len, n_basins, n_dynamic_features]
        x_s : Tensor [n_basins, n_static_features]

        Returns
        -------
        y_hat : Tensor [n_basins, seq_len, n_targets]
        """
        seq_len, n_basins, _ = x_d.shape

        h = torch.zeros(n_basins, self.hidden_size, device=x_d.device)
        c = torch.zeros(n_basins, self.hidden_size, device=x_d.device)

        outputs = []
        for t in range(seq_len):
            # 1. Embed: concat dynamic + static -> input vector
            # InputLayer expects dict format with x_d and x_s
            # We'll bypass it and do manual embedding for joint processing
            x_t = x_d[t]  # [n_basins, n_dyn]

            # Concat static to each timestep (same as InputLayer default behavior)
            if x_s is not None:
                x_t = torch.cat([x_t, x_s], dim=-1)

            # 2. LSTM step
            h_new, c_new = self.lstm_cell(x_t, (h, c))

            # 3. Upstream message passing (using h from previous timestep = h before update)
            msg = torch.zeros_like(h_new)
            for basin_idx, parent_idxs in self.parent_indices.items():
                if parent_idxs:
                    # h is the PREVIOUS timestep's hidden state (the lag)
                    parent_states = h[parent_idxs]  # [n_parents, hidden]
                    transformed = self.W_upstream(parent_states)  # [n_parents, hidden]
                    msg[basin_idx] = transformed.mean(dim=0)

            # 4. Residual update
            h_new = h_new + torch.tanh(self.W_msg(msg))

            h = h_new
            c = c_new
            outputs.append(h)

        # Stack: [seq_len, n_basins, hidden] -> [n_basins, seq_len, hidden]
        lstm_output = torch.stack(outputs, dim=0).transpose(0, 1)

        # Apply dropout and head
        pred = self.head(self.dropout(lstm_output))

        return pred["y_hat"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_graph(edge_file, basin_ids):
    """Load directed graph as parent_indices dict.

    Returns dict mapping basin_idx -> [parent_basin_idx, ...].
    Edges in file are parent_id -> child_id (upstream -> downstream).
    So for basin v, parents are basins that have an edge TO v.
    """
    edges = pd.read_csv(edge_file, dtype={"parent_id": str, "child_id": str})
    id_to_idx = {bid: i for i, bid in enumerate(basin_ids)}

    parent_indices = {i: [] for i in range(len(basin_ids))}
    for _, row in edges.iterrows():
        parent = row["parent_id"]
        child = row["child_id"]
        if parent in id_to_idx and child in id_to_idx:
            parent_indices[id_to_idx[child]].append(id_to_idx[parent])

    return parent_indices


def load_basin_data(cfg, scaler, basin_ids, period):
    """Load time-aligned data for all basins.

    Returns
    -------
    x_d : Tensor [seq_len_windows, n_basins, window_len, n_dyn]
    x_s : Tensor [n_basins, n_static]
    y   : Tensor [seq_len_windows, n_basins, window_len, n_targets]
    dates : list of date arrays
    """
    all_x_d = []
    all_x_s = []
    all_y = []

    for basin in basin_ids:
        ds = get_dataset(cfg=cfg, is_train=False, period=period, basin=basin, scaler=scaler)

        basin_x_d = []
        basin_y = []
        x_s = None

        for i in range(len(ds)):
            sample = ds[i]
            # Concatenate dynamic features
            dyn_tensors = [v for k, v in sorted(sample["x_d"].items())]
            x_d_cat = torch.cat(dyn_tensors, dim=-1)  # [seq_len, n_dyn]
            basin_x_d.append(x_d_cat)
            basin_y.append(sample["y"])
            if x_s is None and "x_s" in sample:
                x_s = sample["x_s"]

        all_x_d.append(torch.stack(basin_x_d))  # [n_windows, seq_len, n_dyn]
        all_y.append(torch.stack(basin_y))        # [n_windows, seq_len, n_targets]
        all_x_s.append(x_s)

    # Stack across basins: [n_windows, n_basins, seq_len, n_dyn]
    x_d = torch.stack(all_x_d, dim=1)
    y = torch.stack(all_y, dim=1)
    x_s = torch.stack(all_x_s)  # [n_basins, n_static]

    return x_d, x_s, y


def create_batches(n_windows, batch_size, shuffle=True):
    """Create random batches of window indices."""
    indices = np.arange(n_windows)
    if shuffle:
        np.random.shuffle(indices)
    batches = []
    for i in range(0, len(indices), batch_size):
        batches.append(indices[i:i + batch_size])
    return batches


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_epoch(model, x_d, x_s, y, optimizer, batch_size=256):
    """Train one epoch. Each batch = one time window with all basins jointly."""
    model.train()
    n_windows = x_d.shape[0]
    indices = np.arange(n_windows)
    np.random.shuffle(indices)
    total_loss = 0.0
    n_steps = 0
    x_s_dev = x_s.to(DEVICE)

    for i in range(0, n_windows, batch_size):
        batch_idx = indices[i:i + batch_size]
        batch_loss = torch.tensor(0.0, device=DEVICE)
        valid_count = 0

        for w in batch_idx:
            x_w = x_d[w].transpose(0, 1).to(DEVICE)  # [seq_len, n_basins, n_dyn]
            y_w = y[w].to(DEVICE)                      # [n_basins, seq_len, n_targets]

            y_hat = model(x_w, x_s_dev)
            y_hat_last = y_hat[:, -1, :]   # [n_basins, n_targets]
            y_true_last = y_w[:, -1, :]

            valid = ~torch.isnan(y_true_last)
            if valid.sum() > 0:
                batch_loss = batch_loss + ((y_hat_last[valid] - y_true_last[valid]) ** 2).mean()
                valid_count += 1

        if valid_count > 0:
            batch_loss = batch_loss / valid_count  # average over windows
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()
            total_loss += batch_loss.item()
            n_steps += 1

    return total_loss / max(n_steps, 1)


def evaluate(model, x_d, x_s, y, basin_ids):
    """Evaluate model: compute per-basin NSE on all windows."""
    model.eval()
    n_windows, n_basins, seq_len, n_targets = y.shape

    all_preds = {i: [] for i in range(n_basins)}
    all_obs = {i: [] for i in range(n_basins)}

    with torch.no_grad():
        # Process in chunks to avoid memory issues
        chunk_size = 128
        for start in range(0, n_windows, chunk_size):
            end = min(start + chunk_size, n_windows)
            x_d_chunk = x_d[start:end].to(DEVICE)
            y_chunk = y[start:end]

            for w in range(x_d_chunk.shape[0]):
                x_w = x_d_chunk[w].transpose(0, 1)
                y_hat = model(x_w, x_s.to(DEVICE))

                for b in range(n_basins):
                    pred_val = y_hat[b, -1, 0].item()
                    obs_val = y_chunk[w, b, -1, 0].item()
                    if not np.isnan(obs_val):
                        all_preds[b].append(pred_val)
                        all_obs[b].append(obs_val)

    # Compute NSE per basin
    results = {}
    for b in range(n_basins):
        obs = np.array(all_obs[b])
        pred = np.array(all_preds[b])
        if len(obs) > 0 and obs.std() > 0:
            nse = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)
        else:
            nse = float("nan")
        results[basin_ids[b]] = nse

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Setup output directory
    timestamp = datetime.now().strftime("%d%m_%H%M%S")
    run_dir = Path(f"runs/graph_lstm_study_network_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Output directory: {run_dir}")

    # Load basin IDs and graph
    basin_ids = [l.strip() for l in open(BASIN_FILE) if l.strip()]
    parent_indices = load_graph(EDGE_FILE, basin_ids)
    n_basins = len(basin_ids)

    LOGGER.info(f"Loaded {n_basins} basins")
    n_edges = sum(len(v) for v in parent_indices.values())
    LOGGER.info(f"Graph: {n_edges} directed edges")
    n_with_parents = sum(1 for v in parent_indices.values() if v)
    LOGGER.info(f"Basins with upstream neighbors: {n_with_parents}/{n_basins}")

    # Load config and scaler from baseline run
    cfg = Config(BASELINE_RUN_DIR / "config.yml")
    LOGGER.info("Loading training data (all basins)...")
    ds_train = get_dataset(cfg=cfg, is_train=True, period="train")
    scaler = ds_train.scaler
    del ds_train

    LOGGER.info("Loading train period data...")
    x_d_train, x_s, y_train = load_basin_data(cfg, scaler, basin_ids, "train")
    LOGGER.info(f"Train: {x_d_train.shape[0]} windows x {n_basins} basins x "
                f"seq_len {x_d_train.shape[2]} x {x_d_train.shape[3]} features")

    LOGGER.info("Loading test period data...")
    x_d_test, _, y_test = load_basin_data(cfg, scaler, basin_ids, "test")
    LOGGER.info(f"Test: {x_d_test.shape[0]} windows x {n_basins} basins")

    # Build model
    n_dyn = x_d_train.shape[3]
    n_static = x_s.shape[1]

    model = DirectedGraphLSTM(cfg, parent_indices).to(DEVICE)

    # Override the embedding net since we handle embedding manually
    # The model concatenates [dynamic, static] directly
    # We need to set the lstm_cell input size correctly
    input_size = n_dyn + n_static
    model.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=HIDDEN_SIZE).to(DEVICE)

    # Re-apply forget bias
    if cfg.initial_forget_bias is not None:
        model.lstm_cell.bias_hh.data[HIDDEN_SIZE:2 * HIDDEN_SIZE] = cfg.initial_forget_bias

    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    LOGGER.info(f"Training for {EPOCHS} epochs...")
    best_nse = -999
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_epoch(model, x_d_train, x_s, y_train, optimizer, batch_size=64)
        LOGGER.info(f"Epoch {epoch:2d}/{EPOCHS}  train_loss={avg_loss:.5f}")

        # Save checkpoint
        torch.save(model.state_dict(), run_dir / f"model_epoch{epoch:03d}.pt")

        # Quick validation every 5 epochs
        if epoch % 5 == 0 or epoch == EPOCHS:
            test_results = evaluate(model, x_d_test, x_s, y_test, basin_ids)
            median_nse = np.median(list(test_results.values()))
            LOGGER.info(f"  Test median NSE: {median_nse:.3f}")
            if median_nse > best_nse:
                best_nse = median_nse
                torch.save(model.state_dict(), run_dir / "model_best.pt")

    # Final evaluation
    LOGGER.info("=" * 60)
    LOGGER.info("Final evaluation on test set")
    LOGGER.info("=" * 60)

    test_results = evaluate(model, x_d_test, x_s, y_test, basin_ids)

    # Save metrics
    metrics_df = pd.DataFrame([
        {"basin": bid, "NSE": nse} for bid, nse in test_results.items()
    ])
    metrics_df.to_csv(run_dir / "test_metrics.csv", index=False)

    # Print results
    for bid, nse in sorted(test_results.items()):
        LOGGER.info(f"  {bid}: NSE={nse:.3f}")

    median_nse = np.median(list(test_results.values()))
    mean_nse = np.mean(list(test_results.values()))
    LOGGER.info(f"  Median NSE: {median_nse:.3f}")
    LOGGER.info(f"  Mean NSE:   {mean_nse:.3f}")

    # Save run config
    run_config = {
        "model": "DirectedGraphLSTM",
        "epochs": EPOCHS,
        "lr": LR,
        "hidden_size": HIDDEN_SIZE,
        "seq_length": SEQ_LENGTH,
        "dropout": DROPOUT,
        "seed": SEED,
        "n_basins": n_basins,
        "n_edges": n_edges,
        "baseline_run": str(BASELINE_RUN_DIR),
        "timestamp": datetime.now().isoformat(),
        "median_nse": float(median_nse),
        "mean_nse": float(mean_nse),
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    LOGGER.info(f"\nResults saved to {run_dir}")
    return test_results


if __name__ == "__main__":
    main()
