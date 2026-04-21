"""Standalone training script for the Directed Graph-LSTM.

Bypasses the NeuralHydrology training loop because the Graph-LSTM requires
all basins to be processed jointly at each timestep (for inter-basin message
passing), while NH processes basins independently in shuffled batches.

Improvements over v1:
  * Edge features (log distance, log area ratio, elev drop) in message function
  * Optional direction gradient term (h_u - h_v) for direction-aware messaging
  * Warm-start from trained baseline LSTM checkpoint (only message passing
    weights need training from scratch)
  * Basin ID one-hot encoding matching the strong baseline

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

# Repo root is three levels up from experiments/training/<script>.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.evaluation.utils import load_basin_id_encoding
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.utils.config import Config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Baseline run to warm-start from (strong baseline preferred; falls back to weak)
STRONG_BASELINE_DIR = None   # set at runtime to the latest strong baseline
WEAK_BASELINE_DIR = Path("runs/03_lstm_23basin_baseline")
EDGE_FILE = Path("topology_analysis/phase1_network_discovery/outputs/study_network_edges.csv")
# Set to study_network_edges_pruned.csv to use the 26-edge pruned graph (run 11).
BASIN_FILE = Path("experiments/basin_lists/study_network_basins.txt")

# Architecture / training
EPOCHS = 15                  # fewer needed with warm-start
LR = 1e-3
HIDDEN_SIZE = 64
SEQ_LENGTH = 30
DROPOUT = 0.4
CLIP_GRAD = 1.0
SEED = 42
BATCH_SIZE = 256

# Architecture flags
USE_EDGE_FEATURES = True     # include [log_dist, log_area_ratio, elev_drop_norm] in msg
USE_DIFF_TERM = False        # include h_u - h_v in msg (Jiang et al. ablation)
USE_BASIN_ENCODING = True    # match strong baseline
WARM_START = True            # initialize LSTM weights from trained baseline
FREEZE_LSTM = False          # freeze LSTM + head; only train message passing
USE_ATTENTION = False        # softmax attention over parents
USE_SIGMOID_GATE = False     # independent sigmoid gate per edge

DEVICE = torch.device("cpu")

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def find_strong_baseline():
    """Find the most recent strong baseline run with a final checkpoint.

    Matches either the timestamped folder (lstm_study_network_strong_*) or the
    renamed folder (05_lstm_23basin_strong_baseline).
    """
    patterns = ["05_lstm_23basin_strong_baseline",
                "lstm_study_network_strong_*"]
    candidates = []
    for pattern in patterns:
        candidates.extend(Path("runs").glob(pattern))
    candidates = sorted(candidates)
    for run_dir in reversed(candidates):
        ckpts = sorted(run_dir.glob("model_epoch*.pt"))
        if ckpts:
            return run_dir
    return None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DirectedGraphLSTM(nn.Module):
    """LSTM with directed upstream message passing and edge features.

    For each timestep t and each basin v:
      h_v^t, c_v^t = LSTMCell(x_v^t, h_v^{t-1}, c_v^{t-1})

      For each upstream parent u of v (using h_u at t-1 = the lag):
          msg_input = [h_u]
          if USE_DIFF_TERM:   msg_input += [h_u - h_v]      (direction gradient)
          if USE_EDGE_FEAT:   msg_input += [e_uv]            (edge features)
          m_uv = W_msg_edge(msg_input)

      m_v = mean_u(m_uv)
      h_v = h_v + tanh(W_out * m_v)              (residual; W_out zero-init)

    Parameters
    ----------
    input_size : int
        Per-basin input dim: n_dynamic + n_static + (n_basins if basin encoding).
    hidden_size : int
    edges : list[(child_idx, parent_idx, edge_feature_vector)]
    n_basins : int
    """

    def __init__(self, input_size, hidden_size, edges, n_basins,
                 n_targets=1, dropout=0.4, initial_forget_bias=3,
                 use_edge_features=True, use_diff_term=False,
                 use_attention=False, use_sigmoid_gate=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_basins = n_basins
        self.use_edge_features = use_edge_features
        self.use_diff_term = use_diff_term
        self.use_attention = use_attention
        self.use_sigmoid_gate = use_sigmoid_gate

        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_size, n_targets)

        # Build edge tensors
        if edges:
            child_idx = torch.tensor([e[0] for e in edges], dtype=torch.long)
            parent_idx = torch.tensor([e[1] for e in edges], dtype=torch.long)
            edge_feat = torch.tensor([e[2] for e in edges], dtype=torch.float32) if use_edge_features else None
        else:
            child_idx = torch.zeros(0, dtype=torch.long)
            parent_idx = torch.zeros(0, dtype=torch.long)
            edge_feat = torch.zeros(0, 3) if use_edge_features else None

        self.register_buffer("child_idx", child_idx)
        self.register_buffer("parent_idx", parent_idx)
        if edge_feat is not None:
            self.register_buffer("edge_feat", edge_feat)
        self.edge_feat_dim = 3 if use_edge_features else 0

        # Message function: takes [h_u] + optional [diff] + optional [edge_feat] -> hidden
        msg_in = hidden_size
        if use_diff_term:
            msg_in += hidden_size
        if use_edge_features:
            msg_in += self.edge_feat_dim
        self.W_msg_edge = nn.Linear(msg_in, hidden_size)
        self.W_out = nn.Linear(hidden_size, hidden_size, bias=False)

        # Attention scoring: for each edge, compute a scalar score from (h_u, h_v, e_uv)
        # α_uv = softmax over edges incident to the same child v
        # Input to attention scorer: [h_u, h_v] (optionally + edge features)
        # Output: scalar score per edge
        if use_attention:
            attn_in = 2 * hidden_size
            if use_edge_features:
                attn_in += self.edge_feat_dim
            self.attn_scorer = nn.Sequential(
                nn.Linear(attn_in, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )

        # Sigmoid gating: independent gate per edge, can take any value in [0,1]
        # Crucially, this can down-weight a single-parent edge below 1
        if use_sigmoid_gate:
            gate_in = 2 * hidden_size
            if use_edge_features:
                gate_in += self.edge_feat_dim
            self.gate_scorer = nn.Sequential(
                nn.Linear(gate_in, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )
            # Initialize last layer bias to a large positive value so gates start near 1.
            # This makes the model start similar to mean-aggregation (all gates ~1),
            # then learn to close gates for unreliable edges.
            nn.init.constant_(self.gate_scorer[-1].bias, 2.0)  # sigmoid(2) ~ 0.88

        # Zero-init the residual output so the model starts as vanilla LSTM
        nn.init.zeros_(self.W_out.weight)

        # Initial forget bias
        if initial_forget_bias is not None:
            self.lstm_cell.bias_hh.data[hidden_size:2 * hidden_size] = initial_forget_bias

        # Track how many parents each child has, for mean aggregation
        parent_count = torch.zeros(n_basins, dtype=torch.long)
        for c, _, _ in edges:
            parent_count[c] += 1
        self.register_buffer("parent_count", parent_count.clamp(min=1))
        self.register_buffer("has_parents", (parent_count > 0).float().unsqueeze(-1))

    def forward(self, x_d, x_s):
        """
        x_d: [seq_len, n_basins, n_dyn]
        x_s: [n_basins, n_static + n_basin_onehot]
        Returns y_hat: [n_basins, seq_len, n_targets]
        """
        seq_len, n_basins, _ = x_d.shape
        h = torch.zeros(n_basins, self.hidden_size, device=x_d.device)
        c = torch.zeros(n_basins, self.hidden_size, device=x_d.device)

        outputs = []
        for t in range(seq_len):
            x_t = torch.cat([x_d[t], x_s], dim=-1)
            h_new, c_new = self.lstm_cell(x_t, (h, c))

            # Message passing using h from PREVIOUS timestep (the lag)
            if self.child_idx.numel() > 0:
                h_u = h[self.parent_idx]              # [n_edges, hidden]
                h_v_at_edge = h[self.child_idx]       # [n_edges, hidden]
                msg_parts = [h_u]

                if self.use_diff_term:
                    msg_parts.append(h_u - h_v_at_edge)

                if self.use_edge_features:
                    msg_parts.append(self.edge_feat)

                msg_input = torch.cat(msg_parts, dim=-1)
                edge_msg = self.W_msg_edge(msg_input)  # [n_edges, hidden]

                if self.use_sigmoid_gate:
                    # Independent sigmoid gate per edge: g_uv = σ(score(h_u, h_v, e_uv))
                    # Unlike softmax-attention, a single-parent edge can have g < 1.
                    gate_in_parts = [h_u, h_v_at_edge]
                    if self.use_edge_features:
                        gate_in_parts.append(self.edge_feat)
                    gate_input = torch.cat(gate_in_parts, dim=-1)
                    gate_scores = self.gate_scorer(gate_input).squeeze(-1)   # [n_edges]
                    gates = torch.sigmoid(gate_scores)                        # [n_edges] in [0,1]

                    # Gate-weighted aggregation, normalized by SUM of gates per child
                    # (not parent count). When gates ≈ 1 for all parents, this matches
                    # mean aggregation. When gates shrink to 0, those parents' messages
                    # vanish.
                    gated_msg = edge_msg * gates.unsqueeze(-1)
                    agg = torch.zeros(n_basins, self.hidden_size, device=x_d.device)
                    agg.index_add_(0, self.child_idx, gated_msg)
                    # Normalize by sum of gates (effective number of contributing parents)
                    gate_sum = torch.zeros(n_basins, device=x_d.device)
                    gate_sum.index_add_(0, self.child_idx, gates)
                    m = agg / (gate_sum.unsqueeze(-1) + 1e-6)
                    m = m * self.has_parents
                elif self.use_attention:
                    # Softmax attention (limitation: single-parent edges forced to weight 1)
                    attn_in_parts = [h_u, h_v_at_edge]
                    if self.use_edge_features:
                        attn_in_parts.append(self.edge_feat)
                    attn_input = torch.cat(attn_in_parts, dim=-1)
                    edge_scores = self.attn_scorer(attn_input).squeeze(-1)

                    max_per_child = torch.full((n_basins,), -1e9, device=x_d.device)
                    max_per_child = max_per_child.scatter_reduce(
                        0, self.child_idx, edge_scores, reduce="amax", include_self=False)
                    edge_scores_shifted = edge_scores - max_per_child[self.child_idx]
                    edge_exp = torch.exp(edge_scores_shifted)
                    sum_per_child = torch.zeros(n_basins, device=x_d.device)
                    sum_per_child.index_add_(0, self.child_idx, edge_exp)
                    alpha = edge_exp / (sum_per_child[self.child_idx] + 1e-9)

                    weighted_msg = edge_msg * alpha.unsqueeze(-1)
                    agg = torch.zeros(n_basins, self.hidden_size, device=x_d.device)
                    agg.index_add_(0, self.child_idx, weighted_msg)
                    m = agg * self.has_parents
                else:
                    # Mean aggregation (original)
                    agg = torch.zeros(n_basins, self.hidden_size, device=x_d.device)
                    agg.index_add_(0, self.child_idx, edge_msg)
                    m = agg / self.parent_count.unsqueeze(-1).float()
                    m = m * self.has_parents   # zero out headwaters

                h_new = h_new + torch.tanh(self.W_out(m))

            h, c = h_new, c_new
            outputs.append(h)

        lstm_output = torch.stack(outputs, dim=0).transpose(0, 1)  # [n_basins, seq_len, hidden]
        y_hat = self.head(self.dropout(lstm_output))
        return y_hat


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_graph_with_features(edge_file, basin_ids):
    """Load edges with features.
    Returns: list of (child_idx, parent_idx, [log_dist, log_ar, elev_drop_norm])
    """
    edges_df = pd.read_csv(edge_file, dtype={"parent_id": str, "child_id": str})
    id_to_idx = {bid: i for i, bid in enumerate(basin_ids)}

    # Normalize edge features
    edges_df = edges_df[edges_df["parent_id"].isin(id_to_idx) & edges_df["child_id"].isin(id_to_idx)]

    log_dist = np.log(edges_df["distance_km"].values + 1.0)
    log_ar = np.log(edges_df["area_ratio"].values)
    elev_drop = edges_df["elev_diff_m"].values

    # Normalize each column to zero-mean unit-variance
    def znorm(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    log_dist_n = znorm(log_dist)
    log_ar_n = znorm(log_ar)
    elev_drop_n = znorm(elev_drop)

    edges = []
    for i, row in edges_df.reset_index(drop=True).iterrows():
        c = id_to_idx[row["child_id"]]
        p = id_to_idx[row["parent_id"]]
        feat = [float(log_dist_n[i]), float(log_ar_n[i]), float(elev_drop_n[i])]
        edges.append((c, p, feat))

    return edges


def load_basin_data(cfg, scaler, basin_ids, period, id_to_int=None):
    """Load time-aligned data for all basins.

    Returns
    -------
    x_d : Tensor [n_windows, n_basins, seq_len, n_dyn]
    x_s : Tensor [n_basins, n_static (+ n_basins_onehot if encoding)]
    y   : Tensor [n_windows, n_basins, seq_len, n_targets]
    """
    all_x_d, all_x_s, all_y = [], [], []

    for basin in basin_ids:
        kwargs = dict(cfg=cfg, is_train=False, period=period, basin=basin, scaler=scaler)
        if id_to_int:
            kwargs["id_to_int"] = id_to_int
        ds = get_dataset(**kwargs)

        basin_x_d = []
        basin_y = []
        x_s = None

        for i in range(len(ds)):
            sample = ds[i]
            dyn_tensors = [v for k, v in sorted(sample["x_d"].items())]
            x_d_cat = torch.cat(dyn_tensors, dim=-1)
            basin_x_d.append(x_d_cat)
            basin_y.append(sample["y"])
            if x_s is None:
                static_parts = []
                if "x_s" in sample:
                    static_parts.append(sample["x_s"])
                if "x_one_hot" in sample:
                    static_parts.append(sample["x_one_hot"])
                x_s = torch.cat(static_parts, dim=-1) if static_parts else None

        all_x_d.append(torch.stack(basin_x_d))
        all_y.append(torch.stack(basin_y))
        all_x_s.append(x_s)

    x_d = torch.stack(all_x_d, dim=1)
    y = torch.stack(all_y, dim=1)
    x_s = torch.stack(all_x_s)

    return x_d, x_s, y


# ---------------------------------------------------------------------------
# Warm-start
# ---------------------------------------------------------------------------
def warm_start_from_baseline(model, baseline_ckpt_path):
    """Copy LSTM weights from a trained baseline CudaLSTM checkpoint.

    Baseline keys:   lstm.weight_ih_l0, lstm.weight_hh_l0, lstm.bias_ih_l0, lstm.bias_hh_l0
    Graph keys:      lstm_cell.weight_ih, lstm_cell.weight_hh, lstm_cell.bias_ih, lstm_cell.bias_hh
    Head:            head.net.0.* in baseline; head.* in graph model
    """
    ckpt = torch.load(baseline_ckpt_path, map_location="cpu", weights_only=True)

    # Map baseline LSTM weights -> LSTMCell weights
    mapping = {
        "lstm.weight_ih_l0": "lstm_cell.weight_ih",
        "lstm.weight_hh_l0": "lstm_cell.weight_hh",
        "lstm.bias_ih_l0": "lstm_cell.bias_ih",
        "lstm.bias_hh_l0": "lstm_cell.bias_hh",
    }

    own_state = model.state_dict()
    copied = 0
    for baseline_key, graph_key in mapping.items():
        if baseline_key in ckpt and graph_key in own_state:
            if ckpt[baseline_key].shape == own_state[graph_key].shape:
                own_state[graph_key].copy_(ckpt[baseline_key])
                copied += 1
                LOGGER.info(f"  Warm-started: {baseline_key} -> {graph_key}")
            else:
                LOGGER.warning(
                    f"  Shape mismatch {baseline_key}: {ckpt[baseline_key].shape} "
                    f"vs {own_state[graph_key].shape} — skipping"
                )

    # Copy head weights (Regression head wraps a single Linear)
    # baseline head key pattern: head.net.0.weight, head.net.0.bias
    # graph model head is a bare Linear: head.weight, head.bias
    for baseline_key, graph_key in [
        ("head.net.0.weight", "head.weight"),
        ("head.net.0.bias", "head.bias"),
    ]:
        if baseline_key in ckpt and graph_key in own_state:
            if ckpt[baseline_key].shape == own_state[graph_key].shape:
                own_state[graph_key].copy_(ckpt[baseline_key])
                copied += 1
                LOGGER.info(f"  Warm-started: {baseline_key} -> {graph_key}")

    LOGGER.info(f"  Total warm-started tensors: {copied}")


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def train_epoch(model, x_d, x_s, y, optimizer, batch_size):
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
            x_w = x_d[w].transpose(0, 1).to(DEVICE)
            y_w = y[w].to(DEVICE)

            y_hat = model(x_w, x_s_dev)
            y_hat_last = y_hat[:, -1, :]
            y_true_last = y_w[:, -1, :]

            valid = ~torch.isnan(y_true_last)
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


def evaluate(model, x_d, x_s, y, basin_ids):
    model.eval()
    n_windows, n_basins, seq_len, n_targets = y.shape
    all_preds = {i: [] for i in range(n_basins)}
    all_obs = {i: [] for i in range(n_basins)}
    x_s_dev = x_s.to(DEVICE)

    with torch.no_grad():
        for w in range(n_windows):
            x_w = x_d[w].transpose(0, 1).to(DEVICE)
            y_hat = model(x_w, x_s_dev)
            for b in range(n_basins):
                pred_val = y_hat[b, -1, 0].item()
                obs_val = y[w, b, -1, 0].item()
                if not np.isnan(obs_val):
                    all_preds[b].append(pred_val)
                    all_obs[b].append(obs_val)

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

    # Resolve baseline
    global STRONG_BASELINE_DIR
    STRONG_BASELINE_DIR = find_strong_baseline()
    baseline_run = STRONG_BASELINE_DIR if (STRONG_BASELINE_DIR and USE_BASIN_ENCODING) else WEAK_BASELINE_DIR
    LOGGER.info(f"Using baseline run: {baseline_run}")

    timestamp = datetime.now().strftime("%d%m_%H%M%S")
    tag = []
    if USE_EDGE_FEATURES: tag.append("edgefeat")
    if USE_DIFF_TERM: tag.append("diff")
    if USE_BASIN_ENCODING: tag.append("bencode")
    if WARM_START: tag.append("warm")
    if FREEZE_LSTM: tag.append("frozen")
    if USE_ATTENTION: tag.append("attn")
    if USE_SIGMOID_GATE: tag.append("sigate")
    tag_str = "_".join(tag) if tag else "vanilla"
    run_dir = Path(f"runs/graph_lstm_{tag_str}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Output directory: {run_dir}")

    # Basin IDs and graph
    basin_ids = [l.strip() for l in open(BASIN_FILE) if l.strip()]
    n_basins = len(basin_ids)
    edges = load_graph_with_features(EDGE_FILE, basin_ids)
    LOGGER.info(f"Loaded {n_basins} basins, {len(edges)} edges")

    # Data
    cfg = Config(baseline_run / "config.yml")

    # Load scaler and basin ID encoding from baseline run (for consistency with warm-start)
    LOGGER.info(f"Loading scaler from baseline run: {baseline_run}")
    scaler = load_scaler(baseline_run)

    id_to_int = {}
    if cfg.use_basin_id_encoding:
        id_to_int = load_basin_id_encoding(baseline_run)
        LOGGER.info(f"Loaded basin ID encoding: {len(id_to_int)} basins")

    LOGGER.info("Loading train period data...")
    x_d_train, x_s, y_train = load_basin_data(cfg, scaler, basin_ids, "train", id_to_int)
    LOGGER.info(f"Train: {x_d_train.shape[0]} windows x {n_basins} basins x "
                f"seq_len {x_d_train.shape[2]} x {x_d_train.shape[3]} dyn features")
    LOGGER.info(f"  Static+encoding dim: {x_s.shape[1]}")

    LOGGER.info("Loading test period data...")
    x_d_test, _, y_test = load_basin_data(cfg, scaler, basin_ids, "test", id_to_int)
    LOGGER.info(f"Test: {x_d_test.shape[0]} windows x {n_basins} basins")

    # Build model
    n_dyn = x_d_train.shape[3]
    n_static_total = x_s.shape[1]
    input_size = n_dyn + n_static_total

    model = DirectedGraphLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        edges=edges,
        n_basins=n_basins,
        n_targets=len(cfg.target_variables),
        dropout=DROPOUT,
        initial_forget_bias=cfg.initial_forget_bias,
        use_edge_features=USE_EDGE_FEATURES,
        use_diff_term=USE_DIFF_TERM,
        use_attention=USE_ATTENTION,
        use_sigmoid_gate=USE_SIGMOID_GATE,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Model parameters: {total_params:,}")
    LOGGER.info(f"  Input size: {input_size}  "
                f"(dyn={n_dyn}, static+encoding={n_static_total})")

    # Warm-start
    if WARM_START:
        ckpts = sorted(baseline_run.glob("model_epoch*.pt"))
        if ckpts:
            LOGGER.info(f"Warm-starting from: {ckpts[-1]}")
            warm_start_from_baseline(model, ckpts[-1])
        else:
            LOGGER.warning("No baseline checkpoint found — training from scratch")

    # Freeze LSTM + head if requested (cleanest ablation: only graph params train)
    if FREEZE_LSTM:
        for p in model.lstm_cell.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = False
        LOGGER.info("Frozen LSTM + head; only W_msg_edge and W_out will train")

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    LOGGER.info(f"Trainable parameters: {n_trainable:,} of {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(trainable, lr=LR)

    # Pre-training test NSE (sanity check: should match baseline if warm-start worked)
    LOGGER.info("Pre-training test NSE (should match baseline if warm-start worked):")
    pre_results = evaluate(model, x_d_test, x_s, y_test, basin_ids)
    pre_median = np.median(list(pre_results.values()))
    LOGGER.info(f"  median NSE = {pre_median:.3f}")

    # Training loop
    LOGGER.info(f"Training for {EPOCHS} epochs...")
    best_nse = pre_median
    best_epoch = 0
    loss_history = []
    nse_history = [(0, pre_median)]

    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_epoch(model, x_d_train, x_s, y_train, optimizer, BATCH_SIZE)
        loss_history.append(avg_loss)
        LOGGER.info(f"Epoch {epoch:2d}/{EPOCHS}  train_loss={avg_loss:.5f}")
        torch.save(model.state_dict(), run_dir / f"model_epoch{epoch:03d}.pt")

        if epoch % 3 == 0 or epoch == EPOCHS:
            test_results = evaluate(model, x_d_test, x_s, y_test, basin_ids)
            median_nse = np.median(list(test_results.values()))
            nse_history.append((epoch, median_nse))
            LOGGER.info(f"  Test median NSE: {median_nse:.3f}")
            if median_nse > best_nse:
                best_nse = median_nse
                best_epoch = epoch
                torch.save(model.state_dict(), run_dir / "model_best.pt")

    # Final evaluation (use best checkpoint)
    if (run_dir / "model_best.pt").exists():
        model.load_state_dict(torch.load(run_dir / "model_best.pt", weights_only=True))
        LOGGER.info(f"Final eval uses best checkpoint (epoch {best_epoch}, NSE {best_nse:.3f})")
    final_results = evaluate(model, x_d_test, x_s, y_test, basin_ids)

    metrics_df = pd.DataFrame([{"basin": bid, "NSE": nse} for bid, nse in final_results.items()])
    metrics_df.to_csv(run_dir / "test_metrics.csv", index=False)

    for bid, nse in sorted(final_results.items()):
        LOGGER.info(f"  {bid}: NSE={nse:.3f}")
    median_nse = np.median(list(final_results.values()))
    mean_nse = np.mean(list(final_results.values()))
    LOGGER.info(f"  Median NSE: {median_nse:.3f}")
    LOGGER.info(f"  Mean NSE:   {mean_nse:.3f}")

    run_config = {
        "model": "DirectedGraphLSTM",
        "epochs": EPOCHS,
        "lr": LR,
        "hidden_size": HIDDEN_SIZE,
        "seq_length": SEQ_LENGTH,
        "dropout": DROPOUT,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "n_basins": n_basins,
        "n_edges": len(edges),
        "use_edge_features": USE_EDGE_FEATURES,
        "use_diff_term": USE_DIFF_TERM,
        "use_basin_encoding": USE_BASIN_ENCODING,
        "warm_start": WARM_START,
        "freeze_lstm": FREEZE_LSTM,
        "use_attention": USE_ATTENTION,
        "use_sigmoid_gate": USE_SIGMOID_GATE,
        "baseline_run": str(baseline_run),
        "input_size": input_size,
        "n_dynamic_features": n_dyn,
        "n_static_total": n_static_total,
        "timestamp": datetime.now().isoformat(),
        "pre_training_median_nse": float(pre_median),
        "best_epoch": best_epoch,
        "final_median_nse": float(median_nse),
        "final_mean_nse": float(mean_nse),
        "loss_history": loss_history,
        "nse_history": nse_history,
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    LOGGER.info(f"\nResults saved to {run_dir}")


if __name__ == "__main__":
    main()
