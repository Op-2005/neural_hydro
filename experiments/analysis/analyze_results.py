"""Deep analysis of Graph-LSTM results: hydrographs, delta-vs-topology, learned weights.

Goal is to extract research insights, not just NSE numbers:
  1. Hydrograph plots for key basins (where baseline and graph predictions diverge)
  2. Delta NSE vs basin properties (depth, n_upstream, area, elevation) to test hypotheses
  3. Learned W_msg_edge and W_out weight inspection
  4. Flow-regime split (high-flow vs low-flow NSE)

Usage:
    /Applications/anaconda3/envs/nh/bin/python experiments/analyze_results.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Repo root is three levels up from experiments/analysis/<script>.py.
# Second insert lets us import train_graph_lstm from experiments/training/.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.evaluation.utils import load_basin_id_encoding
from neuralhydrology.utils.config import Config

from train_graph_lstm import (
    DirectedGraphLSTM,
    load_basin_data,
    load_graph_with_features,
)

ROOT = Path(__file__).parent.parent.parent
STRONG_BASELINE = ROOT / "runs" / "05_lstm_23basin_strong_baseline"
WEAK_BASELINE = ROOT / "runs" / "03_lstm_23basin_baseline"

# Graph runs to analyze. Each entry specifies (run_dir, label)
# The analyzer auto-detects which baseline each was trained against via run_config.json
# Use renamed numbered runs (post-session reorganization)
GRAPH_RUN = ROOT / "runs" / "06_graph_edge_warm_full"
GRAPH_FROZEN_RUN = ROOT / "runs" / "07_graph_edge_frozen"

EDGE_FILE = ROOT / "topology_analysis/phase1_network_discovery/outputs/study_network_edges.csv"
BASIN_FILE = ROOT / "experiments/basin_lists/study_network_basins.txt"
SUMMARY_FILE = ROOT / "topology_analysis/phase1_network_discovery/outputs/study_network_summary.txt"


def detect_baseline(graph_run_dir):
    """Detect which baseline a graph run was trained against, return Path."""
    cfg_path = graph_run_dir / "run_config.json"
    if not cfg_path.exists():
        return STRONG_BASELINE  # default guess
    cfg = json.load(open(cfg_path))
    baseline_str = cfg.get("baseline_run", "")
    # Normalize to a path
    if "03_lstm_23basin_baseline" in baseline_str:
        return WEAK_BASELINE
    if "05_lstm_23basin_strong_baseline" in baseline_str or "lstm_study_network_strong" in baseline_str:
        return STRONG_BASELINE
    return Path(baseline_str) if baseline_str else STRONG_BASELINE

OUT_DIR = ROOT / "experiments" / "analysis_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Basin metadata
# ---------------------------------------------------------------------------
def load_basin_info():
    edges = pd.read_csv(EDGE_FILE, dtype={"parent_id": str, "child_id": str})
    parents = {}
    for _, row in edges.iterrows():
        parents.setdefault(row["child_id"], []).append(row["parent_id"])

    info = {}
    in_table = False
    for line in open(SUMMARY_FILE):
        stripped = line.strip()
        if stripped.startswith("Basin") and "Area_km2" in stripped:
            in_table = True
            continue
        if stripped.startswith("Edge list:"):
            break
        if not in_table:
            continue
        parts = stripped.split()
        # Basin rows have: basin_id area_km2 elev_m depth role
        if len(parts) == 5 and parts[0].isdigit() and len(parts[0]) == 8:
            try:
                bid = parts[0]
                info[bid] = {
                    "area_km2": float(parts[1]),
                    "elev_m": float(parts[2]),
                    "depth": int(parts[3]),
                    "role": parts[4],
                    "n_upstream": len(parents.get(bid, [])),
                }
            except ValueError:
                continue
    return info


# ---------------------------------------------------------------------------
# Load predictions by running both models
# ---------------------------------------------------------------------------
def load_and_evaluate_graph_model(ckpt_path, run_dir, cfg, scaler, id_to_int,
                                    basin_ids, use_edge_features=True,
                                    use_diff_term=False, use_freeze=True):
    """Load a graph model checkpoint, run inference on test data, return
    per-basin predictions as a dict of {basin_id: {'dates': ..., 'pred': ..., 'obs': ...}}.
    Predictions and observations are z-score normalized (consistent with training).
    """
    edges = load_graph_with_features(EDGE_FILE, basin_ids)

    # Load test data with dates
    per_basin = {}
    all_x_d, all_x_s, all_y, all_dates = [], [], [], []
    for basin in basin_ids:
        ds = get_dataset(cfg=cfg, is_train=False, period="test",
                          basin=basin, scaler=scaler, id_to_int=id_to_int)
        basin_x_d, basin_y, basin_dates = [], [], []
        x_s = None
        for i in range(len(ds)):
            sample = ds[i]
            dyn_tensors = [v for k, v in sorted(sample["x_d"].items())]
            x_d_cat = torch.cat(dyn_tensors, dim=-1)
            basin_x_d.append(x_d_cat)
            basin_y.append(sample["y"])
            basin_dates.append(sample["date"][-1])  # last date in the 30-day window
            if x_s is None:
                parts = []
                if "x_s" in sample: parts.append(sample["x_s"])
                if "x_one_hot" in sample: parts.append(sample["x_one_hot"])
                x_s = torch.cat(parts, dim=-1) if parts else None
        all_x_d.append(torch.stack(basin_x_d))
        all_y.append(torch.stack(basin_y))
        all_x_s.append(x_s)
        all_dates.append(basin_dates)

    x_d = torch.stack(all_x_d, dim=1)
    x_s = torch.stack(all_x_s)
    y = torch.stack(all_y, dim=1)
    n_windows, n_basins, _, _ = x_d.shape

    # Build model with same architecture as training run
    # Read run_config.json to get the exact flags used
    run_config_path = run_dir / "run_config.json"
    run_config = json.load(open(run_config_path)) if run_config_path.exists() else {}
    input_size = run_config.get("input_size", x_d.shape[3] + x_s.shape[1])
    n_dyn = x_d.shape[3]
    n_static_total = x_s.shape[1]

    model = DirectedGraphLSTM(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        edges=edges,
        n_basins=n_basins,
        n_targets=len(cfg.target_variables),
        dropout=cfg.output_dropout,
        initial_forget_bias=cfg.initial_forget_bias,
        use_edge_features=run_config.get("use_edge_features", use_edge_features),
        use_diff_term=run_config.get("use_diff_term", use_diff_term),
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=DEVICE))
    model.eval()

    # Run inference
    preds = np.full((n_windows, n_basins), np.nan)
    with torch.no_grad():
        for w in range(n_windows):
            x_w = x_d[w].transpose(0, 1).to(DEVICE)
            y_hat = model(x_w, x_s.to(DEVICE))
            preds[w] = y_hat[:, -1, 0].cpu().numpy()

    obs = y[:, :, -1, 0].numpy()

    for b, bid in enumerate(basin_ids):
        per_basin[bid] = {
            "dates": all_dates[b],
            "pred": preds[:, b],
            "obs": obs[:, b],
        }
    return per_basin, model


def load_baseline_predictions(cfg, scaler, id_to_int, basin_ids, baseline_run=None):
    """Run the baseline CudaLSTM on test data via direct evaluation."""
    import torch.nn as nn
    if baseline_run is None:
        baseline_run = STRONG_BASELINE
    ckpts = sorted(baseline_run.glob("model_epoch*.pt"))
    ckpt = torch.load(ckpts[-1], map_location=DEVICE, weights_only=True)

    # The baseline uses nn.LSTM. Build equivalent via nn.LSTM to get exact same results as NH eval
    lstm = nn.LSTM(input_size=ckpt["lstm.weight_ih_l0"].shape[1],
                    hidden_size=ckpt["lstm.weight_hh_l0"].shape[1]).to(DEVICE)
    lstm.load_state_dict({
        "weight_ih_l0": ckpt["lstm.weight_ih_l0"],
        "weight_hh_l0": ckpt["lstm.weight_hh_l0"],
        "bias_ih_l0": ckpt["lstm.bias_ih_l0"],
        "bias_hh_l0": ckpt["lstm.bias_hh_l0"],
    })
    head = nn.Linear(64, 1).to(DEVICE)
    head.weight.data = ckpt["head.net.0.weight"]
    head.bias.data = ckpt["head.net.0.bias"]
    lstm.eval(); head.eval()

    per_basin = {}
    for basin in basin_ids:
        ds = get_dataset(cfg=cfg, is_train=False, period="test",
                          basin=basin, scaler=scaler, id_to_int=id_to_int)
        preds, obs_list, dates = [], [], []
        with torch.no_grad():
            for i in range(len(ds)):
                sample = ds[i]
                dyn_tensors = [v for k, v in sorted(sample["x_d"].items())]
                x_d = torch.cat(dyn_tensors, dim=-1)  # [30, n_dyn]
                parts = []
                if "x_s" in sample: parts.append(sample["x_s"])
                if "x_one_hot" in sample: parts.append(sample["x_one_hot"])
                x_s = torch.cat(parts, dim=-1)
                # concat static features to each timestep
                x_s_expanded = x_s.unsqueeze(0).expand(x_d.shape[0], -1)
                x_full = torch.cat([x_d, x_s_expanded], dim=-1).unsqueeze(1)  # [30, 1, n_feat]
                lstm_out, _ = lstm(x_full)
                y_hat = head(lstm_out)  # [30, 1, 1]
                preds.append(y_hat[-1, 0, 0].item())
                obs_list.append(sample["y"][-1, 0].item())
                dates.append(sample["date"][-1])
        per_basin[basin] = {
            "dates": dates,
            "pred": np.array(preds),
            "obs": np.array(obs_list),
        }
    return per_basin


def compute_nse(pred, obs):
    mask = ~np.isnan(obs)
    if mask.sum() < 10 or obs[mask].std() == 0:
        return float("nan")
    p, o = pred[mask], obs[mask]
    return 1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)


# ---------------------------------------------------------------------------
# Hydrograph plot
# ---------------------------------------------------------------------------
def plot_hydrograph(basin, pred_b, pred_g, obs, dates, info, title_prefix, save_path):
    """Single-basin hydrograph: obs vs baseline vs graph prediction."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
    ax.plot(dates, obs, color="black", lw=1.2, label="Observed", alpha=0.9)
    ax.plot(dates, pred_b, color="#ef4444", lw=0.9, alpha=0.75, label="Strong LSTM")
    ax.plot(dates, pred_g, color="#3b82f6", lw=0.9, alpha=0.75, label="Graph+Edge")

    nse_b = compute_nse(pred_b, obs)
    nse_g = compute_nse(pred_g, obs)

    ax.set_xlabel("Date (test period)")
    ax.set_ylabel("Discharge (z-score normalized)")
    ax.set_title(f"{title_prefix}: basin {basin}  |  depth={info['depth']}  "
                 f"|  area={info['area_km2']:.0f} km²  |  upstream={info['n_upstream']}\n"
                 f"NSE: baseline={nse_b:.3f}  graph={nse_g:.3f}  Δ={nse_g - nse_b:+.3f}",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Delta vs topology scatter plots
# ---------------------------------------------------------------------------
def plot_delta_vs_properties(deltas_df, save_path):
    """Scatter plots of NSE delta vs basin properties."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    props = [
        ("n_upstream", "Number of upstream basins", axes[0, 0]),
        ("depth", "Max depth from headwater", axes[0, 1]),
        ("area_km2", "Drainage area (km²)", axes[1, 0]),
        ("elev_m", "Mean elevation (m)", axes[1, 1]),
    ]
    for col, xlabel, ax in props:
        colors = ["#ef4444" if d < 0 else "#3b82f6" for d in deltas_df["delta"]]
        ax.scatter(deltas_df[col], deltas_df["delta"], c=colors, s=60, alpha=0.8,
                    edgecolors="black", linewidths=0.4)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("NSE delta (Graph - Baseline)")
        # Label outlier basins
        threshold = 0.1
        for _, row in deltas_df.iterrows():
            if abs(row["delta"]) > threshold:
                ax.annotate(row["basin"][-5:], (row[col], row["delta"]),
                             fontsize=7, xytext=(4, 4), textcoords="offset points")
        # Set x-scale log for area
        if col == "area_km2":
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
    fig.suptitle("NSE improvement vs basin properties", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Flow regime analysis
# ---------------------------------------------------------------------------
def flow_regime_analysis(baseline_preds, graph_preds, basin_ids, basin_info):
    """For each basin, split predictions by flow quantile and compute NSE in each regime."""
    rows = []
    for bid in basin_ids:
        b = baseline_preds[bid]
        g = graph_preds[bid]
        obs = b["obs"]
        mask = ~np.isnan(obs)
        if mask.sum() < 30:
            continue

        # Split by observed flow quantile
        obs_valid = obs[mask]
        b_pred = b["pred"][mask]
        g_pred = g["pred"][mask]

        q75 = np.quantile(obs_valid, 0.75)
        q25 = np.quantile(obs_valid, 0.25)

        high = obs_valid >= q75
        low = obs_valid <= q25
        mid = (~high) & (~low)

        for regime_name, regime_mask in [("high", high), ("mid", mid), ("low", low)]:
            if regime_mask.sum() < 5:
                continue
            nse_b = compute_nse(b_pred[regime_mask], obs_valid[regime_mask])
            nse_g = compute_nse(g_pred[regime_mask], obs_valid[regime_mask])
            rows.append({
                "basin": bid,
                "depth": basin_info[bid]["depth"],
                "regime": regime_name,
                "n": int(regime_mask.sum()),
                "nse_baseline": nse_b,
                "nse_graph": nse_g,
                "delta": nse_g - nse_b,
            })
    return pd.DataFrame(rows)


def plot_flow_regime(regime_df, save_path):
    """Bar chart: delta NSE by depth and flow regime."""
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    depths = sorted(regime_df["depth"].unique())
    regimes = ["low", "mid", "high"]
    colors = {"low": "#60a5fa", "mid": "#9ca3af", "high": "#ef4444"}
    x = np.arange(len(depths))
    width = 0.27

    for i, regime in enumerate(regimes):
        deltas = []
        for d in depths:
            sub = regime_df[(regime_df["depth"] == d) & (regime_df["regime"] == regime)]
            deltas.append(sub["delta"].median() if len(sub) else 0)
        ax.bar(x + (i - 1) * width, deltas, width, label=f"{regime} flow",
               color=colors[regime], edgecolor="black", linewidth=0.4)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"depth={d}\n(n={sum(regime_df['depth'] == d) // 3})" for d in depths])
    ax.set_ylabel("Median delta NSE (Graph - Baseline)")
    ax.set_title("NSE improvement by depth stratum and flow regime")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Weight inspection
# ---------------------------------------------------------------------------
def inspect_weights(graph_model, basin_ids, save_path):
    """Plot what the graph model learned: W_out spectrum, W_msg_edge feature sensitivity."""
    # Get trained weights
    W_out = graph_model.W_out.weight.detach().numpy()  # [64, 64]
    W_msg = graph_model.W_msg_edge.weight.detach().numpy()  # [64, msg_in_dim]

    # W_out spectrum: the tanh(W_out * m) residual strength
    u, s, _ = np.linalg.svd(W_out)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(s, marker="o")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Singular value index")
    axes[0].set_ylabel("Singular value")
    axes[0].set_title("W_out spectrum (residual projection)")
    axes[0].grid(alpha=0.3)

    # W_msg_edge columns: first 64 are h_u inputs, last 3 are edge features (or 67-64=3)
    n_input = W_msg.shape[1]
    # Compute per-column L2 norm
    col_norms = np.linalg.norm(W_msg, axis=0)
    labels = []
    if n_input == 67:
        labels = ["h_u"] * 64 + ["log(dist)", "log(area_ratio)", "elev_drop"]
    elif n_input == 131:
        labels = ["h_u"] * 64 + ["h_u-h_v"] * 64 + ["log(dist)", "log(area_ratio)", "elev_drop"]
    else:
        labels = [f"in_{i}" for i in range(n_input)]

    # Plot: bar chart of column norms, grouped by input type
    axes[1].bar(range(n_input), col_norms, color="#3b82f6", edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("Input dimension")
    axes[1].set_ylabel("Column L2 norm")
    axes[1].set_title("W_msg_edge input sensitivity")
    # Highlight edge feature columns
    edge_feat_start = n_input - 3
    axes[1].axvspan(edge_feat_start - 0.5, n_input - 0.5, alpha=0.2, color="red",
                     label="edge features")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Return summary
    n_hu = 64
    hu_avg = col_norms[:n_hu].mean()
    edge_avg = col_norms[-3:].mean()
    result = {
        "W_out_max_singular_value": float(s.max()),
        "W_out_mean_singular_value": float(s.mean()),
        "W_msg_hu_avg_norm": float(hu_avg),
        "W_msg_edge_avg_norm": float(edge_avg),
        "W_msg_edge_dim_norms": {
            "log_dist": float(col_norms[edge_feat_start]),
            "log_area_ratio": float(col_norms[edge_feat_start + 1]),
            "elev_drop": float(col_norms[edge_feat_start + 2]),
        },
    }
    if n_input == 131:
        result["W_msg_diff_avg_norm"] = float(col_norms[64:128].mean())
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Graph run (Edge, full): {GRAPH_RUN}")
    print(f"Graph run (Edge+Frozen): {GRAPH_FROZEN_RUN}")

    # Each graph run has its own baseline; detect it
    graph_baseline = detect_baseline(GRAPH_RUN)
    frozen_baseline = detect_baseline(GRAPH_FROZEN_RUN)
    print(f"  graph_run was trained from: {graph_baseline}")
    print(f"  frozen_run was trained from: {frozen_baseline}")
    print()

    basin_ids = [l.strip() for l in open(BASIN_FILE) if l.strip()]
    basin_info = load_basin_info()

    # Use strong baseline as the REFERENCE in all comparisons (it's the fair bar)
    reference_cfg = Config(STRONG_BASELINE / "config.yml")
    reference_scaler = load_scaler(STRONG_BASELINE)
    reference_id_to_int = load_basin_id_encoding(STRONG_BASELINE)

    print("Loading STRONG baseline predictions (reference)...")
    baseline_preds = load_baseline_predictions(reference_cfg, reference_scaler,
                                                 reference_id_to_int, basin_ids,
                                                 STRONG_BASELINE)

    print(f"\nLoading graph (Edge, full) predictions [trained from {graph_baseline.name}]...")
    gcfg = Config(graph_baseline / "config.yml")
    gscaler = load_scaler(graph_baseline)
    gid_to_int = load_basin_id_encoding(graph_baseline) if gcfg.use_basin_id_encoding else {}
    best_ckpt = GRAPH_RUN / "model_best.pt"
    graph_preds, graph_model = load_and_evaluate_graph_model(
        best_ckpt, GRAPH_RUN, gcfg, gscaler, gid_to_int, basin_ids)

    print(f"\nLoading graph (Edge+Frozen) predictions [trained from {frozen_baseline.name}]...")
    fcfg = Config(frozen_baseline / "config.yml")
    fscaler = load_scaler(frozen_baseline)
    fid_to_int = load_basin_id_encoding(frozen_baseline) if fcfg.use_basin_id_encoding else {}
    frozen_ckpt = GRAPH_FROZEN_RUN / "model_best.pt"
    frozen_preds, frozen_model = load_and_evaluate_graph_model(
        frozen_ckpt, GRAPH_FROZEN_RUN, fcfg, fscaler, fid_to_int, basin_ids)

    # -------------------------------------------------------------------
    # 1) NSE deltas per basin
    # -------------------------------------------------------------------
    print("\nComputing NSE per basin for all three models...")
    rows = []
    for bid in basin_ids:
        info = basin_info[bid]
        b = baseline_preds[bid]
        g = graph_preds[bid]
        f = frozen_preds[bid]
        nse_b = compute_nse(b["pred"], b["obs"])
        nse_g = compute_nse(g["pred"], g["obs"])
        nse_f = compute_nse(f["pred"], f["obs"])
        rows.append({
            "basin": bid,
            "role": info["role"],
            "depth": info["depth"],
            "n_upstream": info["n_upstream"],
            "area_km2": info["area_km2"],
            "elev_m": info["elev_m"],
            "nse_baseline": nse_b,
            "nse_graph": nse_g,
            "nse_frozen": nse_f,
            "delta": nse_g - nse_b,
            "delta_frozen": nse_f - nse_b,
        })
    deltas_df = pd.DataFrame(rows)
    deltas_df.to_csv(OUT_DIR / "per_basin_analysis.csv", index=False)

    # -------------------------------------------------------------------
    # 2) Hydrographs for representative basins
    # -------------------------------------------------------------------
    print("Generating hydrographs...")
    # Pick: best graph improvement (deep), worst (catastrophic), one neutral
    deltas_sorted = deltas_df.sort_values("delta", ascending=False)
    key_basins = [
        deltas_sorted.iloc[0]["basin"],       # largest improvement
        deltas_sorted.iloc[-1]["basin"],      # largest degradation
        "08189500",                             # depth-3 outlet
        "08158700",                             # depth-1 big improvement
    ]
    key_basins = list(dict.fromkeys(key_basins))  # dedup, preserve order

    for bid in key_basins:
        info = basin_info[bid]
        b = baseline_preds[bid]
        g = graph_preds[bid]
        plot_hydrograph(bid, b["pred"], g["pred"], b["obs"],
                         b["dates"], info,
                         f"Graph+Edge vs Baseline",
                         OUT_DIR / f"hydrograph_{bid}.png")

    # -------------------------------------------------------------------
    # 3) Delta vs topology
    # -------------------------------------------------------------------
    print("Plotting delta vs basin properties...")
    plot_delta_vs_properties(deltas_df, OUT_DIR / "delta_vs_properties.png")

    # -------------------------------------------------------------------
    # 4) Flow regime analysis
    # -------------------------------------------------------------------
    print("Flow regime analysis...")
    regime_df = flow_regime_analysis(baseline_preds, graph_preds, basin_ids, basin_info)
    regime_df.to_csv(OUT_DIR / "flow_regime_nse.csv", index=False)
    plot_flow_regime(regime_df, OUT_DIR / "flow_regime_delta.png")

    # -------------------------------------------------------------------
    # 5) Learned weights inspection (frozen model is cleanest — only these
    #    weights matter there)
    # -------------------------------------------------------------------
    print("Inspecting learned weights (frozen model)...")
    weight_stats = inspect_weights(frozen_model, basin_ids, OUT_DIR / "learned_weights_frozen.png")
    with open(OUT_DIR / "learned_weights_summary.json", "w") as f:
        json.dump(weight_stats, f, indent=2)

    # -------------------------------------------------------------------
    # Summary printout
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\nOverall NSE medians:")
    print(f"  Baseline: {deltas_df['nse_baseline'].median():.3f}")
    print(f"  Graph+Edge: {deltas_df['nse_graph'].median():.3f}")
    print(f"  Graph+Edge+Frozen: {deltas_df['nse_frozen'].median():.3f}")

    print(f"\nBy depth (median delta = Graph - Baseline):")
    for d in sorted(deltas_df["depth"].unique()):
        sub = deltas_df[deltas_df["depth"] == d]
        print(f"  depth {d}: n={len(sub)}  delta_median={sub['delta'].median():+.3f}  "
              f"delta_frozen_median={sub['delta_frozen'].median():+.3f}")

    print(f"\nWeight analysis (frozen model):")
    print(f"  W_out max singular value: {weight_stats['W_out_max_singular_value']:.3f}")
    print(f"  W_msg h_u avg column norm: {weight_stats['W_msg_hu_avg_norm']:.3f}")
    print(f"  W_msg edge-feat avg norm: {weight_stats['W_msg_edge_avg_norm']:.3f}")
    print(f"  Edge feature breakdown:")
    for k, v in weight_stats["W_msg_edge_dim_norms"].items():
        print(f"    {k}: {v:.3f}")

    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
