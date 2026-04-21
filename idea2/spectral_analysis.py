"""Step 1 of the pre-registered hypothesis (HYPOTHESIS.md §6).

Post-hoc spectral analysis of existing run predictions.

Goal: determine whether DirectedGraph-LSTM residual-error power in the high-
frequency band differs from the strong-baseline residual-error power. If it
does NOT (Falsifier A territory), the temporal-lag-preserves-high-freq claim
is wounded and we revisit the hypothesis before running anything on Component 0.

For each of three models:
  * BASELINE  : strong Kratzert-style LSTM (run 05). Implemented as the
                DirectedGraph-LSTM class with NO edges, warm-started from run 05.
                Architecturally equivalent predictions (LSTMCell unroll vs. nn.LSTM
                gives numerically identical outputs for matched weights & zero init).
  * GRAPH+WARM: full-finetune Graph-LSTM with edge features (run 06, headline).
  * GRAPH+FROZEN: frozen-LSTM graph (run 07, isolates the pure graph contribution).

We dump per-window predictions on the test period, compute residuals vs.
observations, take the Welch PSD of each basin's residual series, stratify by
basin depth, and produce:

  * experiments/analysis_outputs/spectral/psd_by_depth.png
  * experiments/analysis_outputs/spectral/psd_by_basin.csv
  * experiments/analysis_outputs/spectral/high_freq_power_summary.csv
  * experiments/analysis_outputs/spectral/predictions.npz  (cache)

Usage:
    /Applications/anaconda3/envs/nh/bin/python experiments/spectral_analysis.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import signal as sp_signal

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.evaluation.utils import load_basin_id_encoding
from neuralhydrology.utils.config import Config

from train_graph_lstm import (
    DirectedGraphLSTM,
    load_graph_with_features,
    load_basin_data,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
LOGGER = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
BASELINE_DIR = ROOT / "runs/05_lstm_23basin_strong_baseline"
GRAPH_WARM_DIR = ROOT / "runs/06_graph_edge_warm_full"
GRAPH_FROZEN_DIR = ROOT / "runs/07_graph_edge_frozen"
BASIN_FILE = ROOT / "experiments/study_network_basins.txt"
EDGE_FILE = ROOT / "topology_analysis/phase1_network_discovery/outputs/study_network_edges.csv"
SUMMARY_FILE = ROOT / "topology_analysis/phase1_network_discovery/outputs/study_network_summary.txt"
OUT_DIR = ROOT / "experiments/analysis_outputs/spectral"

DEVICE = torch.device("cpu")
HIDDEN_SIZE = 64
DROPOUT = 0.4


def load_basin_depth_map():
    """Parse depth info from study_network_summary.txt."""
    depth = {}
    lines = open(SUMMARY_FILE).readlines()
    in_table = False
    for line in lines:
        line = line.strip()
        if line.startswith("Basin") and "Area_km2" in line:
            in_table = True
            continue
        if line.startswith("-----"):
            continue
        if in_table and line and not line.startswith("Edge"):
            parts = line.split()
            if len(parts) >= 5:
                depth[parts[0]] = int(parts[3])
        if line.startswith("Edge list:"):
            break
    return depth


def build_model(input_size, n_basins, edges, use_edge_features, initial_forget_bias):
    return DirectedGraphLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        edges=edges,
        n_basins=n_basins,
        n_targets=1,
        dropout=DROPOUT,
        initial_forget_bias=initial_forget_bias,
        use_edge_features=use_edge_features,
        use_diff_term=False,
        use_attention=False,
        use_sigmoid_gate=False,
    ).to(DEVICE)


def warm_start_from_baseline_ckpt(model, ckpt_path):
    """Copy nn.LSTM weights from the NH baseline into our LSTMCell model."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    mapping = {
        "lstm.weight_ih_l0": "lstm_cell.weight_ih",
        "lstm.weight_hh_l0": "lstm_cell.weight_hh",
        "lstm.bias_ih_l0": "lstm_cell.bias_ih",
        "lstm.bias_hh_l0": "lstm_cell.bias_hh",
        "head.net.0.weight": "head.weight",
        "head.net.0.bias": "head.bias",
    }
    own = model.state_dict()
    copied = 0
    for src, dst in mapping.items():
        if src in ckpt and dst in own and ckpt[src].shape == own[dst].shape:
            own[dst].copy_(ckpt[src])
            copied += 1
    LOGGER.info(f"  copied {copied} tensors from {ckpt_path.name}")


def evaluate_dump(model, x_d, x_s):
    """Return predictions [n_windows, n_basins] — one per last-step-of-window."""
    model.eval()
    n_windows = x_d.shape[0]
    n_basins = x_d.shape[1]
    preds = np.full((n_windows, n_basins), np.nan, dtype=np.float32)
    x_s_dev = x_s.to(DEVICE)
    with torch.no_grad():
        for w in range(n_windows):
            x_dw = x_d[w].transpose(0, 1).to(DEVICE)
            y_hat = model(x_dw, x_s_dev)
            preds[w] = y_hat[:, -1, 0].cpu().numpy()
    return preds


def get_predictions():
    """Run all three models on the same test data, return dict of predictions."""
    cache = OUT_DIR / "predictions.npz"
    if cache.exists():
        LOGGER.info(f"Loading cached predictions from {cache}")
        data = np.load(cache, allow_pickle=True)
        return {
            "basins": list(data["basins"]),
            "obs": data["obs"],
            "baseline": data["baseline"],
            "graph_warm": data["graph_warm"],
            "graph_frozen": data["graph_frozen"],
        }

    basin_ids = [l.strip() for l in open(BASIN_FILE) if l.strip()]
    edges = load_graph_with_features(EDGE_FILE, basin_ids)
    LOGGER.info(f"Basins: {len(basin_ids)}   Edges: {len(edges)}")

    cfg = Config(BASELINE_DIR / "config.yml")
    scaler = load_scaler(BASELINE_DIR)
    id_to_int = load_basin_id_encoding(BASELINE_DIR) if cfg.use_basin_id_encoding else {}

    LOGGER.info("Loading test-period data for all basins...")
    x_d_test, x_s, y_test = load_basin_data(cfg, scaler, basin_ids, "test", id_to_int)
    LOGGER.info(f"Test tensor: {x_d_test.shape}   Static dim: {x_s.shape[1]}")

    n_dyn = x_d_test.shape[3]
    n_basins = x_d_test.shape[1]
    input_size = n_dyn + x_s.shape[1]

    results = {}
    obs = y_test[:, :, -1, 0].numpy()
    results["obs"] = obs
    results["basins"] = basin_ids

    # -------- BASELINE (no edges, warm-started from run 05) ---------------
    LOGGER.info("Model 1/3: baseline (empty-graph warm-started from run 05)")
    m_base = build_model(input_size, n_basins, edges=[],
                         use_edge_features=False,
                         initial_forget_bias=cfg.initial_forget_bias)
    base_ckpt = sorted(BASELINE_DIR.glob("model_epoch*.pt"))[-1]
    warm_start_from_baseline_ckpt(m_base, base_ckpt)
    results["baseline"] = evaluate_dump(m_base, x_d_test, x_s)

    # -------- GRAPH + WARM  (run 06) --------------------------------------
    LOGGER.info("Model 2/3: graph+warm (run 06)")
    m_warm = build_model(input_size, n_basins, edges=edges,
                         use_edge_features=True,
                         initial_forget_bias=cfg.initial_forget_bias)
    m_warm.load_state_dict(
        torch.load(GRAPH_WARM_DIR / "model_best.pt", map_location="cpu", weights_only=True)
    )
    results["graph_warm"] = evaluate_dump(m_warm, x_d_test, x_s)

    # -------- GRAPH + FROZEN (run 07) -------------------------------------
    LOGGER.info("Model 3/3: graph+frozen (run 07)")
    m_frozen = build_model(input_size, n_basins, edges=edges,
                           use_edge_features=True,
                           initial_forget_bias=cfg.initial_forget_bias)
    m_frozen.load_state_dict(
        torch.load(GRAPH_FROZEN_DIR / "model_best.pt", map_location="cpu", weights_only=True)
    )
    results["graph_frozen"] = evaluate_dump(m_frozen, x_d_test, x_s)

    np.savez(cache,
             basins=np.array(basin_ids, dtype=object),
             obs=results["obs"],
             baseline=results["baseline"],
             graph_warm=results["graph_warm"],
             graph_frozen=results["graph_frozen"])
    LOGGER.info(f"Cached predictions -> {cache}")
    return results


def welch_psd(series, fs=1.0, nperseg=None):
    """Welch PSD; skip NaNs via interpolation of short gaps."""
    s = np.asarray(series, dtype=float)
    if np.all(np.isnan(s)):
        return None, None
    if np.any(np.isnan(s)):
        # Linear-interpolate interior NaNs; drop leading/trailing NaNs
        idx = np.arange(len(s))
        good = ~np.isnan(s)
        if good.sum() < 16:
            return None, None
        s = np.interp(idx, idx[good], s[good])
    if nperseg is None:
        nperseg = min(256, len(s) // 4) if len(s) >= 64 else len(s)
    if nperseg < 8:
        return None, None
    f, Pxx = sp_signal.welch(s, fs=fs, nperseg=nperseg, detrend="constant")
    return f, Pxx


def band_power(f, Pxx, band_frac=(0.75, 1.0)):
    """Return integrated power in [band_frac[0]*f_nyq, band_frac[1]*f_nyq]."""
    if f is None:
        return np.nan
    f_nyq = f[-1]
    lo, hi = band_frac[0] * f_nyq, band_frac[1] * f_nyq
    mask = (f >= lo) & (f <= hi)
    if mask.sum() < 2:
        return np.nan
    return np.trapz(Pxx[mask], f[mask])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    preds = get_predictions()
    basin_ids = preds["basins"]
    depth = load_basin_depth_map()

    obs = preds["obs"]                  # [n_windows, n_basins]
    base = preds["baseline"]
    warm = preds["graph_warm"]
    frozen = preds["graph_frozen"]

    resid_base = base - obs
    resid_warm = warm - obs
    resid_frozen = frozen - obs

    # ---- Per-basin PSD for residuals --------------------------------------
    rows = []
    psd_store = {"baseline": {}, "graph_warm": {}, "graph_frozen": {}}
    for i, bid in enumerate(basin_ids):
        for name, r in (("baseline", resid_base[:, i]),
                        ("graph_warm", resid_warm[:, i]),
                        ("graph_frozen", resid_frozen[:, i])):
            f, Pxx = welch_psd(r)
            if f is None:
                continue
            hp = band_power(f, Pxx, (0.75, 1.0))
            mp = band_power(f, Pxx, (0.25, 0.75))
            lp = band_power(f, Pxx, (0.0, 0.25))
            total = hp + mp + lp
            psd_store[name][bid] = (f, Pxx)
            rows.append({
                "basin": bid,
                "depth": depth.get(bid, -1),
                "model": name,
                "low_power_0_25": lp,
                "mid_power_25_75": mp,
                "high_power_75_100": hp,
                "total_power": total,
                "high_frac": hp / total if total > 0 else np.nan,
                "obs_std": np.nanstd(obs[:, i]),
                "resid_std": np.nanstd(r),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "psd_by_basin.csv", index=False)
    LOGGER.info(f"Wrote per-basin PSD summary: {OUT_DIR/'psd_by_basin.csv'}")

    # ---- High-frequency summary stratified by depth -----------------------
    summary_rows = []
    for d in sorted(df["depth"].unique()):
        if d < 0:
            continue
        sub = df[df["depth"] == d]
        piv = sub.pivot_table(index="basin", columns="model",
                               values="high_power_75_100", aggfunc="first")
        if not {"baseline", "graph_warm", "graph_frozen"}.issubset(piv.columns):
            continue
        n = len(piv)
        summary_rows.append({
            "depth": d, "n_basins": n,
            "baseline_median_highpow": piv["baseline"].median(),
            "graph_warm_median_highpow": piv["graph_warm"].median(),
            "graph_frozen_median_highpow": piv["graph_frozen"].median(),
            "warm_vs_base_ratio": piv["graph_warm"].median() / piv["baseline"].median(),
            "frozen_vs_base_ratio": piv["graph_frozen"].median() / piv["baseline"].median(),
            "warm_vs_base_pct_reduction": 100 * (1 - piv["graph_warm"].median() / piv["baseline"].median()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "high_freq_power_summary.csv", index=False)
    LOGGER.info("High-frequency-band residual power, by depth:")
    LOGGER.info("\n" + summary_df.to_string(index=False))

    # ---- PSD figure: mean PSD per model per depth stratum -----------------
    depths = sorted(set(d for d in df["depth"].unique() if d >= 0))
    fig, axes = plt.subplots(1, len(depths), figsize=(4 * len(depths), 3.5), sharey=True)
    if len(depths) == 1:
        axes = [axes]
    colors = {"baseline": "#444", "graph_warm": "#1f77b4", "graph_frozen": "#d62728"}
    for ax, d in zip(axes, depths):
        ax.set_title(f"depth={d}  (n={sum(depth.get(b, -1) == d for b in basin_ids)})")
        for name in ("baseline", "graph_warm", "graph_frozen"):
            psds = []
            for bid in basin_ids:
                if depth.get(bid, -1) != d:
                    continue
                if bid not in psd_store[name]:
                    continue
                f, Pxx = psd_store[name][bid]
                psds.append((f, Pxx))
            if not psds:
                continue
            f0 = psds[0][0]
            # Resample all PSDs to f0 for mean (they should already share f due to equal-length)
            stacked = np.vstack([p[1] for p in psds if len(p[1]) == len(f0)])
            median = np.median(stacked, axis=0)
            ax.loglog(f0, median, color=colors[name], label=name, lw=1.5)
        ax.set_xlabel("frequency (cycles / day)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("residual power (median across basins)")
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.suptitle("Residual error PSD, stratified by basin depth", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "psd_by_depth.png", dpi=140)
    LOGGER.info(f"Wrote figure: {OUT_DIR/'psd_by_depth.png'}")

    # ---- Verdict print ----------------------------------------------------
    LOGGER.info("=" * 70)
    LOGGER.info("VERDICT (per HYPOTHESIS.md Falsifier A and evidentiary bar §4 item 1):")
    LOGGER.info("  At each depth stratum, compare graph_warm high-freq power vs. baseline.")
    LOGGER.info("  Target: graph_warm high-freq residual power LOWER than baseline by >=15%.")
    LOGGER.info("=" * 70)
    for _, row in summary_df.iterrows():
        pct = row["warm_vs_base_pct_reduction"]
        verdict = "PASS (>=15% reduction)" if pct >= 15 else (
            "SOFT PASS (>=0% reduction)" if pct >= 0 else "FAIL (increase)"
        )
        LOGGER.info(f"  depth={int(row['depth'])}  n={int(row['n_basins']):2d}  "
                     f"Δhighpow={pct:+6.1f}%   {verdict}")

    # Dump a small JSON decision record
    decision = {
        "n_depth_strata_pass_15pct": int((summary_df["warm_vs_base_pct_reduction"] >= 15).sum()),
        "n_depth_strata_soft_pass_0pct": int((summary_df["warm_vs_base_pct_reduction"] >= 0).sum()),
        "n_depth_strata_fail": int((summary_df["warm_vs_base_pct_reduction"] < 0).sum()),
        "per_depth": summary_df.to_dict(orient="records"),
    }
    with open(OUT_DIR / "decision_record.json", "w") as f:
        json.dump(decision, f, indent=2, default=float)
    LOGGER.info(f"Wrote decision record: {OUT_DIR/'decision_record.json'}")


if __name__ == "__main__":
    main()
