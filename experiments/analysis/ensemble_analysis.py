"""Ensemble analysis: do the 3 graph variants make COMPLEMENTARY errors?

If each variant captures different aspects of the upstream signal, averaging their
predictions should beat any individual variant. If they all make the same errors,
ensembling won't help.

Also: check error correlation across variants. Low correlation = potential for ensembling.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Repo root is three levels up from experiments/analysis/<script>.py.
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neuralhydrology.datasetzoo import get_dataset
from neuralhydrology.datautils.utils import load_scaler
from neuralhydrology.evaluation.utils import load_basin_id_encoding
from neuralhydrology.utils.config import Config
from train_graph_lstm import DirectedGraphLSTM, load_graph_with_features

ROOT = Path(__file__).parent.parent.parent
STRONG_BASELINE = ROOT / "runs" / "05_lstm_23basin_strong_baseline"
EDGE_FILE = ROOT / "topology_analysis/phase1_network_discovery/outputs/study_network_edges.csv"
BASIN_FILE = ROOT / "experiments/basin_lists/study_network_basins.txt"
DEVICE = torch.device("cpu")

# Variants to ensemble
VARIANTS = {
    "mean_agg": ROOT / "runs" / "06_graph_edge_warm_full",
    "attention": ROOT / "runs" / "09_graph_edge_attention",
    "sigmoid_gate": ROOT / "runs" / "10_graph_edge_sigmoid_gate",
}
VARIANTS = {k: v for k, v in VARIANTS.items() if v.exists()}


def compute_nse(pred, obs):
    mask = ~np.isnan(obs)
    if mask.sum() < 10 or obs[mask].std() == 0:
        return float("nan")
    p, o = pred[mask], obs[mask]
    return 1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)


def load_all_test_data(cfg, scaler, id_to_int, basin_ids):
    all_x_d, all_x_s, all_y = [], [], []
    for basin in basin_ids:
        ds = get_dataset(cfg=cfg, is_train=False, period="test",
                          basin=basin, scaler=scaler, id_to_int=id_to_int)
        bx_d, by = [], []
        x_s = None
        for i in range(len(ds)):
            sample = ds[i]
            dyn = [v for k, v in sorted(sample["x_d"].items())]
            bx_d.append(torch.cat(dyn, dim=-1))
            by.append(sample["y"])
            if x_s is None:
                parts = []
                if "x_s" in sample: parts.append(sample["x_s"])
                if "x_one_hot" in sample: parts.append(sample["x_one_hot"])
                x_s = torch.cat(parts, dim=-1) if parts else None
        all_x_d.append(torch.stack(bx_d))
        all_y.append(torch.stack(by))
        all_x_s.append(x_s)
    return (torch.stack(all_x_d, dim=1), torch.stack(all_x_s),
            torch.stack(all_y, dim=1))


def predict_variant(run_dir, cfg, x_d, x_s, basin_ids):
    edges = load_graph_with_features(EDGE_FILE, basin_ids)
    run_cfg = json.load(open(run_dir / "run_config.json"))
    model = DirectedGraphLSTM(
        input_size=run_cfg["input_size"], hidden_size=cfg.hidden_size,
        edges=edges, n_basins=x_d.shape[1],
        n_targets=len(cfg.target_variables),
        dropout=cfg.output_dropout, initial_forget_bias=cfg.initial_forget_bias,
        use_edge_features=run_cfg.get("use_edge_features", True),
        use_diff_term=run_cfg.get("use_diff_term", False),
        use_attention=run_cfg.get("use_attention", False),
        use_sigmoid_gate=run_cfg.get("use_sigmoid_gate", False),
    ).to(DEVICE)
    model.load_state_dict(torch.load(run_dir / "model_best.pt",
                                       weights_only=True, map_location=DEVICE))
    model.eval()
    n_windows = x_d.shape[0]
    preds = np.full((n_windows, x_d.shape[1]), np.nan)
    x_s_dev = x_s.to(DEVICE)
    with torch.no_grad():
        for w in range(n_windows):
            x_dw = x_d[w].transpose(0, 1).to(DEVICE)
            y_hat = model(x_dw, x_s_dev)
            preds[w] = y_hat[:, -1, 0].numpy()
    return preds


def main():
    basin_ids = [l.strip() for l in open(BASIN_FILE) if l.strip()]
    cfg = Config(STRONG_BASELINE / "config.yml")
    scaler = load_scaler(STRONG_BASELINE)
    id_to_int = load_basin_id_encoding(STRONG_BASELINE)
    x_d, x_s, y = load_all_test_data(cfg, scaler, id_to_int, basin_ids)
    obs = y[:, :, -1, 0].numpy()

    # Baseline (nn.LSTM)
    ckpts = sorted(STRONG_BASELINE.glob("model_epoch*.pt"))
    ckpt = torch.load(ckpts[-1], map_location=DEVICE, weights_only=True)
    lstm = nn.LSTM(input_size=ckpt["lstm.weight_ih_l0"].shape[1], hidden_size=64).to(DEVICE)
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
    n_windows, n_basins, seq_len, _ = x_d.shape
    pred_base = np.full((n_windows, n_basins), np.nan)
    with torch.no_grad():
        for w in range(n_windows):
            x_dw = x_d[w]
            x_sw_exp = x_s.unsqueeze(1).expand(-1, seq_len, -1)
            x_full = torch.cat([x_dw, x_sw_exp], dim=-1).transpose(0, 1)
            lstm_out, _ = lstm(x_full)
            y_hat = head(lstm_out[-1])
            pred_base[w] = y_hat[:, 0].numpy()

    # Variants
    preds = {"baseline": pred_base}
    for name, run_dir in VARIANTS.items():
        print(f"Predicting with {name} ({run_dir.name})...")
        preds[name] = predict_variant(run_dir, cfg, x_d, x_s, basin_ids)

    # Ensemble: simple mean of 3 graph variants
    graph_names = [k for k in preds if k != "baseline"]
    ensemble = np.mean([preds[k] for k in graph_names], axis=0)
    preds["ensemble_mean"] = ensemble

    # Per-basin NSE
    results = []
    for name in preds:
        nse_list = [compute_nse(preds[name][:, b], obs[:, b]) for b in range(n_basins)]
        results.append({"variant": name, "median_nse": float(np.nanmedian(nse_list))})

    # Print
    print("\n" + "=" * 60)
    print("Median NSE by variant")
    print("=" * 60)
    for r in results:
        print(f"  {r['variant']:20s}: {r['median_nse']:.4f}")

    # Error correlation analysis
    print("\n" + "=" * 60)
    print("Per-basin error vector correlations (1 = identical errors)")
    print("=" * 60)
    errs = {}
    for name in preds:
        err = preds[name] - obs
        errs[name] = err.reshape(-1)  # flatten windows x basins
        mask = ~np.isnan(errs[name])
        errs[name] = errs[name][mask]

    # Need to align masks
    masks = {name: ~np.isnan(preds[name] - obs) for name in preds}
    common_mask = np.logical_and.reduce(list(masks.values())).reshape(-1)
    flat_errs = {name: (preds[name] - obs).reshape(-1)[common_mask] for name in preds}

    print(f"\nUsing {common_mask.sum()} common valid predictions")
    print()
    print(f"{'':>20s}  " + "  ".join(f"{n:>15s}" for n in graph_names))
    for n1 in graph_names:
        row = [f"{n1:>20s}"]
        for n2 in graph_names:
            r = np.corrcoef(flat_errs[n1], flat_errs[n2])[0, 1]
            row.append(f"{r:>15.4f}")
        print("  ".join(row))

    # Compare to baseline errors
    print(f"\nError correlation with BASELINE:")
    for n in graph_names + ["ensemble_mean"]:
        r = np.corrcoef(flat_errs[n], flat_errs["baseline"])[0, 1]
        print(f"  {n:20s}: {r:.4f}")

    # Per-basin: best variant per basin
    print("\n" + "=" * 60)
    print("Per-basin: which variant wins?")
    print("=" * 60)
    header_cols = ["basin"] + list(preds.keys())
    rows_perbasin = []
    for b, bid in enumerate(basin_ids):
        row = {"basin": bid}
        for name in preds:
            row[name] = compute_nse(preds[name][:, b], obs[:, b])
        row["best_variant"] = max(graph_names + ["ensemble_mean"],
                                    key=lambda n: row[n] if not np.isnan(row[n]) else -1e9)
        row["best_nse"] = row[row["best_variant"]]
        rows_perbasin.append(row)
    df = pd.DataFrame(rows_perbasin)

    # Count wins per variant
    from collections import Counter
    wins = Counter(df["best_variant"])
    print(f"\nWins (basins where this variant had highest NSE):")
    for name, count in wins.most_common():
        print(f"  {name:20s}: {count}")

    # Show per-basin for interesting ones
    print(f"\nPer-basin details:")
    interesting = ["08195000", "08190500", "08189500", "08158700"]
    for bid in interesting:
        row = df[df["basin"] == bid].iloc[0]
        print(f"  {bid}: " + "  ".join(f"{n}={row[n]:+.3f}" for n in preds if n != "basin"))

    # Output CSV
    df.to_csv(ROOT / "experiments" / "analysis_outputs" / "ensemble_analysis.csv", index=False)


if __name__ == "__main__":
    main()
