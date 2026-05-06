"""A/B/C scaled comparison on Component 0 (183 basins).

Reads the test_metrics.csv files from runs 14 (Condition A), 15 (Condition B),
and 16 (Condition C — pending) and produces:

  experiments/analysis_outputs/abc_component0/
    summary.json                    cross-condition median NSE + per-basin Δ stats
    per_basin_long.csv              one row per (condition, seed, basin)
    per_basin_deltas.csv            wide table with A, B, C per-basin NSE + B-A, C-A, C-B
    delta_distributions.png         histograms of B-A, C-A, C-B per-basin Δ
    nse_by_depth.png                median NSE per condition per graph depth
    summary_table.txt               human-readable headline numbers

Usage:
    python experiments/analysis/compare_abc_component0.py [--seeds 42 11 13 17 19 23]

If a condition is missing or has fewer seeds than requested, the script reports
what it found and proceeds with what's available (so partial-result
intermediate states are still informative).
"""
import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
OUT_DIR = ROOT / "experiments" / "analysis_outputs" / "abc_component0"

# Glob patterns: {seed} fills in seed number. Each pattern picks up either the
# numbered run (run 14/15/16) for seed=42, OR the multi-seed-suffix runs added later.
PATTERNS = {
    "A_baseline": [
        # numbered seed-42 run
        f"{ROOT}/runs/14_lstm_component0_baseline_seed42/test/model_epoch030/test_metrics.csv",
        # multi-seed runs (when added)
        f"{ROOT}/runs/lstm_component0_baseline_seed*/test/model_epoch030/test_metrics.csv",
    ],
    "B_topology_features": [
        f"{ROOT}/runs/15_graph_c0_topology_features_seed42/test_metrics.csv",
        f"{ROOT}/runs/graph_c0_topology_features_seed*/test_metrics.csv",
    ],
    "C_graph_messages": [
        f"{ROOT}/runs/16_graph_c0_warm_seed42/test_metrics.csv",
        f"{ROOT}/runs/graph_c0_warm_seed*/test_metrics.csv",
    ],
}

# Topology-depth file (from Phase 1 inference) for depth-stratified analysis
DEPTH_FILE = ROOT / "topology_analysis" / "phase1_network_discovery" / "outputs" / "component0_depth.csv"


def load_metrics(patterns, exclude="SMOKE"):
    """Load test_metrics.csv files matching any pattern; return {seed: {basin: NSE}}."""
    out = {}
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            if exclude and exclude in p:
                continue
            m = re.search(r"seed(\d+)", p)
            if not m:
                continue
            seed = int(m.group(1))
            if seed in out:
                continue  # first match wins (numbered run takes precedence)
            df = pd.read_csv(p, dtype={"basin": str})
            out[seed] = df.set_index("basin")["NSE"].to_dict()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                         help="Restrict to these seeds; default = all found")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all conditions
    results = {}
    for cond, pats in PATTERNS.items():
        loaded = load_metrics(pats)
        if args.seeds is not None:
            loaded = {s: v for s, v in loaded.items() if s in args.seeds}
        results[cond] = loaded
        if loaded:
            print(f"{cond}: {len(loaded)} seed(s) found: {sorted(loaded.keys())}")
        else:
            print(f"{cond}: NO RESULTS YET")
    print()

    # Per-condition summary
    summary = {}
    for label, r in results.items():
        if not r:
            continue
        medians = sorted(np.median(list(d.values())) for d in r.values())
        means = [np.mean(list(d.values())) for d in r.values()]
        summary[label] = {
            "n_seeds": len(medians),
            "seeds": sorted(r.keys()),
            "median_NSE_per_seed": [float(x) for x in medians],
            "mean_NSE_per_seed": [float(x) for x in means],
            "cross_seed_median": float(np.median(medians)),
            "cross_seed_std": float(np.std(medians)) if len(medians) > 1 else 0.0,
            "cross_seed_mean_of_means": float(np.mean(means)),
        }

    # Per-basin long-format table
    long_rows = [{"condition": c, "seed": s, "basin": b, "NSE": v}
                  for c, r in results.items()
                  for s, d in r.items()
                  for b, v in d.items()]
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(OUT_DIR / "per_basin_long.csv", index=False)

    # Per-basin wide table with deltas, restricted to seeds present in all loaded conditions
    common_seeds = set.intersection(*[set(r.keys()) for r in results.values() if r]) if any(results.values()) else set()
    wide_rows = []
    for seed in sorted(common_seeds):
        basins_per_cond = [set(results[c][seed].keys()) for c in results if results[c]]
        common_basins = set.intersection(*basins_per_cond)
        for b in sorted(common_basins):
            row = {"seed": seed, "basin": b}
            for c, r in results.items():
                if r and seed in r and b in r[seed]:
                    row[c] = r[seed][b]
            if "A_baseline" in row and "B_topology_features" in row:
                row["B_minus_A"] = row["B_topology_features"] - row["A_baseline"]
            if "A_baseline" in row and "C_graph_messages" in row:
                row["C_minus_A"] = row["C_graph_messages"] - row["A_baseline"]
            if "B_topology_features" in row and "C_graph_messages" in row:
                row["C_minus_B"] = row["C_graph_messages"] - row["B_topology_features"]
            wide_rows.append(row)
    wide_df = pd.DataFrame(wide_rows)
    wide_df.to_csv(OUT_DIR / "per_basin_deltas.csv", index=False)

    # Per-basin delta summaries
    delta_summary = {}
    for delta_col in ["B_minus_A", "C_minus_A", "C_minus_B"]:
        if delta_col not in wide_df.columns:
            continue
        v = wide_df[delta_col].dropna()
        if len(v) == 0:
            continue
        delta_summary[delta_col] = {
            "n": int(len(v)),
            "median": float(v.median()),
            "mean": float(v.mean()),
            "std": float(v.std()),
            "n_positive": int((v > 0).sum()),
            "n_strongly_positive": int((v > 0.05).sum()),
            "n_strongly_negative": int((v < -0.05).sum()),
            "p25": float(v.quantile(0.25)),
            "p75": float(v.quantile(0.75)),
        }
    summary["per_basin_deltas"] = delta_summary

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot delta distributions
    delta_cols_present = [c for c in ["B_minus_A", "C_minus_A", "C_minus_B"]
                            if c in wide_df.columns]
    if delta_cols_present:
        fig, axes = plt.subplots(1, len(delta_cols_present),
                                   figsize=(4.5 * len(delta_cols_present), 4),
                                   squeeze=False)
        for ax, col in zip(axes[0], delta_cols_present):
            v = wide_df[col].dropna()
            ax.hist(v, bins=40, color="C0", alpha=0.7, edgecolor="white", lw=0.4)
            ax.axvline(0, color="k", lw=0.7, ls="--")
            ax.axvline(v.median(), color="C3", lw=1.5, label=f"median {v.median():+.3f}")
            ax.set_title(f"{col}\nn={len(v)}, mean {v.mean():+.3f}, std {v.std():.3f}")
            ax.set_xlabel("Δ NSE")
            ax.set_ylabel("# basins")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=9)
        fig.suptitle(f"Per-basin ΔNSE distributions (Component 0, "
                      f"{len(common_seeds)} seed(s))", fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "delta_distributions.png", dpi=140)
        plt.close(fig)

    # Depth-stratified plot (if depth file exists)
    if DEPTH_FILE.exists():
        depth_df = pd.read_csv(DEPTH_FILE, dtype={"basin": str})
        wide_with_depth = wide_df.merge(depth_df[["basin", "depth"]], on="basin", how="left")
        depths = sorted(wide_with_depth["depth"].dropna().unique())
        cond_cols_present = [c for c in ["A_baseline", "B_topology_features", "C_graph_messages"]
                              if c in wide_df.columns]
        if cond_cols_present and depths:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
            depth_summary_rows = []
            for cond in cond_cols_present:
                medians = []
                for d in depths:
                    sub = wide_with_depth[wide_with_depth["depth"] == d][cond].dropna()
                    medians.append(sub.median() if len(sub) else np.nan)
                    depth_summary_rows.append({
                        "condition": cond, "depth": int(d),
                        "n_basins": int(len(sub)),
                        "median_NSE": float(sub.median()) if len(sub) else None,
                    })
                ax.plot(depths, medians, "o-", lw=1.6, label=cond)
            ax.set_xlabel("graph depth (0 = headwater)")
            ax.set_ylabel("median NSE across basins at this depth")
            ax.set_title("Depth-stratified median NSE — Component 0")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT_DIR / "nse_by_depth.png", dpi=140)
            plt.close(fig)
            pd.DataFrame(depth_summary_rows).to_csv(
                OUT_DIR / "depth_stratified.csv", index=False)

    # Human-readable summary
    lines = []
    lines.append("=" * 70)
    lines.append("A/B/C scaled comparison — Component 0 (183 basins)")
    lines.append("=" * 70)
    lines.append("")
    for cond, s in [("A_baseline", "A_baseline"),
                      ("B_topology_features", "B_topology_features"),
                      ("C_graph_messages", "C_graph_messages")]:
        if s in summary:
            row = summary[s]
            lines.append(f"{cond}:  n_seeds={row['n_seeds']}  "
                          f"median NSE = {row['cross_seed_median']:.4f}  "
                          f"(±{row['cross_seed_std']:.4f} across seeds)")
        else:
            lines.append(f"{cond}:  not yet run")
    lines.append("")
    if delta_summary:
        lines.append("Per-basin ΔNSE (across all common seeds × basins):")
        for col, d in delta_summary.items():
            lines.append(
                f"  {col:<12}  n={d['n']:>4}  "
                f"median {d['median']:+.4f}  mean {d['mean']:+.4f}  "
                f"std {d['std']:.3f}  "
                f"+/0/-: {d['n_strongly_positive']}/"
                f"{d['n']-d['n_strongly_positive']-d['n_strongly_negative']}/"
                f"{d['n_strongly_negative']}")
    lines.append("")
    lines.append(f"Outputs:")
    lines.append(f"  {OUT_DIR/'summary.json'}")
    lines.append(f"  {OUT_DIR/'per_basin_long.csv'}")
    lines.append(f"  {OUT_DIR/'per_basin_deltas.csv'}")
    lines.append(f"  {OUT_DIR/'delta_distributions.png'}")
    if DEPTH_FILE.exists():
        lines.append(f"  {OUT_DIR/'nse_by_depth.png'}")
    summary_text = "\n".join(lines)
    with open(OUT_DIR / "summary_table.txt", "w") as f:
        f.write(summary_text)
    print(summary_text)


if __name__ == "__main__":
    main()
