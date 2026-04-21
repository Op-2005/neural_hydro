"""Compare Directed Graph-LSTM variants vs. Baseline LSTM on the study network.

Supports comparing one baseline against multiple graph-LSTM runs at once.

Usage:
    # Strong baseline vs headline graph result
    python experiments/compare_results.py \\
        --baseline runs/05_lstm_23basin_strong_baseline/test/model_epoch030/test_metrics.csv \\
        --baseline-label "Strong LSTM" \\
        --graph runs/06_graph_edge_warm_full/test_metrics.csv:Edge+Warm

    # Multiple graph variants with labels
    python experiments/compare_results.py \\
        --baseline runs/05_lstm_23basin_strong_baseline/test/model_epoch030/test_metrics.csv \\
        --baseline-label "Strong LSTM" \\
        --graph runs/06_graph_edge_warm_full/test_metrics.csv:Headline \\
                runs/07_graph_edge_frozen/test_metrics.csv:Frozen \\
                runs/11_graph_edge_pruned_edges/test_metrics.csv:Pruned
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

EDGE_FILE = Path("topology_analysis/phase1_network_discovery/outputs/study_network_edges.csv")
SUMMARY_FILE = Path("topology_analysis/phase1_network_discovery/outputs/study_network_summary.txt")


def load_basin_info():
    """Parse basin depth and role from study_network_summary.txt."""
    info = {}
    edges = pd.read_csv(EDGE_FILE, dtype={"parent_id": str, "child_id": str})

    parents = {}
    for _, row in edges.iterrows():
        parents.setdefault(row["child_id"], []).append(row["parent_id"])

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
                bid = parts[0]
                info[bid] = {
                    "area_km2": float(parts[1]),
                    "elev_m": float(parts[2]),
                    "depth": int(parts[3]),
                    "role": parts[4],
                    "n_upstream": len(parents.get(bid, [])),
                }
        if line.startswith("Edge list:"):
            break

    return info


def resolve_glob(pattern):
    """Expand glob pattern to a single existing file."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for: {pattern}")
    return Path(matches[-1])   # take the most recent match (sorted)


def load_metrics(path_spec):
    """Load metrics csv from a path. Supports 'path:label' format."""
    if ":" in path_spec and not path_spec.startswith("/"):
        path, label = path_spec.rsplit(":", 1)
    else:
        path, label = path_spec, None

    resolved = resolve_glob(path)
    df = pd.read_csv(resolved, dtype={"basin": str}).set_index("basin")

    if label is None:
        label = resolved.parent.parent.name if "test" in str(resolved) else resolved.parent.name
    return label, df


def print_table(baseline_label, baseline, graph_labels, graph_dfs, basin_info):
    """Print the comparison table for baseline vs one or more graph variants."""
    common = sorted(set(baseline.index).intersection(*(g.index for g in graph_dfs)))

    # Header
    col_width = 11
    header_cols = [("Basin", 10), ("Role", 10), ("Depth", 5), ("Upstream", 8),
                   (baseline_label[:col_width], col_width)]
    for label in graph_labels:
        header_cols.append((label[:col_width], col_width))
    header_cols.append(("Best Δ", 8))

    fmt = "  ".join(f"{{:>{w}}}" for _, w in header_cols)
    print()
    print("=" * sum(w + 2 for _, w in header_cols))
    title = f"{baseline_label}  vs  " + ", ".join(graph_labels)
    print(title)
    print("=" * sum(w + 2 for _, w in header_cols))
    print(fmt.format(*[c for c, _ in header_cols]))
    print(fmt.format(*["-" * w for _, w in header_cols]))

    rows = []
    for bid in common:
        info = basin_info.get(bid, {})
        b_nse = baseline.loc[bid, "NSE"]
        g_nses = [g.loc[bid, "NSE"] for g in graph_dfs]
        deltas = [g - b_nse for g in g_nses]
        best_delta = max(deltas)
        rows.append({
            "basin": bid,
            "role": info.get("role", "?"),
            "depth": info.get("depth", 0),
            "n_upstream": info.get("n_upstream", 0),
            "baseline": b_nse,
            "graphs": g_nses,
            "deltas": deltas,
            "best_delta": best_delta,
        })

    rows.sort(key=lambda r: (r["depth"], -r["best_delta"]))

    for r in rows:
        parts = [r["basin"], r["role"], f"{r['depth']}", f"{r['n_upstream']}",
                 f"{r['baseline']:.3f}"]
        parts.extend(f"{g:.3f}" for g in r["graphs"])
        bd = r["best_delta"]
        parts.append(f"{'+' if bd >= 0 else ''}{bd:.3f}")
        print(fmt.format(*parts))

    print()
    print("-" * sum(w + 2 for _, w in header_cols))

    # Summary by depth
    for depth in sorted(set(r["depth"] for r in rows)):
        drows = [r for r in rows if r["depth"] == depth]
        b_med = np.median([r["baseline"] for r in drows])
        label = "headwater" if depth == 0 else f"depth={depth}"
        parts_summary = [f"  {label:12s} ({len(drows):2d} basins):  base={b_med:+.3f}"]
        for i, glabel in enumerate(graph_labels):
            g_med = np.median([r["graphs"][i] for r in drows])
            d_med = np.median([r["deltas"][i] for r in drows])
            parts_summary.append(f"{glabel[:11]}={g_med:+.3f} (Δ{d_med:+.3f})")
        print("  ".join(parts_summary))

    # Overall
    print()
    all_b = np.median([r["baseline"] for r in rows])
    overall = [f"  Overall ({len(rows)} basins): base={all_b:+.3f}"]
    for i, glabel in enumerate(graph_labels):
        g_med = np.median([r["graphs"][i] for r in rows])
        d_med = np.median([r["deltas"][i] for r in rows])
        overall.append(f"{glabel[:11]}={g_med:+.3f} (Δ{d_med:+.3f})")
    print("  ".join(overall))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True,
                        help="Path or glob to baseline test_metrics.csv")
    parser.add_argument("--baseline-label", type=str, default="Baseline",
                        help="Label for the baseline column")
    parser.add_argument("--graph", type=str, nargs="+", required=True,
                        help="One or more graph metrics: 'path' or 'path:label'")
    args = parser.parse_args()

    basin_info = load_basin_info()

    _, baseline = load_metrics(args.baseline)
    graph_entries = [load_metrics(g) for g in args.graph]
    graph_labels = [g[0] for g in graph_entries]
    graph_dfs = [g[1] for g in graph_entries]

    print_table(args.baseline_label, baseline, graph_labels, graph_dfs, basin_info)


if __name__ == "__main__":
    main()
