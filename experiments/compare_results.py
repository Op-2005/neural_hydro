"""Compare Directed Graph-LSTM vs. Baseline LSTM on the study network.

Usage:
    /Applications/anaconda3/envs/nh/bin/python experiments/compare_results.py \\
        --baseline runs/lstm_study_network_1304_222043/test/model_epoch030/test_metrics.csv \\
        --graph runs/graph_lstm_study_network_*/test_metrics.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EDGE_FILE = Path("topology_analysis/phase1_network_discovery/outputs/study_network_edges.csv")
SUMMARY_FILE = Path("topology_analysis/phase1_network_discovery/outputs/study_network_summary.txt")


def load_basin_info():
    """Parse basin depth and role from study_network_summary.txt."""
    info = {}
    edges = pd.read_csv(EDGE_FILE, dtype={"parent_id": str, "child_id": str})

    # Build graph to compute in-degree / out-degree
    children = {}
    parents = {}
    for _, row in edges.iterrows():
        children.setdefault(row["parent_id"], []).append(row["child_id"])
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline test_metrics.csv")
    parser.add_argument("--graph", type=str, required=True, help="Path to graph-lstm test_metrics.csv")
    args = parser.parse_args()

    baseline = pd.read_csv(args.baseline, dtype={"basin": str}).set_index("basin")
    graph = pd.read_csv(args.graph, dtype={"basin": str}).set_index("basin")
    basin_info = load_basin_info()

    # Merge
    common = sorted(set(baseline.index) & set(graph.index))

    print()
    print("=" * 90)
    print("Directed Graph-LSTM vs. Baseline LSTM: Study Network Comparison")
    print("=" * 90)
    print()
    print(f"{'Basin':>10}  {'Role':>10}  {'Depth':>5}  {'Upstream':>8}  "
          f"{'Baseline':>10}  {'GraphLSTM':>10}  {'Delta':>8}")
    print(f"{'-----':>10}  {'----':>10}  {'-----':>5}  {'--------':>8}  "
          f"{'--------':>10}  {'---------':>10}  {'-----':>8}")

    rows = []
    for bid in common:
        info = basin_info.get(bid, {})
        b_nse = baseline.loc[bid, "NSE"]
        g_nse = graph.loc[bid, "NSE"]
        delta = g_nse - b_nse
        rows.append({
            "basin": bid,
            "role": info.get("role", "?"),
            "depth": info.get("depth", 0),
            "n_upstream": info.get("n_upstream", 0),
            "baseline": b_nse,
            "graph": g_nse,
            "delta": delta,
        })

    rows.sort(key=lambda r: (r["depth"], -r["delta"]))

    for r in rows:
        sign = "+" if r["delta"] >= 0 else ""
        print(f"{r['basin']:>10}  {r['role']:>10}  {r['depth']:5d}  {r['n_upstream']:8d}  "
              f"{r['baseline']:10.3f}  {r['graph']:10.3f}  {sign}{r['delta']:7.3f}")

    print()
    print("-" * 90)

    # Summary by depth
    for depth in sorted(set(r["depth"] for r in rows)):
        depth_rows = [r for r in rows if r["depth"] == depth]
        deltas = [r["delta"] for r in depth_rows]
        b_nses = [r["baseline"] for r in depth_rows]
        g_nses = [r["graph"] for r in depth_rows]
        label = "headwater" if depth == 0 else f"depth={depth}"
        expected = "~0, no upstream" if depth == 0 else "positive if graph helps"
        print(f"  {label:12s} ({len(depth_rows):2d} basins): "
              f"baseline median={np.median(b_nses):.3f}  "
              f"graph median={np.median(g_nses):.3f}  "
              f"delta median={np.median(deltas):+.3f}  "
              f"[expected: {expected}]")

    all_b = [r["baseline"] for r in rows]
    all_g = [r["graph"] for r in rows]
    all_d = [r["delta"] for r in rows]
    print()
    print(f"  Overall ({len(rows)} basins): "
          f"baseline median={np.median(all_b):.3f}  "
          f"graph median={np.median(all_g):.3f}  "
          f"delta median={np.median(all_d):+.3f}")
    print()


if __name__ == "__main__":
    main()
