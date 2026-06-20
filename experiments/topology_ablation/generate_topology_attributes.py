"""Generate network-position topology features as a NeuralHydrology static-attributes file.

KEY INSIGHT (2026-06-20): NeuralHydrology auto-loads every
`camels_attributes_v2.0/camels_*.txt` file (semicolon-separated, gauge_id index)
and concatenates it into the static-attribute table. So topology features written
in that format become first-class static attributes selectable via `static_attributes:`
in any config — and run on STOCK NH cudalstm with zero custom model code.

This dissolves the architecture/undertraining confound that plagued the custom
DirectedGraphLSTM: the topology-feature question (does network position help?) is
now a pure config-level ablation on the well-tuned NH trainer.

Features (computed on the full inferred river network, raw values — NH applies its
own training-period standardization):
  graph_depth            longest path from any headwater to the basin (FIXED: longest,
                         not shortest — the old compute_topology_features used shortest)
  n_upstream             transitive upstream basin count (absolute)
  total_upstream_area    sum of all upstream basins' drainage areas (km^2) — the
                         physically load-bearing feature; discharge scales with
                         contributing area
  in_degree              number of immediate upstream parents
  frac_upstream_area     total_upstream_area / own_area (dimensionless magnitude)

Written for ALL CAMELS basins (isolated basins get zeros) so any basin subset runs.

Output: datasets/camels_us/camels_attributes_v2.0/camels_topology.txt

Usage:
    python experiments/topology_ablation/generate_topology_attributes.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).parent.parent.parent
EDGE_FILE = ROOT / "topology_analysis/phase1_network_discovery/outputs/full_network_edges.csv"
ATTR_DIR = ROOT / "datasets/camels_us/camels_attributes_v2.0"
TOPO_TXT = ATTR_DIR / "camels_topo.txt"
OUT_FILE = ATTR_DIR / "camels_topology.txt"


def longest_depth(G, node, memo):
    """Longest path from any headwater (in-degree 0 ancestor) to `node`."""
    if node in memo:
        return memo[node]
    preds = list(G.predecessors(node))
    if not preds:
        memo[node] = 0
        return 0
    d = 1 + max(longest_depth(G, p, memo) for p in preds)
    memo[node] = d
    return d


def main():
    edges = pd.read_csv(EDGE_FILE, dtype={"parent_id": str, "child_id": str})
    # Own-area from the CAMELS topo attributes
    topo = pd.read_csv(TOPO_TXT, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
    own_area = topo["area_gages2"].to_dict()

    # All CAMELS basins (so every subset has topology attributes; isolated -> zeros)
    all_basins = sorted(topo.index.tolist())

    G = nx.DiGraph()
    for b in all_basins:
        G.add_node(b)
    for _, r in edges.iterrows():
        if r["parent_id"] in own_area and r["child_id"] in own_area:
            G.add_edge(r["parent_id"], r["child_id"])

    memo = {}
    rows = []
    for b in all_basins:
        ancestors = nx.ancestors(G, b) if b in G else set()
        up_area = sum(float(own_area.get(a, 0.0)) for a in ancestors)
        ob = float(own_area.get(b, 1.0)) or 1.0
        rows.append({
            "gauge_id": b,
            "graph_depth": longest_depth(G, b, memo),
            "n_upstream": len(ancestors),
            "total_upstream_area": round(up_area, 3),
            "in_degree": G.in_degree(b) if b in G else 0,
            "frac_upstream_area": round(up_area / ob, 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, sep=";", index=False)

    connected = (df["n_upstream"] > 0).sum()
    print(f"Wrote {len(df)} basins to {OUT_FILE}")
    print(f"  {connected} basins have >=1 upstream; {len(df) - connected} isolated (zeros).")
    print(f"  graph_depth range: {df['graph_depth'].min()}-{df['graph_depth'].max()}")
    print(f"  total_upstream_area: median {df[df['n_upstream']>0]['total_upstream_area'].median():.0f} km^2 "
          f"(connected basins)")
    print("\nUse in a config via:")
    print("  static_attributes:")
    print("    - graph_depth")
    print("    - n_upstream")
    print("    - total_upstream_area")
    print("    - in_degree")
    print("    - frac_upstream_area")


if __name__ == "__main__":
    main()
