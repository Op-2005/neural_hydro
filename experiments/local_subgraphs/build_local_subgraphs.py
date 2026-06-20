"""Build locally-coherent basin subgraphs via a shortest-path walker.

Implements the professor's prescription (meeting 2026-05-13):
  "make the basin graph with distances and then use a shortest path walker,
   randomly choose a node and trim within shortest path distance — guaranteed
   to get a local geography and get local seeding — use that as a standard."

Method
------
1. Build an undirected graph over Component-0 basins, weighted by edge distance_km.
2. Pick a seed node (deterministically, from a pre-committed seed list — NOT
   re-randomized each run, so the subgraphs are a fixed, citable standard).
3. Collect all basins within `radius_km` shortest-path distance of the seed.
4. Keep the subgraph if it has >= min_size basins and >= 2 graph depths.

The deterministic seed list makes this reproducible and pre-registered: the
5 subgraphs are fixed once and reused across every model/architecture revision,
so the loss-distribution invariant is comparable run-to-run.

Outputs (to experiments/local_subgraphs/basin_lists/):
  <name>_basins.txt   — one basin id per line
  <name>_edges.csv    — the induced directed edges (subset of component0_edges)
  subgraph_manifest.csv — summary table of all subgraphs

Usage:
    python experiments/local_subgraphs/build_local_subgraphs.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).parent.parent.parent
EDGE_FILE = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_edges.csv"
DEPTH_FILE = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_depth.csv"
OUT_DIR = Path(__file__).parent / "basin_lists"

# ---------------------------------------------------------------------------
# Pre-committed walker seeds. Each is (name, seed_basin, radius_km).
# Seeds chosen to land in distinct HUC regions and produce subgraphs in the
# 15-40 basin range. These are FIXED — the standard test set. Do not
# re-randomize between runs.
# ---------------------------------------------------------------------------
WALKER_SEEDS = [
    # name,            seed_basin,  radius_km
    ("sg_midatlantic", "01594950",  120),   # HUC 02 root, large system
    ("sg_ohio",        "03026500",  120),   # HUC 05 Ohio
    ("sg_tennessee",   "03455500",  120),   # HUC 06 Tennessee
    ("sg_southeast",   "02055100",  120),   # HUC 02/03 boundary
    ("sg_northeast",   "01516500",  120),   # HUC 02/04/05
]

MIN_SIZE = 12
MIN_DEPTHS = 2


def build_undirected_distance_graph(edges_df):
    """Undirected graph weighted by edge distance_km (for the walker)."""
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["parent_id"], row["child_id"],
                   weight=float(row["distance_km"]))
    return G


def build_directed_graph(edges_df):
    """Directed parent->child graph (for inducing the subgraph + depth)."""
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["parent_id"], row["child_id"])
    return G


def walker_subgraph(undirected_G, seed_basin, radius_km):
    """All basins within radius_km shortest-path distance of seed_basin."""
    if seed_basin not in undirected_G:
        return set()
    lengths = nx.single_source_dijkstra_path_length(
        undirected_G, seed_basin, cutoff=radius_km, weight="weight")
    return set(lengths.keys())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    edges_df = pd.read_csv(EDGE_FILE, dtype={"parent_id": str, "child_id": str})
    depth_df = pd.read_csv(DEPTH_FILE, dtype={"basin": str, "huc_02": str}).set_index("basin")

    undirected_G = build_undirected_distance_graph(edges_df)
    directed_G = build_directed_graph(edges_df)

    manifest_rows = []
    for name, seed_basin, radius_km in WALKER_SEEDS:
        members = walker_subgraph(undirected_G, seed_basin, radius_km)
        members = {b for b in members if b in depth_df.index}

        if len(members) < MIN_SIZE:
            print(f"[skip] {name}: only {len(members)} basins (< {MIN_SIZE})")
            continue

        sub_depths = depth_df.loc[depth_df.index.isin(members), "depth"]
        if sub_depths.nunique() < MIN_DEPTHS:
            print(f"[skip] {name}: only {sub_depths.nunique()} depth(s) (< {MIN_DEPTHS})")
            continue

        # Induced directed edges among members
        induced = edges_df[edges_df["parent_id"].isin(members)
                            & edges_df["child_id"].isin(members)].copy()

        # Write basin list (sorted for determinism)
        basins_sorted = sorted(members)
        with open(OUT_DIR / f"{name}_basins.txt", "w") as f:
            f.write("\n".join(basins_sorted) + "\n")
        induced.to_csv(OUT_DIR / f"{name}_edges.csv", index=False)

        hucs = sorted(depth_df.loc[depth_df.index.isin(members), "huc_02"].unique().tolist())
        manifest_rows.append({
            "name": name,
            "seed_basin": seed_basin,
            "radius_km": radius_km,
            "n_basins": len(members),
            "n_edges": len(induced),
            "depths": ",".join(str(d) for d in sorted(sub_depths.unique())),
            "n_depths": int(sub_depths.nunique()),
            "hucs": ",".join(hucs),
            "n_hucs": len(hucs),
        })
        print(f"[ok]   {name}: {len(members)} basins, {len(induced)} edges, "
               f"depths {sorted(sub_depths.unique())}, HUCs {hucs}")

    # Also copy the historical Texas pilot as a climate-coherent anchor
    pilot_src = ROOT / "experiments/basin_lists/study_network_basins.txt"
    pilot_edges = ROOT / "topology_analysis/phase1_network_discovery/outputs/study_network_edges.csv"
    if pilot_src.exists():
        pilot_basins = [l.strip() for l in open(pilot_src) if l.strip()]
        with open(OUT_DIR / "sg_texas_pilot_basins.txt", "w") as f:
            f.write("\n".join(pilot_basins) + "\n")
        if pilot_edges.exists():
            pd.read_csv(pilot_edges, dtype=str).to_csv(
                OUT_DIR / "sg_texas_pilot_edges.csv", index=False)
        manifest_rows.append({
            "name": "sg_texas_pilot", "seed_basin": "(historical)",
            "radius_km": "n/a", "n_basins": len(pilot_basins),
            "n_edges": sum(1 for _ in open(pilot_edges)) - 1 if pilot_edges.exists() else 0,
            "depths": "0,1,2,3", "n_depths": 4, "hucs": "12", "n_hucs": 1,
        })
        print(f"[ok]   sg_texas_pilot: {len(pilot_basins)} basins (HUC 12 climate-coherent anchor)")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "subgraph_manifest.csv", index=False)
    print(f"\nWrote {len(manifest)} subgraphs + manifest to {OUT_DIR}")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
