"""Extract Component 0 (183-basin eastern-US network) from the full CAMELS topology.

Uses the already-computed `full_network_edges.csv` produced by discover_network.py.
This is the minimum-viable-statistical-power target per HYPOTHESIS.md Amendment 1.

Outputs:
  * topology_analysis/phase1_network_discovery/outputs/component0_basins.txt
  * topology_analysis/phase1_network_discovery/outputs/component0_edges.csv
  * topology_analysis/phase1_network_discovery/outputs/component0_summary.txt
  * topology_analysis/phase1_network_discovery/outputs/component0_depth.csv
"""
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
OUT = Path(__file__).parent / "outputs"
EDGES_FULL = OUT / "full_network_edges.csv"
TOPO_FILE = ROOT / "datasets/camels_us/camels_attributes_v2.0/camels_topo.txt"
NAME_FILE = ROOT / "datasets/camels_us/camels_attributes_v2.0/camels_name.txt"


def main():
    edges = pd.read_csv(EDGES_FULL, dtype={"parent_id": str, "child_id": str})
    topo = pd.read_csv(TOPO_FILE, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
    names = pd.read_csv(NAME_FILE, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")

    G = nx.DiGraph()
    for _, r in edges.iterrows():
        G.add_edge(r["parent_id"], r["child_id"],
                    area_ratio=r["area_ratio"],
                    elev_diff_m=r["elev_diff_m"],
                    distance_km=r["distance_km"])

    components = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    print(f"Total weakly-connected components (>=2 nodes): "
          f"{sum(1 for c in components if len(c) >= 2)}")

    comp0 = components[0]
    sub = G.subgraph(comp0).copy()
    roots = [n for n in sub if sub.in_degree(n) == 0]
    leaves = [n for n in sub if sub.out_degree(n) == 0]

    max_depth = 0
    depth_map = {n: 0 for n in sub}
    for root in roots:
        lengths = nx.single_source_shortest_path_length(sub, root)
        for node, d in lengths.items():
            depth_map[node] = max(depth_map[node], d)
            max_depth = max(max_depth, d)

    by_depth = pd.Series(depth_map).value_counts().sort_index()

    print(f"\nComponent 0:")
    print(f"  nodes:     {sub.number_of_nodes()}")
    print(f"  edges:     {sub.number_of_edges()}")
    print(f"  roots:     {len(roots)}")
    print(f"  leaves:    {len(leaves)}")
    print(f"  max depth: {max_depth}")
    hucs = sorted(set(str(names.loc[n, "huc_02"]).zfill(2) for n in sub if n in names.index))
    print(f"  HUC regions: {', '.join(hucs)}")
    print(f"\n  basins per depth:")
    for d, n in by_depth.items():
        print(f"    depth {d}: {n:3d} basins")

    # Data-completeness check against maurer forcings + usgs streamflow
    missing = []
    for bid in sub:
        sf = list((ROOT / "datasets/camels_us/usgs_streamflow").glob(f"**/{bid}_streamflow_qc.txt"))
        mf = list((ROOT / "datasets/camels_us/basin_mean_forcing/maurer").glob(f"**/{bid}_*_forcing_leap.txt"))
        if not sf or not mf:
            missing.append(bid)
    print(f"\n  {sub.number_of_nodes() - len(missing)}/{sub.number_of_nodes()} basins "
          f"have maurer + streamflow; missing: {len(missing)}")

    usable = [b for b in sub if b not in missing]
    usable_G = sub.subgraph(usable).copy()
    usable_edges = edges[edges["parent_id"].isin(usable) & edges["child_id"].isin(usable)]

    # Recompute depth over usable subgraph
    usable_roots = [n for n in usable_G if usable_G.in_degree(n) == 0]
    usable_depth = {n: 0 for n in usable_G}
    for r in usable_roots:
        for node, d in nx.single_source_shortest_path_length(usable_G, r).items():
            usable_depth[node] = max(usable_depth[node], d)
    usable_max_depth = max(usable_depth.values())

    print(f"\n  After data-completeness filter: {len(usable)} basins, "
          f"{usable_G.number_of_edges()} edges, max_depth={usable_max_depth}")
    print(f"  Basins per depth (usable):")
    for d, n in pd.Series(usable_depth).value_counts().sort_index().items():
        print(f"    depth {d}: {n:3d} basins")

    # Save
    with open(OUT / "component0_basins.txt", "w") as f:
        for b in sorted(usable):
            f.write(f"{b}\n")

    usable_edges.to_csv(OUT / "component0_edges.csv", index=False)

    depth_df = pd.DataFrame([
        {"basin": b,
         "depth": usable_depth[b],
         "role": "headwater" if usable_G.in_degree(b) == 0
                 else ("outlet" if usable_G.out_degree(b) == 0 else "interior"),
         "n_upstream": usable_G.in_degree(b),
         "n_downstream": usable_G.out_degree(b),
         "area_km2": float(topo.loc[b, "area_gages2"]) if b in topo.index else np.nan,
         "elev_m": float(topo.loc[b, "elev_mean"]) if b in topo.index else np.nan,
         "huc_02": str(names.loc[b, "huc_02"]).zfill(2) if b in names.index else "??",
         }
        for b in sorted(usable)
    ])
    depth_df.to_csv(OUT / "component0_depth.csv", index=False)

    with open(OUT / "component0_summary.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Component 0 (minimum-viable-statistical-power test network)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Source: full_network_edges.csv (heuristic inference, 671 basins)\n")
        f.write(f"Params: max_dist=150km, area_ratio>=1.5, elev_decreasing=True\n\n")
        f.write(f"Basins (after data-completeness filter): {len(usable)}\n")
        f.write(f"Edges:     {usable_G.number_of_edges()}\n")
        f.write(f"Roots:     {len(usable_roots)}\n")
        f.write(f"Max depth: {usable_max_depth}\n")
        f.write(f"HUC regions: {', '.join(hucs)}\n\n")
        f.write("Basins per depth:\n")
        for d, n in pd.Series(usable_depth).value_counts().sort_index().items():
            f.write(f"  depth {d}: {n:3d}\n")

    print(f"\nWrote:")
    print(f"  {OUT/'component0_basins.txt'}   ({len(usable)} basins)")
    print(f"  {OUT/'component0_edges.csv'}    ({usable_G.number_of_edges()} edges)")
    print(f"  {OUT/'component0_depth.csv'}")
    print(f"  {OUT/'component0_summary.txt'}")


if __name__ == "__main__":
    main()
