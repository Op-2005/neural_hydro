"""Discover river network topology from all CAMELS-US basins.

Runs heuristic upstream-downstream inference on all 671 CAMELS basins,
identifies connected components, and selects the best study network
for the directed Graph-LSTM experiment.

Usage:
    python topology_analysis/phase1_network_discovery/discover_network.py
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "datasets" / "camels_us"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Inference parameters (tuned for full national dataset)
MAX_DISTANCE_KM = 150
AREA_RATIO_THRESHOLD = 1.5
ELEVATION_MUST_DECREASE = True

# Study network selection criteria
MIN_NETWORK_SIZE = 15
MIN_DEPTH = 3


def load_all_basins() -> pd.DataFrame:
    """Load topographic attributes for all CAMELS-US basins."""
    topo_file = DATA_DIR / "camels_attributes_v2.0" / "camels_topo.txt"
    df = pd.read_csv(topo_file, sep=";", dtype={"gauge_id": str})
    df = df.set_index("gauge_id")
    return df


def haversine_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Compute pairwise haversine distances (km) between all points."""
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2)
    return 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def infer_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Infer all upstream->downstream edges using vectorized heuristics.

    Returns DataFrame with columns:
        parent_id, child_id, parent_area_km2, child_area_km2,
        area_ratio, elev_diff_m, distance_km
    """
    n = len(df)
    ids = df.index.values
    areas = df["area_gages2"].values
    elevs = df["elev_mean"].values
    lats = df["gauge_lat"].values
    lons = df["gauge_lon"].values

    # Pairwise matrices: row=parent, col=child
    area_ratio = areas[None, :] / areas[:, None]       # child_area / parent_area
    elev_diff = elevs[:, None] - elevs[None, :]        # parent_elev - child_elev (positive = child lower)
    dist_matrix = haversine_matrix(lats, lons)

    # Apply all three heuristics
    mask = (
        (area_ratio >= AREA_RATIO_THRESHOLD) &   # child area >= parent * threshold
        (elev_diff > 0) &                         # child is lower
        (dist_matrix <= MAX_DISTANCE_KM) &        # within distance
        ~np.eye(n, dtype=bool)                    # not self
    )

    parent_idx, child_idx = np.where(mask)

    edges = pd.DataFrame({
        "parent_id": ids[parent_idx],
        "child_id": ids[child_idx],
        "parent_area_km2": areas[parent_idx],
        "child_area_km2": areas[child_idx],
        "area_ratio": area_ratio[parent_idx, child_idx],
        "elev_diff_m": elev_diff[parent_idx, child_idx],
        "distance_km": dist_matrix[parent_idx, child_idx],
    })

    return edges.sort_values(["parent_id", "child_id"]).reset_index(drop=True)


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    """Build a directed graph from the edge DataFrame."""
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row["parent_id"], row["child_id"],
                    area_ratio=row["area_ratio"],
                    elev_diff_m=row["elev_diff_m"],
                    distance_km=row["distance_km"])
    return G


def analyze_components(G: nx.DiGraph, topo: pd.DataFrame) -> pd.DataFrame:
    """Analyze all weakly connected components with >= 2 nodes."""
    components = []
    for i, nodes in enumerate(
        sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    ):
        if len(nodes) < 2:
            continue
        sub = G.subgraph(nodes)

        # Roots: no incoming edges within the component
        roots = [n for n in sub if sub.in_degree(n) == 0]
        # Leaves: no outgoing edges within the component
        leaves = [n for n in sub if sub.out_degree(n) == 0]

        # Max directed path length (longest root->leaf path)
        max_depth = 0
        for root in roots:
            lengths = nx.single_source_shortest_path_length(sub, root)
            max_depth = max(max_depth, max(lengths.values()))

        # Geographic bounding box
        lats = topo.loc[list(nodes), "gauge_lat"]
        lons = topo.loc[list(nodes), "gauge_lon"]

        # HUC regions (from name file)
        name_file = DATA_DIR / "camels_attributes_v2.0" / "camels_name.txt"
        names = pd.read_csv(name_file, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
        hucs = sorted(set(str(names.loc[n, "huc_02"]).zfill(2) for n in nodes if n in names.index))

        components.append({
            "component_id": i,
            "nodes": len(nodes),
            "edges": sub.number_of_edges(),
            "roots": len(roots),
            "leaves": len(leaves),
            "max_depth": max_depth,
            "huc_regions": ", ".join(hucs),
            "lat_min": lats.min(),
            "lat_max": lats.max(),
            "lon_min": lons.min(),
            "lon_max": lons.max(),
            "node_ids": sorted(nodes),
            "root_ids": sorted(roots),
            "leaf_ids": sorted(leaves),
        })

    return pd.DataFrame(components)


def select_study_network(comp_df: pd.DataFrame, G: nx.DiGraph,
                         topo: pd.DataFrame) -> Tuple[int, str]:
    """Select the best component for the graph experiment.

    Criteria (in order):
        1. Size >= MIN_NETWORK_SIZE (at least 15 nodes)
        2. Depth >= MIN_DEPTH (at least 3 hops)
        3. Geographic coherence (prefer fewer HUC regions)
        4. If multiple qualify, take the largest

    Returns (component_id, reason_string).
    """
    candidates = comp_df.copy()

    # Filter by minimum size
    size_ok = candidates[candidates["nodes"] >= MIN_NETWORK_SIZE]

    if len(size_ok) > 0:
        # Among large-enough components, prefer depth >= MIN_DEPTH
        deep_ok = size_ok[size_ok["max_depth"] >= MIN_DEPTH]
        if len(deep_ok) > 0:
            # Among deep-enough, prefer fewer HUC regions (geographic coherence)
            deep_ok = deep_ok.copy()
            deep_ok["n_hucs"] = deep_ok["huc_regions"].apply(lambda x: len(x.split(", ")))
            deep_ok = deep_ok.sort_values(["n_hucs", "nodes"], ascending=[True, False])
            chosen = deep_ok.iloc[0]
            reason = (f"Meets all criteria: {chosen['nodes']} nodes, "
                      f"depth {chosen['max_depth']}, "
                      f"HUC regions: {chosen['huc_regions']}")
        else:
            # No deep-enough component, take the deepest large one
            chosen = size_ok.sort_values("max_depth", ascending=False).iloc[0]
            reason = (f"Largest depth among size>={MIN_NETWORK_SIZE}: "
                      f"{chosen['nodes']} nodes, depth {chosen['max_depth']}")
    else:
        # No component with >= MIN_NETWORK_SIZE, take the largest overall
        chosen = candidates.iloc[0]
        reason = (f"Largest component (none >= {MIN_NETWORK_SIZE} nodes): "
                  f"{chosen['nodes']} nodes, depth {chosen['max_depth']}")

    return int(chosen["component_id"]), reason


def verify_data_completeness(basin_ids: list) -> list:
    """Check that all basins have maurer forcings and streamflow data."""
    missing = []
    for bid in basin_ids:
        sf = list((DATA_DIR / "usgs_streamflow").glob(f"**/{bid}_streamflow_qc.txt"))
        mf = list((DATA_DIR / "basin_mean_forcing" / "maurer").glob(f"**/{bid}_*_forcing_leap.txt"))
        if not sf or not mf:
            missing.append(bid)
    return missing


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CAMELS-US River Network Discovery")
    print("=" * 60)

    # Load all basins
    topo = load_all_basins()
    print(f"\nLoaded {len(topo)} basins from CAMELS-US")
    print(f"  Area range: {topo.area_gages2.min():.1f} - {topo.area_gages2.max():.1f} km²")
    print(f"  Elevation range: {topo.elev_mean.min():.1f} - {topo.elev_mean.max():.1f} m")

    # Infer edges
    print(f"\nInference parameters:")
    print(f"  max_distance_km = {MAX_DISTANCE_KM}")
    print(f"  area_ratio_threshold = {AREA_RATIO_THRESHOLD}")
    print(f"  elevation_must_decrease = {ELEVATION_MUST_DECREASE}")

    edges = infer_edges(topo)
    print(f"\nFound {len(edges)} directed edges")

    # Save full edge list
    edges.to_csv(OUTPUT_DIR / "full_network_edges.csv", index=False)
    print(f"Saved full edge list to outputs/full_network_edges.csv")

    # Build graph and analyze components
    G = build_graph(edges)
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    comp_df = analyze_components(G, topo)
    print(f"Connected components with >= 2 nodes: {len(comp_df)}")

    # Print top 10 components
    print(f"\nTop 10 components by size:")
    print(f"  {'Comp':>4}  {'Nodes':>5}  {'Edges':>5}  {'Roots':>5}  {'Leaves':>6}  "
          f"{'Depth':>5}  HUC regions")
    print(f"  {'----':>4}  {'-----':>5}  {'-----':>5}  {'-----':>5}  {'------':>6}  "
          f"{'-----':>5}  -----------")
    for _, row in comp_df.head(10).iterrows():
        print(f"  {row['component_id']:4d}  {row['nodes']:5d}  {row['edges']:5d}  "
              f"{row['roots']:5d}  {row['leaves']:6d}  {row['max_depth']:5d}  "
              f"{row['huc_regions']}")

    # Select study network
    print(f"\n{'=' * 60}")
    print("Study Network Selection")
    print("=" * 60)

    comp_id, reason = select_study_network(comp_df, G, topo)
    selected = comp_df[comp_df["component_id"] == comp_id].iloc[0]
    study_basins = selected["node_ids"]

    print(f"\nSelected component {comp_id}: {reason}")
    print(f"  Basins: {len(study_basins)}")
    print(f"  Edges: {selected['edges']}")
    print(f"  Roots: {selected['roots']} — {selected['root_ids']}")
    print(f"  Leaves: {selected['leaves']} — {selected['leaf_ids']}")
    print(f"  Max depth: {selected['max_depth']}")
    print(f"  Bounding box: ({selected['lat_min']:.2f}, {selected['lon_min']:.2f}) "
          f"to ({selected['lat_max']:.2f}, {selected['lon_max']:.2f})")

    # Verify data completeness
    missing = verify_data_completeness(study_basins)
    if missing:
        print(f"\n  WARNING: {len(missing)} basins missing data: {missing}")
    else:
        print(f"\n  All {len(study_basins)} basins have maurer forcings and streamflow data.")

    # Extract study network edges
    study_edges = edges[
        edges["parent_id"].isin(study_basins) & edges["child_id"].isin(study_basins)
    ].copy()

    # Save study network files
    study_edges.to_csv(OUTPUT_DIR / "study_network_edges.csv", index=False)

    with open(OUTPUT_DIR / "study_network_basins.txt", "w") as f:
        for bid in sorted(study_basins):
            f.write(f"{bid}\n")

    # Compute per-basin depth (max distance from any root)
    sub_G = G.subgraph(study_basins).copy()
    roots = [n for n in sub_G if sub_G.in_degree(n) == 0]
    basin_depth = {n: 0 for n in study_basins}
    for root in roots:
        lengths = nx.single_source_shortest_path_length(sub_G, root)
        for node, d in lengths.items():
            basin_depth[node] = max(basin_depth[node], d)

    # Write summary
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("CAMELS-US Study Network Summary")
    summary_lines.append("=" * 70)
    summary_lines.append(f"")
    summary_lines.append(f"Source: Full CAMELS-US topology inference ({len(topo)} basins, {len(edges)} edges)")
    summary_lines.append(f"Parameters: max_dist={MAX_DISTANCE_KM}km, "
                         f"area_ratio>={AREA_RATIO_THRESHOLD}, elev_decreasing=True")
    summary_lines.append(f"")
    summary_lines.append(f"Selected component: {comp_id}")
    summary_lines.append(f"Reason: {reason}")
    summary_lines.append(f"")
    summary_lines.append(f"Basins:     {len(study_basins)}")
    summary_lines.append(f"Edges:      {selected['edges']}")
    summary_lines.append(f"Roots:      {selected['roots']}")
    summary_lines.append(f"Leaves:     {selected['leaves']}")
    summary_lines.append(f"Max depth:  {selected['max_depth']}")
    summary_lines.append(f"HUC regions: {selected['huc_regions']}")
    summary_lines.append(f"")
    summary_lines.append(f"Basin inventory:")
    summary_lines.append(f"  {'Basin':>10}  {'Area_km2':>10}  {'Elev_m':>8}  {'Depth':>5}  Role")
    summary_lines.append(f"  {'-----':>10}  {'--------':>10}  {'------':>8}  {'-----':>5}  ----")

    for bid in sorted(study_basins):
        area = topo.loc[bid, "area_gages2"]
        elev = topo.loc[bid, "elev_mean"]
        depth = basin_depth[bid]
        in_deg = sub_G.in_degree(bid)
        out_deg = sub_G.out_degree(bid)
        if in_deg == 0:
            role = "headwater"
        elif out_deg == 0:
            role = "outlet"
        else:
            role = "interior"
        summary_lines.append(f"  {bid:>10}  {area:10.1f}  {elev:8.1f}  {depth:5d}  {role}")

    summary_lines.append(f"")
    summary_lines.append(f"Edge list:")
    for _, row in study_edges.iterrows():
        summary_lines.append(
            f"  {row['parent_id']} -> {row['child_id']}  "
            f"(area_ratio={row['area_ratio']:.2f}, "
            f"elev_diff={row['elev_diff_m']:.1f}m, "
            f"dist={row['distance_km']:.1f}km)")

    summary_text = "\n".join(summary_lines)
    with open(OUTPUT_DIR / "study_network_summary.txt", "w") as f:
        f.write(summary_text)

    print(f"\nSaved:")
    print(f"  outputs/full_network_edges.csv        ({len(edges)} edges)")
    print(f"  outputs/study_network_edges.csv       ({len(study_edges)} edges)")
    print(f"  outputs/study_network_basins.txt      ({len(study_basins)} basins)")
    print(f"  outputs/study_network_summary.txt")

    return study_basins, study_edges, sub_G, topo, basin_depth


if __name__ == "__main__":
    main()
