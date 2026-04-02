"""Visualize the full CAMELS-US network discovery results.

Produces:
  1. All-basins geographic map with edges colored by component
  2. Component size distribution (bar chart)
  3. Edge diagnostics (distance vs area_ratio scatter)
  4. Study network detail with depth coloring

Usage:
    python topology_analysis/phase1_network_discovery/plot_full_network.py
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "datasets" / "camels_us"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_data():
    topo = pd.read_csv(
        DATA_DIR / "camels_attributes_v2.0" / "camels_topo.txt",
        sep=";", dtype={"gauge_id": str}
    ).set_index("gauge_id")

    edges = pd.read_csv(OUTPUT_DIR / "full_network_edges.csv",
                         dtype={"parent_id": str, "child_id": str})

    study_edges = pd.read_csv(OUTPUT_DIR / "study_network_edges.csv",
                               dtype={"parent_id": str, "child_id": str})

    with open(OUTPUT_DIR / "study_network_basins.txt") as f:
        study_basins = [l.strip() for l in f if l.strip()]

    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row["parent_id"], row["child_id"])

    return topo, edges, study_edges, study_basins, G


def plot_national_map(topo, G):
    """All 671 basins on a map. Nodes in components colored, isolated basins grey."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    # Assign component IDs
    components = list(nx.weakly_connected_components(G))
    components.sort(key=len, reverse=True)
    node_comp = {}
    for i, nodes in enumerate(components):
        for n in nodes:
            node_comp[n] = i

    # All basins
    all_basins = topo.index.tolist()
    lons = topo["gauge_lon"].values
    lats = topo["gauge_lat"].values

    # Draw edges (light, behind everything)
    for u, v in G.edges():
        if u in topo.index and v in topo.index:
            x0, y0 = topo.loc[u, "gauge_lon"], topo.loc[u, "gauge_lat"]
            x1, y1 = topo.loc[v, "gauge_lon"], topo.loc[v, "gauge_lat"]
            ax.plot([x0, x1], [y0, y1], color="#d1d5db", lw=0.3, zorder=1)

    # Isolated basins (not in any edge)
    in_graph = set(G.nodes())
    isolated_mask = np.array([b not in in_graph for b in all_basins])
    ax.scatter(lons[isolated_mask], lats[isolated_mask],
               c="#e5e7eb", s=8, zorder=2, edgecolors="none", alpha=0.6)

    # Connected basins — color top 5 components, rest grey
    cmap = plt.cm.Set1
    for b in all_basins:
        if b not in in_graph:
            continue
        comp_id = node_comp[b]
        if comp_id < 5:
            color = cmap(comp_id / 5)
        else:
            color = "#9ca3af"
        ax.scatter(topo.loc[b, "gauge_lon"], topo.loc[b, "gauge_lat"],
                   c=[color], s=15, zorder=3, edgecolors="none")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=cmap(0 / 5), label=f"Component 0 ({len(components[0])} basins)"),
        mpatches.Patch(facecolor=cmap(1 / 5), label=f"Component 1 ({len(components[1])} basins)"),
        mpatches.Patch(facecolor=cmap(2 / 5), label=f"Component 2 ({len(components[2])} basins)"),
        mpatches.Patch(facecolor=cmap(3 / 5), label=f"Component 3 — study network ({len(components[3])} basins)"),
        mpatches.Patch(facecolor=cmap(4 / 5), label=f"Component 4 ({len(components[4])} basins)"),
        mpatches.Patch(facecolor="#9ca3af", label=f"Other components"),
        mpatches.Patch(facecolor="#e5e7eb", label=f"Isolated basins ({sum(isolated_mask)})"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8, framealpha=0.9)

    ax.set_title(f"CAMELS-US Topology Inference — {len(topo)} basins, "
                 f"{G.number_of_edges()} edges, {len(components)} components",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "national_network_map.png", dpi=150, bbox_inches="tight")
    print(f"Saved national_network_map.png")
    plt.close(fig)


def plot_component_distribution(G):
    """Bar chart of component sizes."""
    components = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    sizes = [len(c) for c in components]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of top 15 components
    top_n = min(15, len(sizes))
    ax1.bar(range(top_n), sizes[:top_n], color="#3b82f6", edgecolor="white")
    ax1.set_xlabel("Component rank")
    ax1.set_ylabel("Number of basins")
    ax1.set_title("Top 15 Connected Components by Size")
    ax1.set_xticks(range(top_n))

    # Highlight component 3 (study network)
    if top_n > 3:
        ax1.bar(3, sizes[3], color="#ef4444", edgecolor="white")
        ax1.annotate("Study\nnetwork", (3, sizes[3]), textcoords="offset points",
                      xytext=(0, 8), ha="center", fontsize=8, color="#ef4444",
                      fontweight="bold")

    # Histogram of all component sizes
    ax2.hist(sizes, bins=range(1, max(sizes) + 2), color="#6b7280", edgecolor="white",
             align="left")
    ax2.set_xlabel("Component size (basins)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Component Size Distribution ({len(components)} total)")
    ax2.set_yscale("log")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "component_distribution.png", dpi=150, bbox_inches="tight")
    print(f"Saved component_distribution.png")
    plt.close(fig)


def plot_edge_diagnostics(edges):
    """Scatter: distance vs area_ratio, colored by elevation difference."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    sc = ax.scatter(edges["distance_km"], edges["area_ratio"],
                     c=edges["elev_diff_m"], cmap="viridis", s=8, alpha=0.6,
                     edgecolors="none")

    cbar = fig.colorbar(sc, ax=ax, label="Elevation difference (m)")
    ax.set_xlabel("Distance between basins (km)")
    ax.set_ylabel("Area ratio (child / parent)")
    ax.set_title(f"Edge Diagnostics — {len(edges)} inferred edges\n"
                 f"(distance <= 150 km, area ratio >= 1.5, child lower than parent)")
    ax.axhline(y=1.5, color="#ef4444", lw=0.8, ls="--", alpha=0.5, label="area_ratio threshold")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "edge_diagnostics.png", dpi=150, bbox_inches="tight")
    print(f"Saved edge_diagnostics.png")
    plt.close(fig)


def plot_study_network_depth(topo, study_basins, study_edges):
    """Study network colored by depth (0=headwater to 3=deep outlet)."""
    G = nx.DiGraph()
    G.add_nodes_from(study_basins)
    for _, row in study_edges.iterrows():
        G.add_edge(row["parent_id"], row["child_id"])

    roots = [n for n in G if G.in_degree(n) == 0]
    depth = {n: 0 for n in study_basins}
    for root in roots:
        for node, d in nx.single_source_shortest_path_length(G, root).items():
            depth[node] = max(depth[node], d)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Edges
    for u, v in G.edges():
        x0, y0 = topo.loc[u, "gauge_lon"], topo.loc[u, "gauge_lat"]
        x1, y1 = topo.loc[v, "gauge_lon"], topo.loc[v, "gauge_lat"]
        ax.annotate("",
                     xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="-|>", color="#9ca3af",
                                     lw=1.0, shrinkA=6, shrinkB=6,
                                     connectionstyle="arc3,rad=0.1"))

    # Nodes colored by depth
    max_depth = max(depth.values())
    cmap = plt.cm.RdYlBu_r
    norm = Normalize(vmin=0, vmax=max_depth)

    for bid in study_basins:
        x = topo.loc[bid, "gauge_lon"]
        y = topo.loc[bid, "gauge_lat"]
        area = topo.loc[bid, "area_gages2"]
        size = 60 + area * 0.03
        color = cmap(norm(depth[bid]))
        ax.scatter(x, y, c=[color], s=size, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(f"{bid[-5:]}\nd={depth[bid]}", (x, y),
                     textcoords="offset points", xytext=(6, 6), fontsize=7,
                     color="#374151")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label="Depth (hops from headwater)", ticks=range(max_depth + 1))

    ax.set_title(f"Study Network — Basin Depth (0 = headwater, {max_depth} = deepest outlet)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "study_network_depth.png", dpi=150, bbox_inches="tight")
    print(f"Saved study_network_depth.png")
    plt.close(fig)


def main():
    topo, edges, study_edges, study_basins, G = load_data()

    print(f"Full network: {len(topo)} basins, {G.number_of_edges()} edges")
    print(f"Study network: {len(study_basins)} basins, {len(study_edges)} edges")
    print()

    plot_national_map(topo, G)
    plot_component_distribution(G)
    plot_edge_diagnostics(edges)
    plot_study_network_depth(topo, study_basins, study_edges)


if __name__ == "__main__":
    main()
