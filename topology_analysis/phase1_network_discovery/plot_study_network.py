"""Visualize the selected study network.

Produces two figures:
  1. Geographic map (basin centroids at lat/lon, with directed edges)
  2. Topological DAG layout (tree structure, ignoring geography)

Usage:
    python topology_analysis/phase1_network_discovery/plot_study_network.py
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "datasets" / "camels_us"
OUTPUT_DIR = Path(__file__).parent / "outputs"


def load_study_network():
    """Load study network graph, basin attributes, and per-basin depth."""
    edges = pd.read_csv(OUTPUT_DIR / "study_network_edges.csv", dtype={"parent_id": str, "child_id": str})
    with open(OUTPUT_DIR / "study_network_basins.txt") as f:
        basins = [line.strip() for line in f if line.strip()]

    topo = pd.read_csv(
        DATA_DIR / "camels_attributes_v2.0" / "camels_topo.txt",
        sep=";", dtype={"gauge_id": str}
    ).set_index("gauge_id")

    G = nx.DiGraph()
    G.add_nodes_from(basins)
    for _, row in edges.iterrows():
        G.add_edge(row["parent_id"], row["child_id"])

    # Compute depth per basin (max distance from any root)
    roots = [n for n in G if G.in_degree(n) == 0]
    depth = {n: 0 for n in basins}
    for root in roots:
        for node, d in nx.single_source_shortest_path_length(G, root).items():
            depth[node] = max(depth[node], d)

    max_depth = max(depth.values())

    return G, basins, topo, edges, depth, roots, max_depth


def get_node_colors(G, basins):
    """Color by role: headwater=blue, interior=grey, outlet=red."""
    colors = []
    for b in basins:
        if G.in_degree(b) == 0:
            colors.append("#3b82f6")   # blue (headwater/root)
        elif G.out_degree(b) == 0:
            colors.append("#ef4444")   # red (outlet/leaf)
        else:
            colors.append("#9ca3af")   # grey (interior)
    return colors


def get_node_sizes(topo, basins, base=40, scale=0.02):
    """Size proportional to drainage area."""
    areas = [topo.loc[b, "area_gages2"] for b in basins]
    return [base + a * scale for a in areas]


def plot_geographic_map(G, basins, topo, depth, roots, max_depth):
    """Plot basins at their geographic coordinates with directed edges."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    lons = [topo.loc[b, "gauge_lon"] for b in basins]
    lats = [topo.loc[b, "gauge_lat"] for b in basins]
    colors = get_node_colors(G, basins)
    sizes = get_node_sizes(topo, basins, base=60, scale=0.03)

    # Draw edges as arrows
    for u, v in G.edges():
        x0, y0 = topo.loc[u, "gauge_lon"], topo.loc[u, "gauge_lat"]
        x1, y1 = topo.loc[v, "gauge_lon"], topo.loc[v, "gauge_lat"]
        dx, dy = x1 - x0, y1 - y0
        ax.annotate("",
                     xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="-|>", color="#6b7280",
                                     lw=1.0, shrinkA=6, shrinkB=6,
                                     connectionstyle="arc3,rad=0.1"))

    # Draw nodes
    ax.scatter(lons, lats, c=colors, s=sizes, zorder=5, edgecolors="black", linewidths=0.5)

    # Label with last 5 digits + depth
    for b, x, y in zip(basins, lons, lats):
        label = b[-5:]
        ax.annotate(label, (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=7, color="#374151")

    n_edges = G.number_of_edges()
    ax.set_title(f"CAMELS-US Study Network — {len(basins)} basins, "
                 f"{n_edges} edges, max depth {max_depth}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#3b82f6", edgecolor="black", label="Headwater (root)"),
        mpatches.Patch(facecolor="#9ca3af", edgecolor="black", label="Interior"),
        mpatches.Patch(facecolor="#ef4444", edgecolor="black", label="Outlet (leaf)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "study_network_map.png", dpi=150, bbox_inches="tight")
    print(f"Saved geographic map to outputs/study_network_map.png")
    plt.close(fig)


def plot_dag_layout(G, basins, topo, depth, roots, max_depth):
    """Plot the topological DAG structure."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    # Try graphviz 'dot' layout, fall back to spring
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except (ImportError, Exception):
        try:
            from networkx.drawing.nx_pydot import graphviz_layout
            pos = graphviz_layout(G, prog="dot")
        except (ImportError, Exception):
            # Fallback: manual layered layout by depth
            pos = _layered_layout(G, basins, depth)

    colors = get_node_colors(G, basins)
    sizes = get_node_sizes(topo, basins, base=80, scale=0.04)

    # Create ordered positions for drawing
    node_list = basins
    pos_arr = np.array([pos[b] for b in node_list])
    color_list = colors
    size_list = sizes

    # Draw edges
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.annotate("",
                     xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="-|>", color="#6b7280",
                                     lw=1.0, shrinkA=8, shrinkB=8))

    # Draw nodes
    ax.scatter(pos_arr[:, 0], pos_arr[:, 1], c=color_list, s=size_list,
               zorder=5, edgecolors="black", linewidths=0.5)

    # Labels: last 5 digits
    for b in basins:
        x, y = pos[b]
        label = b[-5:]
        ax.annotate(label, (x, y), textcoords="offset points",
                     xytext=(6, 6), fontsize=7, color="#374151")

    n_edges = G.number_of_edges()
    ax.set_title(f"Study Network DAG — {len(basins)} basins, "
                 f"{n_edges} edges, max depth {max_depth}",
                 fontsize=13, fontweight="bold")

    legend_elements = [
        mpatches.Patch(facecolor="#3b82f6", edgecolor="black", label="Headwater (depth=0)"),
        mpatches.Patch(facecolor="#9ca3af", edgecolor="black", label="Interior"),
        mpatches.Patch(facecolor="#ef4444", edgecolor="black", label="Outlet (leaf)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "study_network_dag.png", dpi=150, bbox_inches="tight")
    print(f"Saved DAG layout to outputs/study_network_dag.png")
    plt.close(fig)


def _layered_layout(G, basins, depth):
    """Manual layered layout: nodes placed by depth (y) and spread within layer (x)."""
    layers = {}
    for b in basins:
        d = depth[b]
        if d not in layers:
            layers[d] = []
        layers[d].append(b)

    pos = {}
    max_layer_size = max(len(v) for v in layers.values())
    for d, nodes in sorted(layers.items()):
        n = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            x = (i - (n - 1) / 2) * (max_layer_size / max(n, 1))
            y = -d * 2  # depth increases downward
            pos[node] = (x, y)
    return pos


def main():
    G, basins, topo, edges, depth, roots, max_depth = load_study_network()

    print(f"Study network: {len(basins)} basins, {G.number_of_edges()} edges, "
          f"max depth {max_depth}")

    plot_geographic_map(G, basins, topo, depth, roots, max_depth)
    plot_dag_layout(G, basins, topo, depth, roots, max_depth)


if __name__ == "__main__":
    main()
