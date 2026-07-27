"""Build directionality-control upstream-Q features: REVERSED edges and RANDOM-REWIRED graph.

Tests whether the upstream-flow gain is directionally specific (routing) and topology-specific,
or merely generic spatial correlation. Mirror image of Kirschstein & Sun (ICML 2024): GNNs are
direction-INSENSITIVE (similar performance whether edges maintained/reversed/permuted). If our
feature is direction-SENSITIVE, we exhibit the property whose absence explains the GNN null.

All variants use OBSERVED discharge (oracle-style) so the ONLY variable vs the forward oracle is
the edge set. Aggregation is identical to build_upstream_discharge_feature.py (area-weighted mean
of parents' lagged observed Q). Feature index named 'date' (NH concatenation requirement; the
same root cause as commit 52bd535).

Variants:
  reversed: parent<->child swapped (aggregate DOWNSTREAM flow as fake "upstream")
  random  : degree-preserving random rewire (each basin keeps its in-degree; random parents; seed 42)

Usage (Colab, CAMELS on Drive):
  python experiments/topology_ablation/build_directionality_variants.py --network component0 --lag-days 1
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from neuralhydrology.datasetzoo.camelsus import load_camels_us_discharge, load_camels_us_forcings

DATA_DIR = ROOT / "datasets" / "camels_us"
TOPO_TXT = DATA_DIR / "camels_attributes_v2.0" / "camels_topo.txt"
P1 = ROOT / "topology_analysis" / "phase1_network_discovery" / "outputs"
OUT_DIR = Path(__file__).parent / "features"
RNG_SEED = 42  # fixed for reproducible random rewire


def build_feature(G, basins, qobs, area, lag):
    """Area-weighted mean of parents' lagged observed discharge. Index named 'date'."""
    feats = {}
    for b in basins:
        own = qobs.get(b)
        if own is None:
            continue
        idx = pd.DatetimeIndex(own.index, name="date")  # <- named index (NH requirement)
        parents = list(G.predecessors(b))
        if not parents:
            feats[b] = pd.DataFrame({"upstream_q": np.zeros(len(idx))}, index=idx)
            continue
        agg = pd.Series(0.0, index=own.index)
        wsum = 0.0
        for p in parents:
            qp = qobs.get(p)
            if qp is None:
                continue
            pa = float(area.get(p, 0.0))
            agg = agg.add((qp.reindex(own.index) * pa).fillna(0.0), fill_value=0.0)
            wsum += pa
        if wsum > 0:
            agg = agg / wsum
        agg = agg.shift(lag).fillna(0.0)
        feats[b] = pd.DataFrame({"upstream_q": agg.values}, index=idx)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="component0")
    ap.add_argument("--lag-days", type=int, default=1)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    basins = [l.strip() for l in open(P1 / f"{args.network}_basins.txt") if l.strip()]
    edges = pd.read_csv(P1 / f"{args.network}_edges.csv", dtype={"parent_id": str, "child_id": str})
    topo = pd.read_csv(TOPO_TXT, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
    area = topo["area_gages2"].to_dict()

    # observed discharge (mm/d) per basin — same loader as the forward builder
    qobs = {}
    for b in basins:
        try:
            _, a_m2 = load_camels_us_forcings(DATA_DIR, b, "maurer")
            qobs[b] = load_camels_us_discharge(DATA_DIR, b, a_m2)
        except Exception as e:
            print(f"  [warn] no discharge for {b}: {e}")
            qobs[b] = None

    fwd = [(r.parent_id, r.child_id) for r in edges.itertuples()
           if r.parent_id in basins and r.child_id in basins]
    indeg = {b: 0 for b in basins}
    for _, c in fwd:
        indeg[c] += 1

    # ===== REVERSED: swap parent<->child (aggregate downstream flow) =====
    Grev = nx.DiGraph()
    Grev.add_nodes_from(basins)
    for p, c in fwd:
        Grev.add_edge(c, p)
    feats_rev = build_feature(Grev, basins, qobs, area, args.lag_days)
    pickle.dump(feats_rev, open(OUT_DIR / f"upstream_q_reversed_{args.network}_lag{args.lag_days}.p", "wb"))
    pd.DataFrame([(c, p) for p, c in fwd], columns=["parent_id", "child_id"]).to_csv(
        P1 / f"{args.network}_edges_reversed.csv", index=False)

    # ===== RANDOM: degree-preserving rewire (each basin keeps its in-degree) =====
    rng = np.random.default_rng(RNG_SEED)
    Grand = nx.DiGraph()
    Grand.add_nodes_from(basins)
    basin_arr = np.array(basins)
    rand_edges = []
    for c in basins:
        k = indeg[c]
        if k == 0:
            continue
        choices = basin_arr[basin_arr != c]
        parents = rng.choice(choices, size=min(k, len(choices)), replace=False)
        for p in parents:
            Grand.add_edge(str(p), c)
            rand_edges.append((str(p), c))
    feats_rand = build_feature(Grand, basins, qobs, area, args.lag_days)
    pickle.dump(feats_rand, open(OUT_DIR / f"upstream_q_random_{args.network}_lag{args.lag_days}.p", "wb"))
    pd.DataFrame(rand_edges, columns=["parent_id", "child_id"]).to_csv(
        P1 / f"{args.network}_edges_random.csv", index=False)

    for tag, f in [("reversed", feats_rev), ("random", feats_rand)]:
        nz = sum(1 for v in f.values() if np.abs(v["upstream_q"].values).mean() > 0)
        nm = f[next(iter(f))].index.name
        print(f"{tag}: {len(f)} basins, {nz} connected, index.name={nm!r} -> saved")


if __name__ == "__main__":
    main()
