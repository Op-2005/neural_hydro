"""Build the DISTANCE-PRESERVING graph control feature (experiment #1).

Motivation. The paper's headline mechanism result is that true edges >> a degree-preserving RANDOM
rewire (+0.034 NSE). But the random rewire destroys BOTH the true topology AND spatial proximity: it
connects each basin to arbitrary, often distant basins. A reviewer can therefore argue the gap
conflates "real upstream routing" with "nearby basins share weather/flow." This control removes that
confound: it rewires each basin to NON-upstream basins at the SAME distances as its true parents, so
proximity is preserved and only the true upstream topology is destroyed.

Design (degree- AND distance-preserving):
  For each basin c with true parents {p_i} at haversine distances {d_i}, substitute each with the
  available basin q (q != c, q not a true parent of c, not already chosen for c) whose distance to c
  is CLOSEST to d_i. This keeps c's in-degree exactly and matches each true edge's length as tightly
  as the basin set allows, while every parent is a different basin. On component0 this yields mean
  edge length 101 km vs true 92 km, versus 511 km for the uniform random rewire: the control holds
  proximity nearly fixed while destroying the true topology, which the random rewire cannot do. (The
  M_NEAREST knob can randomise among the M nearest substitutes; M=1 is the tightest, canonical match.)

All variants use OBSERVED discharge (oracle-style), aggregation identical to
build_directionality_variants.py, so the only variable vs the forward oracle is the edge set.

Dry-run the graph logic (no discharge needed):
  python experiments/topology_ablation/build_distance_control.py --dry-run
Full build (Colab, CAMELS on Drive):
  python experiments/topology_ablation/build_distance_control.py --network component0 --lag-days 1
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

DATA_DIR = ROOT / "datasets" / "camels_us"
TOPO_TXT = DATA_DIR / "camels_attributes_v2.0" / "camels_topo.txt"
P1 = ROOT / "topology_analysis" / "phase1_network_discovery" / "outputs"
OUT_DIR = Path(__file__).parent / "features"
RNG_SEED = 42
M_NEAREST = 1  # substitute each true edge with the NEAREST-distance available basin (tightest match)


def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_distance_graph(basins, fwd, coords, seed=RNG_SEED):
    """Return a degree- and distance-preserving rewired DiGraph (parents != true, distances matched)."""
    lat = {b: coords[b][0] for b in basins}
    lon = {b: coords[b][1] for b in basins}

    def dist(a, b):
        return haversine(lat[a], lon[a], lat[b], lon[b])

    true_parents = {b: [] for b in basins}
    for p, c in fwd:
        true_parents[c].append(p)

    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(basins)
    edges_out = []
    for c in basins:
        tps = true_parents[c]
        if not tps:
            continue
        forbidden = set(tps) | {c}
        chosen = []
        for p in tps:
            dt = dist(c, p)
            avail = [q for q in basins if q not in forbidden and q not in chosen]
            avail.sort(key=lambda q: abs(dist(c, q) - dt))  # nearest edge-length first
            pick = str(rng.choice(avail[:M_NEAREST]))       # random among the M nearest
            chosen.append(pick)
            G.add_edge(pick, c)
            edges_out.append((pick, c))
    return G, edges_out, dist


def load_inputs(network):
    basins = [l.strip() for l in open(P1 / f"{network}_basins.txt") if l.strip()]
    edges = pd.read_csv(P1 / f"{network}_edges.csv", dtype={"parent_id": str, "child_id": str})
    fwd = [(r.parent_id, r.child_id) for r in edges.itertuples()
           if r.parent_id in basins and r.child_id in basins]
    topo = pd.read_csv(TOPO_TXT, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
    coords = {b: (float(topo.loc[b, "gauge_lat"]), float(topo.loc[b, "gauge_lon"]))
              for b in basins if b in topo.index}
    return basins, fwd, coords, edges


def build_feature(G, basins, qobs, area, lag):
    """Area-weighted mean of parents' lagged observed discharge. Index named 'date' (NH requirement)."""
    feats = {}
    for b in basins:
        own = qobs.get(b)
        if own is None:
            continue
        idx = pd.DatetimeIndex(own.index, name="date")
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
    ap.add_argument("--dry-run", action="store_true",
                    help="build + validate the graph only (no discharge, no feature file)")
    args = ap.parse_args()

    basins, fwd, coords, edges = load_inputs(args.network)
    G, edges_out, dist = build_distance_graph(basins, fwd, coords)

    # ---- validation: degree preserved, distances matched, topology destroyed ----
    true_indeg, new_indeg = {b: 0 for b in basins}, {b: 0 for b in basins}
    for _, c in fwd:
        true_indeg[c] += 1
    for _, c in edges_out:
        new_indeg[c] += 1
    deg_ok = all(true_indeg[b] == new_indeg[b] for b in basins)
    true_set = set(fwd)
    overlap = sum(1 for e in edges_out if e in true_set)
    true_d = np.array([dist(c, p) for p, c in fwd])
    new_d = np.array([dist(p, c) for p, c in edges_out])
    print(f"[{args.network}] true edges {len(fwd)} | rewired edges {len(edges_out)}")
    print(f"  in-degree preserved exactly: {deg_ok}")
    print(f"  rewired edges identical to a true edge: {overlap} ({100*overlap/len(edges_out):.1f}%)")
    print(f"  edge-distance km  true : mean {true_d.mean():.1f}  median {np.median(true_d):.1f}")
    print(f"  edge-distance km  dist : mean {new_d.mean():.1f}  median {np.median(new_d):.1f}")
    print(f"  per-edge |Δdistance| : mean {np.abs(new_d - true_d).mean():.2f} km "
          f"(median {np.median(np.abs(new_d - true_d)):.2f})")

    if args.dry_run:
        pd.DataFrame(edges_out, columns=["parent_id", "child_id"]).to_csv(
            P1 / f"{args.network}_edges_distctrl.csv", index=False)
        print(f"  -> wrote edge list {P1}/{args.network}_edges_distctrl.csv (dry-run, no feature)")
        return

    from neuralhydrology.datasetzoo.camelsus import (load_camels_us_discharge,
                                                     load_camels_us_forcings)
    topo = pd.read_csv(TOPO_TXT, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
    area = topo["area_gages2"].to_dict()
    qobs = {}
    for b in basins:
        try:
            _, a_m2 = load_camels_us_forcings(DATA_DIR, b, "maurer")
            qobs[b] = load_camels_us_discharge(DATA_DIR, b, a_m2)
        except Exception as e:
            print(f"  [warn] no discharge for {b}: {e}")
            qobs[b] = None
    feats = build_feature(G, basins, qobs, area, args.lag_days)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"upstream_q_distctrl_{args.network}_lag{args.lag_days}.p"
    pickle.dump(feats, open(out, "wb"))
    nz = sum(1 for v in feats.values() if np.abs(v["upstream_q"].values).mean() > 0)
    print(f"  feature: {len(feats)} basins, {nz} connected, "
          f"index.name={feats[next(iter(feats))].index.name!r} -> {out}")


if __name__ == "__main__":
    main()
