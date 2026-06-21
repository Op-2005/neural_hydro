"""Build the 'upstream observed discharge' dynamic feature — the ORACLE upper bound
on what river-network structure can buy.

RESEARCH INTENT (the bounding experiment):
The PI's framing (2026-04-21) is that the LSTM self-stabilizes into its own attractor;
the question is what EXTERNAL information usefully destabilizes it. Static topology
features are constant (can't destabilize). Learned message passing is a weak proxy.
The STRONGEST possible structural signal is the actual lagged observed discharge of
upstream basins — literally the water that will arrive downstream.

If feeding a downstream basin its upstream neighbours' lagged *observed* discharge does
NOT improve prediction, then no learned message-passing scheme can either — structure
is uninformative for this target, full stop. If it DOES help, it sets the ceiling and
justifies pursuing a realizable (predicted-upstream) message-passing model.

This is an ORACLE / upper-bound feature (uses ground-truth upstream Q). In an operational
nowcast upstream gauges report in real time, so it's also a fair realistic input; but we
frame it as the upper bound to bound the whole enterprise.

Aggregation: area-weighted sum of upstream observed discharge, lagged `lag_days`, then
re-expressed per downstream area (so units stay mm/d). Basins with no upstream get a zero
column.

Output: a pickle {basin: DataFrame(index=date, columns=['upstream_q'])} consumable via
NH's `additional_feature_files`. Add 'upstream_q' to `dynamic_inputs` in the config.

Usage:
    python experiments/topology_ablation/build_upstream_discharge_feature.py \
        --network component0 --lag-days 1
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

import sys
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from neuralhydrology.datasetzoo.camelsus import load_camels_us_discharge, load_camels_us_forcings

DATA_DIR = ROOT / "datasets" / "camels_us"
TOPO_TXT = DATA_DIR / "camels_attributes_v2.0" / "camels_topo.txt"
OUT_DIR = Path(__file__).parent / "features"


def basin_file_for(network):
    if network == "component0":
        return ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_basins.txt"
    return ROOT / f"experiments/local_subgraphs/basin_lists/{network}_basins.txt"


def edge_file_for(network):
    if network == "component0":
        return ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_edges.csv"
    return ROOT / f"experiments/local_subgraphs/basin_lists/{network}_edges.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True)
    ap.add_argument("--lag-days", type=int, default=1)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    basins = [l.strip() for l in open(basin_file_for(args.network)) if l.strip()]
    edges = pd.read_csv(edge_file_for(args.network), dtype={"parent_id": str, "child_id": str})
    topo = pd.read_csv(TOPO_TXT, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
    area = topo["area_gages2"].to_dict()

    # immediate parents per child (direct upstream)
    G = nx.DiGraph()
    for b in basins:
        G.add_node(b)
    for _, r in edges.iterrows():
        if r["parent_id"] in basins and r["child_id"] in basins:
            G.add_edge(r["parent_id"], r["child_id"])

    # Load observed discharge (mm/d) for every basin once.
    # NH's load_camels_us_discharge expects area in m^2 from the forcing-file header
    # (NOT area_gages2 in km^2 — off by 1e6). Use the forcing-header area.
    qobs = {}
    area_m2 = {}
    for b in basins:
        try:
            _, a_m2 = load_camels_us_forcings(DATA_DIR, b, "maurer")
            area_m2[b] = a_m2
            qobs[b] = load_camels_us_discharge(DATA_DIR, b, a_m2)
        except Exception as e:
            print(f"  [warn] no discharge for {b}: {e}")
            qobs[b] = None

    feats = {}
    n_connected = 0
    for b in basins:
        parents = list(G.predecessors(b))
        # downstream date index from its own discharge series
        own = qobs[b]
        if own is None:
            continue
        idx = own.index
        if not parents:
            feats[b] = pd.DataFrame({"upstream_q": np.zeros(len(idx))}, index=idx)
            continue
        # Area-weighted MEAN of upstream discharge in mm/d (already area-normalized,
        # so this stays O(1-10) like the target — avoids one large basin dominating
        # after NH's per-feature standardization). Represents typical upstream runoff
        # intensity weighted by contributing area.
        agg = pd.Series(0.0, index=idx)
        wsum = 0.0
        for p in parents:
            qp = qobs.get(p)
            if qp is None:
                continue
            pa = float(area.get(p, 0.0))
            agg = agg.add((qp.reindex(idx) * pa).fillna(0.0), fill_value=0.0)
            wsum += pa
        if wsum > 0:
            agg = agg / wsum
        agg = agg.shift(args.lag_days).fillna(0.0)   # lag by travel time
        feats[b] = pd.DataFrame({"upstream_q": agg.values}, index=idx)
        n_connected += 1

    out = OUT_DIR / f"upstream_q_{args.network}_lag{args.lag_days}.p"
    with open(out, "wb") as f:
        pickle.dump(feats, f)
    print(f"Wrote upstream_q feature for {len(feats)} basins ({n_connected} with upstream) -> {out}")
    print(f"Add to config:\n  dynamic_inputs: [..., upstream_q]\n  additional_feature_files:\n    - {out}")


if __name__ == "__main__":
    main()
