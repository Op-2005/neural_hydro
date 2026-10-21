"""Build the inputs for the own-lagged-discharge conditions (camera-ready, reviewer request).

Two kinds of {basin: DataFrame} pickle for NH `additional_feature_files`:

  self_q      : the basin's OWN observed discharge, lagged one day (mm/d). Missing days are
                forward-filled with the last available reading, which is what an operational
                forecaster would hold; any value still missing before the record starts is 0.
                Filling is required, not cosmetic: NH drops training samples with a NaN input
                but keeps them at evaluation, where the NaN would poison that day's prediction
                and the next seq_length days, changing the evaluated day set relative to L.

  upstream_q  : neighbour discharge from ANY edge file, using exactly the aggregation of
                build_upstream_discharge_feature.py (area-weighted mean of mm/d series, lagged,
                missing filled with 0). Driving it from an edge file lets the network and the
                nearest-gauge selector be rebuilt from the committed edge lists.

Usage:
    python experiments/topology_ablation/build_own_lag_inputs.py --self-lag
    python experiments/topology_ablation/build_own_lag_inputs.py \
        --edge-file topology_analysis/phase1_network_discovery/outputs/component0_edges_knn2.csv --tag knn2
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from neuralhydrology.datasetzoo.camelsus import load_camels_us_discharge, load_camels_us_forcings

DATA_DIR = ROOT / "datasets" / "camels_us"
TOPO_TXT = DATA_DIR / "camels_attributes_v2.0" / "camels_topo.txt"
BASINS = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_basins.txt"
OUT_DIR = Path(__file__).parent / "features"


def load_discharge(basins):
    # Same loader and area source as build_upstream_discharge_feature.py: the forcing-header
    # area in m^2, not area_gages2 in km^2.
    qobs = {}
    for b in basins:
        try:
            _, a_m2 = load_camels_us_forcings(DATA_DIR, b, "maurer")
            qobs[b] = load_camels_us_discharge(DATA_DIR, b, a_m2)
        except Exception as e:
            print(f"  [warn] no discharge for {b}: {e}")
            qobs[b] = None
    return qobs


def build_self_lag(basins, qobs, lag):
    feats, filled = {}, []
    for b in basins:
        own = qobs[b]
        if own is None:
            continue
        s = own.shift(lag)
        n_nan = int(s.isna().sum())
        s = s.ffill().fillna(0.0)
        filled.append(n_nan / len(s))
        feats[b] = pd.DataFrame({"self_q": s.values}, index=own.index)
    print(f"self_q: {len(feats)} basins; days forward-filled per basin: "
          f"mean {100 * np.mean(filled):.2f}%, max {100 * np.max(filled):.2f}%")
    return feats


def build_from_edges(basins, qobs, edge_file, lag):
    edges = pd.read_csv(edge_file, dtype={"parent_id": str, "child_id": str})
    topo = pd.read_csv(TOPO_TXT, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
    area = topo["area_gages2"].to_dict()
    bset = set(basins)
    parents = {b: [] for b in basins}
    for _, r in edges.iterrows():
        if r["parent_id"] in bset and r["child_id"] in bset:
            parents[r["child_id"]].append(r["parent_id"])
    feats = {}
    for b in basins:
        own = qobs[b]
        if own is None:
            continue
        idx = own.index
        if not parents[b]:
            feats[b] = pd.DataFrame({"upstream_q": np.zeros(len(idx))}, index=idx)
            continue
        agg, wsum = pd.Series(0.0, index=idx), 0.0
        for p in parents[b]:
            qp = qobs.get(p)
            if qp is None:
                continue
            pa = float(area.get(p, 0.0))
            agg = agg.add((qp.reindex(idx) * pa).fillna(0.0), fill_value=0.0)
            wsum += pa
        if wsum > 0:
            agg = agg / wsum
        agg = agg.shift(lag).fillna(0.0)
        feats[b] = pd.DataFrame({"upstream_q": agg.values}, index=idx)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-lag", action="store_true", help="build the own-lagged-discharge pickle")
    ap.add_argument("--edge-file", help="build upstream_q from this edge list instead")
    ap.add_argument("--tag", help="name for the edge-file pickle, e.g. 'network' or 'knn2'")
    ap.add_argument("--lag-days", type=int, default=1)
    args = ap.parse_args()
    if bool(args.self_lag) == bool(args.edge_file):
        ap.error("pass exactly one of --self-lag or --edge-file")
    if args.edge_file and not args.tag:
        ap.error("--edge-file needs --tag")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    basins = [l.strip() for l in open(BASINS) if l.strip()]
    qobs = load_discharge(basins)
    if args.self_lag:
        feats = build_self_lag(basins, qobs, args.lag_days)
        out = OUT_DIR / f"self_q_component0_lag{args.lag_days}.p"
    else:
        feats = build_from_edges(basins, qobs, ROOT / args.edge_file, args.lag_days)
        out = OUT_DIR / f"upstream_q_{args.tag}_edgefile_component0_lag{args.lag_days}.p"
    # NH merges on a column named 'date'; an unnamed index raises KeyError at training start.
    for b in feats:
        feats[b].index.name = "date"
    with open(out, "wb") as f:
        pickle.dump(feats, f)
    print(f"wrote {len(feats)} basins -> {out}")


if __name__ == "__main__":
    main()
