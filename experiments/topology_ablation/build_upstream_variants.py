"""Build upstream-signal VARIANT features for the post-oracle chain:
  - upstream_q_shuffled : the oracle upstream_q, values permuted in time per basin
                          (null control — same marginal, destroyed temporal signal)
  - upstream_precip     : area-weighted lagged upstream PRECIPITATION (content test:
                          is the gain just upstream rain rather than discharge?)

Reuses the verified oracle aggregation (area-weighted mean; discharge area in m^2 from
the forcing header). Output is a {basin: DataFrame} pickle for NH additional_feature_files.

Usage:
    python experiments/topology_ablation/build_upstream_variants.py --network component0 \
        --variant shuffled --lag-days 1 --seed 11
    python experiments/topology_ablation/build_upstream_variants.py --network component0 \
        --variant precip --lag-days 1
"""
import argparse, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from neuralhydrology.datasetzoo.camelsus import load_camels_us_discharge, load_camels_us_forcings

DATA_DIR = ROOT / "datasets" / "camels_us"
TOPO_TXT = DATA_DIR / "camels_attributes_v2.0" / "camels_topo.txt"
OUT_DIR = Path(__file__).parent / "features"


def files_for(net):
    if net == "component0":
        return (ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_basins.txt",
                ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_edges.csv")
    return (ROOT / f"experiments/local_subgraphs/basin_lists/{net}_basins.txt",
            ROOT / f"experiments/local_subgraphs/basin_lists/{net}_edges.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True)
    ap.add_argument("--variant", required=True, choices=["shuffled", "precip"])
    ap.add_argument("--lag-days", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11, help="permutation seed for shuffled")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bf, ef = files_for(args.network)
    basins = [l.strip() for l in open(bf) if l.strip()]
    edges = pd.read_csv(ef, dtype={"parent_id": str, "child_id": str})
    area = pd.read_csv(TOPO_TXT, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")["area_gages2"].to_dict()

    G = nx.DiGraph()
    for b in basins:
        G.add_node(b)
    for _, r in edges.iterrows():
        if r["parent_id"] in basins and r["child_id"] in basins:
            G.add_edge(r["parent_id"], r["child_id"])

    # Per-basin upstream signal series (discharge or precip), area-weighted mean, lagged.
    series = {}   # basin -> per-day signal from its OWN forcing/discharge
    for b in basins:
        try:
            forc, a_m2 = load_camels_us_forcings(DATA_DIR, b, "maurer")
            if args.variant == "precip":
                series[b] = forc["PRCP(mm/day)"]
            else:  # shuffled uses discharge (same source as the oracle)
                series[b] = load_camels_us_discharge(DATA_DIR, b, a_m2)
        except Exception as e:
            print(f"  [warn] {b}: {e}")
            series[b] = None

    colname = "upstream_q" if args.variant == "shuffled" else "upstream_q"  # keep col name 'upstream_q' so configs are identical
    rng = np.random.default_rng(args.seed)
    feats = {}
    n_conn = 0
    for b in basins:
        own = series[b]
        if own is None:
            continue
        idx = own.index
        parents = list(G.predecessors(b))
        if not parents:
            feats[b] = pd.DataFrame({colname: np.zeros(len(idx))}, index=idx)
            continue
        agg = pd.Series(0.0, index=idx); wsum = 0.0
        for p in parents:
            sp = series.get(p)
            if sp is None:
                continue
            pa = float(area.get(p, 0.0))
            agg = agg.add((sp.reindex(idx) * pa).fillna(0.0), fill_value=0.0)
            wsum += pa
        if wsum > 0:
            agg = agg / wsum
        agg = agg.shift(args.lag_days).fillna(0.0)
        vals = agg.values
        if args.variant == "shuffled":
            vals = vals.copy(); rng.shuffle(vals)   # permute in time, destroy alignment
        feats[b] = pd.DataFrame({colname: vals}, index=idx)
        n_conn += 1

    tag = "upstream_q_shuffled" if args.variant == "shuffled" else "upstream_precip"
    out = OUT_DIR / f"{tag}_{args.network}_lag{args.lag_days}.p"
    pickle.dump(feats, open(out, "wb"))
    print(f"Wrote {tag} for {len(feats)} basins ({n_conn} with upstream) -> {out}")


if __name__ == "__main__":
    main()
