"""Build the REALIZABLE upstream-Q feature: area-weighted upstream PREDICTED discharge.

Stage 1: run the trained L baseline over the full span (1990-2008) to get predicted Q
         for every basin (no observed Q used as the signal -> deployable).
Stage 2: aggregate upstream basins' predicted Q (area-weighted mean, lagged) per downstream
         basin. Same aggregation as the oracle, swapping observed for predicted.

Output: {basin: DataFrame(col 'upstream_q')} pickle for NH additional_feature_files
(col name kept 'upstream_q' so the downstream config is identical to the oracle's).

Usage:
    python experiments/topology_ablation/build_predicted_upstream_q.py --network component0 --lag-days 1
"""
import argparse, pickle, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd, networkx as nx, yaml

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "datasets" / "camels_us"
TOPO_TXT = DATA_DIR / "camels_attributes_v2.0" / "camels_topo.txt"
OUT_DIR = Path(__file__).parent / "features"
L_RUN = ROOT / "runs" / "topology_ablation" / "component0" / "L_component0_seed11"


def files_for(net):
    if net == "component0":
        return (ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_basins.txt",
                ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_edges.csv")
    return (ROOT / f"experiments/local_subgraphs/basin_lists/{net}_basins.txt",
            ROOT / f"experiments/local_subgraphs/basin_lists/{net}_edges.csv")


def full_span_predictions():
    """Copy L run, override eval period to 1990-2008, run evaluate, read predicted Q."""
    eval_dir = ROOT / "runs" / "topology_ablation" / "component0" / "_Lfullspan_eval"
    if not (eval_dir / "test" / "model_epoch030" / "test_results.p").exists():
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        shutil.copytree(L_RUN, eval_dir)
        cfg = yaml.safe_load(open(eval_dir / "config.yml"))
        cfg["test_start_date"] = "01/01/1990"
        cfg["test_end_date"] = "31/12/2008"
        cfg["run_dir"] = str(eval_dir)
        import torch as _t
        cfg["device"] = "cuda:0" if _t.cuda.is_available() else "cpu"
        # remove stale test outputs so evaluate regenerates over the new span
        if (eval_dir / "test").exists():
            shutil.rmtree(eval_dir / "test")
        yaml.safe_dump(cfg, open(eval_dir / "config.yml", "w"), sort_keys=False)
        print("Running full-span evaluate (1990-2008)...")
        subprocess.run([sys.executable, "neuralhydrology/nh_run.py", "evaluate",
                        "--run-dir", str(eval_dir), "--epoch", "30"], cwd=ROOT, check=True)
    res = pickle.load(open(eval_dir / "test" / "model_epoch030" / "test_results.p", "rb"))
    pred = {}
    for basin, byfreq in res.items():
        ds = byfreq[next(iter(byfreq))]["xr"]
        sim = ds["QObs(mm/d)_sim"].values.flatten()
        dates = pd.to_datetime(ds["date"].values)
        pred[str(basin)] = pd.Series(sim, index=dates)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="component0")
    ap.add_argument("--lag-days", type=int, default=1)
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

    pred = full_span_predictions()   # {basin: predicted Q series}

    feats, n_conn = {}, 0
    for b in basins:
        if b not in pred:
            continue
        idx = pred[b].index
        parents = list(G.predecessors(b))
        if not parents:
            feats[b] = pd.DataFrame({"upstream_q": np.zeros(len(idx))}, index=idx)
            continue
        agg = pd.Series(0.0, index=idx); wsum = 0.0
        for p in parents:
            if p not in pred:
                continue
            pa = float(area.get(p, 0.0))
            agg = agg.add((pred[p].reindex(idx) * pa).fillna(0.0), fill_value=0.0)
            wsum += pa
        if wsum > 0:
            agg = agg / wsum
        agg = agg.shift(args.lag_days).fillna(0.0)
        feats[b] = pd.DataFrame({"upstream_q": agg.values}, index=idx)
        n_conn += 1

    out = OUT_DIR / f"upstream_q_pred_{args.network}_lag{args.lag_days}.p"
    pickle.dump(feats, open(out, "wb"))
    vals = np.concatenate([np.abs(d["upstream_q"].values) for d in feats.values()])
    print(f"Wrote upstream_q_pred for {len(feats)} basins ({n_conn} with upstream) -> {out}")
    print(f"  mean |upstream_q_pred| = {vals[vals>0].mean():.3f} mm/d (sane if O(0.1-10))")


if __name__ == "__main__":
    main()
