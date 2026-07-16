"""Graph-robustness chain (Steps A/B/C) — ZERO training.

Tests whether the routing signal is an artifact of the heuristic graph's OVER-CONNECTIVITY
(child in-degree mean 4.16 / max 15 vs real confluences ~2-3).

Method: rebuild upstream_q on alternative graphs from observed discharge (fullspan eval) +
area weights, then score signal strength via the no-ML lstsq routing baseline
(Qhat = a*upstream_q + b, fit on TRAIN 1990-99, scored on TEST 2005-08). Relative comparison
across graph variants is the deliverable; absolute NSE is a lower bound on LSTM-extractable gain.

Step A: prune to k nearest (or largest-area) parents per child; does R1 NSE survive?
Step B: does the depth hierarchy survive k=2 pruning?
Step C: random edge-dropout sensitivity (5 draws, 20% and 40%).

Pre-reg: preregistration_graph_robustness_chain.md.
Writes analysis/GRAPH_ROBUSTNESS.md.
"""
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "topology_ablation" / "component0"
P1 = ROOT / "topology_analysis/phase1_network_discovery/outputs"
TOPO_TXT = ROOT / "datasets/camels_us/camels_attributes_v2.0/camels_topo.txt"
OUT = ROOT / "experiments/topology_ablation/analysis/GRAPH_ROBUSTNESS.md"

TRAIN = ("1990-01-01", "1999-12-31")
TEST = ("2005-01-01", "2008-12-31")
LAG = 1


def load_obs():
    """observed Q per basin over 1990-2008 (from fullspan eval), indexed by date."""
    p = BASE / "_Lfullspan_eval_seed11" / "test" / "model_epoch030" / "test_results.p"
    res = pickle.load(open(p, "rb"))
    out = {}
    for b, d in res.items():
        xr = d["1D"]["xr"]
        idx = pd.to_datetime(xr["date"].values)
        out[b] = pd.Series(xr["QObs(mm/d)_obs"].values.squeeze(), index=idx)
    return out


def build_upstream_q(edges_df, basins, obs, area):
    """Rebuild the area-weighted-mean lagged upstream_q feature for a given edge set."""
    G = nx.DiGraph()
    G.add_nodes_from(basins)
    for _, r in edges_df.iterrows():
        if r["parent_id"] in basins and r["child_id"] in basins:
            G.add_edge(r["parent_id"], r["child_id"])
    feats = {}
    for b in basins:
        if b not in obs:
            continue
        idx = obs[b].index
        parents = list(G.predecessors(b))
        if not parents:
            feats[b] = pd.Series(0.0, index=idx)
            continue
        agg = pd.Series(0.0, index=idx); wsum = 0.0
        for p in parents:
            if p not in obs:
                continue
            pa = float(area.get(p, 0.0))
            agg = agg.add((obs[p].reindex(idx) * pa).fillna(0.0), fill_value=0.0)
            wsum += pa
        if wsum > 0:
            agg = agg / wsum
        feats[b] = agg.shift(LAG).fillna(0.0)
    return feats, G


def r1_nse(feats, obs):
    """Median test-NSE of the pure-routing lstsq predictor over connected basins."""
    nses = []
    for b, upq in feats.items():
        if b not in obs or np.abs(upq.values).mean() == 0:
            continue
        df = pd.DataFrame({"obs": obs[b], "upq": upq}).dropna()
        tr = df.loc[TRAIN[0]:TRAIN[1]]; te = df.loc[TEST[0]:TEST[1]]
        if len(tr) < 100 or len(te) < 100:
            continue
        A = np.column_stack([tr["upq"].values, np.ones(len(tr))])
        coef, *_ = np.linalg.lstsq(A, tr["obs"].values, rcond=None)
        pred = te["upq"].values * coef[0] + coef[1]
        o = te["obs"].values
        den = np.sum((o - o.mean()) ** 2)
        if den > 0:
            nses.append(1 - np.sum((o - pred) ** 2) / den)
    return np.median(nses) if nses else np.nan, len(nses)


def compute_depths(G):
    if not nx.is_directed_acyclic_graph(G):
        return None
    depths = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        depths[node] = 0 if not preds else 1 + max(depths[p] for p in preds)
    return depths


def prune_knearest(edges_df, k, by="distance_km", ascending=True):
    """Keep only the k best parents per child (nearest by distance, or largest by area)."""
    return edges_df.sort_values(by, ascending=ascending).groupby("child_id").head(k)


def main():
    obs = load_obs()
    basins = [l.strip() for l in open(P1 / "component0_basins.txt") if l.strip()]
    edges = pd.read_csv(P1 / "component0_edges.csv", dtype={"parent_id": str, "child_id": str})
    topo = pd.read_csv(TOPO_TXT, sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
    area = topo["area_gages2"].to_dict()

    md = ["# Graph-Robustness Chain — is the routing signal an over-connectivity artifact?", "",
          "ZERO training. upstream_q rebuilt on alternative graphs from observed discharge; "
          "signal strength scored via no-ML lstsq routing baseline (R1: Qhat=a·upstream_q+b, "
          "fit TRAIN 1990-99, scored TEST 2005-08, median over connected basins). "
          "Pre-reg: `preregistration_graph_robustness_chain.md`.", ""]

    # ---------- baseline: full graph ----------
    feats_full, G_full = build_upstream_q(edges, basins, obs, area)
    r1_full, n_full = r1_nse(feats_full, obs)
    depths_full = compute_depths(G_full)
    md += [f"**Full graph** (624 edges, in-degree mean 4.16 / max 15): "
           f"R1 median test-NSE = **{r1_full:+.4f}** (n={n_full} connected basins).", ""]

    # ================= STEP A =================
    md += ["## Step A — pruned-graph robustness (over-connectivity test)", "",
           "| pruning | rule | edges kept | R1 median NSE | % of full |",
           "|---|---|---|---|---|"]
    resA = {}
    for k in [1, 2, 3]:
        for by, asc, rule in [("distance_km", True, "nearest"), ("area_ratio", True, "smallest-ratio")]:
            pruned = prune_knearest(edges, k, by=by, ascending=asc)
            feats_k, _ = build_upstream_q(pruned, basins, obs, area)
            r1_k, n_k = r1_nse(feats_k, obs)
            pct = 100 * r1_k / r1_full if r1_full else np.nan
            resA[(k, rule)] = (r1_k, pct)
            md.append(f"| in-degree ≤ {k} | {rule} | {len(pruned)} | {r1_k:+.4f} | {pct:.0f}% |")
    md.append("")
    r1_k2 = resA[(2, "nearest")][1]
    passA = r1_k2 >= 70
    md.append(f"**k=2 (nearest) retains {r1_k2:.0f}% of full-graph R1 NSE.** "
              + ("PASS (≥70%) — routing signal is NOT an over-connectivity artifact."
                 if passA else
                 ("PARTIAL (50-70%)" if r1_k2 >= 50 else
                  "FAIL (<50%) — signal depends on excess edges; STOP chain.")))
    md.append("")

    if r1_k2 < 50:
        OUT.write_text("\n".join(md) + "\n")
        print("\n".join(md)); print("\nStep A FAILED — chain stopped per pre-reg.")
        return

    # ================= STEP B =================
    md += ["## Step B — depth-structure stability under k=2 pruning", ""]
    pruned2 = prune_knearest(edges, 2, by="distance_km", ascending=True)
    _, G2 = build_upstream_q(pruned2, basins, obs, area)
    depths2 = compute_depths(G2)
    if depths_full and depths2:
        common = [b for b in depths_full if b in depths2]
        within1 = sum(1 for b in common if abs(depths_full[b] - depths2[b]) <= 1)
        pct_stable = 100 * within1 / len(common)
        maxd_full = max(depths_full.values()); maxd2 = max(depths2.values())
        md += [f"- basins retaining depth within ±1: **{pct_stable:.0f}%** ({within1}/{len(common)})",
               f"- max depth: full {maxd_full} → pruned {maxd2} (Δ={maxd2-maxd_full})",
               f"- DAG preserved: {nx.is_directed_acyclic_graph(G2)}", ""]
        passB = pct_stable >= 80 and abs(maxd2 - maxd_full) <= 1
        md.append(f"**{'PASS' if passB else 'PARTIAL'}** — depth hierarchy "
                  + ("survives pruning." if passB else "shifts under pruning; scope the claim."))
    else:
        passB = False
        md.append("**depth recomputation failed (non-DAG?)** — investigate.")
    md.append("")

    if not passB:
        OUT.write_text("\n".join(md) + "\n")
        print("\n".join(md)); print("\nStep B did not fully pass — reporting, chain stops per pre-reg.")
        return

    # ================= STEP C =================
    md += ["## Step C — random edge-dropout sensitivity", "",
           "Random dropout of a fraction of edges, 5 fixed-seed draws each; R1 median NSE.", "",
           "| dropout | R1 NSE mean ± std (5 draws) | % of full | max-min spread |",
           "|---|---|---|---|"]
    n_edges = len(edges)
    passC = True
    for frac in [0.20, 0.40]:
        vals = []
        for draw in range(5):
            rng = np.random.default_rng(1000 * draw + int(frac * 100))
            keep_mask = rng.random(n_edges) >= frac
            kept = edges[keep_mask]
            feats_d, _ = build_upstream_q(kept, basins, obs, area)
            r1_d, _ = r1_nse(feats_d, obs)
            vals.append(r1_d)
        vals = np.array(vals)
        pct = 100 * vals.mean() / r1_full if r1_full else np.nan
        spread = vals.max() - vals.min()
        md.append(f"| {int(frac*100)}% | {vals.mean():+.4f} ± {vals.std():.4f} | "
                  f"{pct:.0f}% | {spread:.4f} |")
        if frac == 0.20:
            within15 = abs(pct - 100) <= 15
            low_var = vals.std() <= 0.20 * abs(vals.mean())
            passC = within15 and low_var
    md.append("")
    md.append(f"**20% dropout: {'PASS' if passC else 'PARTIAL'}** — "
              + ("signal degrades gracefully; not dependent on specific edges."
                 if passC else "sensitive to edge choice; report variance honestly."))

    OUT.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n-> wrote {OUT}")


if __name__ == "__main__":
    main()
