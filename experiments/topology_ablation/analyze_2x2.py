"""Analyze the encoding x topology 2x2 — the controlled topology-signal test.

Reports per-network:
  - median NSE for each of the 4 conditions (L, L_T, L_noID, L_noID_T)
  - the two key paired contrasts:
      topo_benefit_with_ID    = (L_T - L)            [predict ~0: redundant w/ one-hot]
      topo_benefit_without_ID = (L_noID_T - L_noID)  [predict >0: the headline]
  - the interaction = topo_benefit_with_ID - topo_benefit_without_ID
  - the encoding cost = (L - L_noID): how much the one-hot itself buys

Reads:  runs/topology_ablation/<network>/<cond>_<network>_seed<N>/test/model_epoch030/test_metrics.csv
Writes: experiments/topology_ablation/analysis/{table.csv, RESULTS.md}

Usage:
    python experiments/topology_ablation/analyze_2x2.py
"""
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
RUN_ROOT = ROOT / "runs" / "topology_ablation"
OUT_DIR = Path(__file__).parent / "analysis"
CONDITIONS = ["L", "L_T", "L_noID", "L_noID_T"]


def load(net, cond):
    """Return {basin: NSE} for the (network, condition), merging seeds if several."""
    out = {}
    for p in sorted((RUN_ROOT / net).glob(f"{cond}_{net}_seed*/test/model_epoch*/test_metrics.csv")):
        df = pd.read_csv(p, dtype={"basin": str})
        if "NSE" in df.columns:
            for _, r in df.iterrows():
                out.setdefault(r["basin"], []).append(r["NSE"])
    # average across seeds per basin
    return {b: float(np.mean(v)) for b, v in out.items()}


def paired(a, b):
    common = set(a) & set(b)
    d = np.array([a[x] - b[x] for x in common if not (np.isnan(a[x]) or np.isnan(b[x]))])
    if len(d) == 0:
        return None
    return {"n": len(d), "median": float(np.median(d)), "mean": float(np.mean(d)),
            "frac_pos": float((d > 0).mean())}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    networks = sorted(d.name for d in RUN_ROOT.iterdir() if d.is_dir()) if RUN_ROOT.is_dir() else []

    rows, contrasts = [], []
    for net in networks:
        data = {c: load(net, c) for c in CONDITIONS}
        present = [c for c in CONDITIONS if data[c]]
        if not present:
            continue
        print(f"\n=== {net} ===")
        for c in present:
            med = np.median(list(data[c].values()))
            rows.append({"network": net, "condition": c, "median_NSE": float(med),
                         "n_basins": len(data[c])})
            print(f"  {c:>9}: median NSE {med:+.3f}  (n={len(data[c])})")

        cset = {
            "topo_benefit_with_ID":    ("L_T", "L"),
            "topo_benefit_without_ID": ("L_noID_T", "L_noID"),
            "encoding_cost":           ("L", "L_noID"),
        }
        net_c = {}
        for label, (a, b) in cset.items():
            if data.get(a) and data.get(b):
                r = paired(data[a], data[b])
                if r:
                    net_c[label] = r
                    contrasts.append({"network": net, "contrast": label, **r})
                    print(f"    {label}: median Δ {r['median']:+.3f}  ({r['frac_pos']*100:.0f}% +, n={r['n']})")
        if "topo_benefit_with_ID" in net_c and "topo_benefit_without_ID" in net_c:
            inter = net_c["topo_benefit_with_ID"]["median"] - net_c["topo_benefit_without_ID"]["median"]
            contrasts.append({"network": net, "contrast": "interaction",
                              "median": inter, "n": None, "mean": None, "frac_pos": None})
            print(f"    interaction (with-ID − without-ID topo benefit): {inter:+.3f}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "table.csv", index=False)
    pd.DataFrame(contrasts).to_csv(OUT_DIR / "contrasts.csv", index=False)

    # RESULTS.md
    md = ["# Topology-Ablation 2x2 — Does network position help, and when?", "",
          "Controlled experiment on **stock NH cudalstm** (identical trainer; only "
          "`use_basin_id_encoding` and `static_attributes` differ). Tests whether the "
          "basin one-hot encoding makes topology features redundant.", "",
          "## Per-network median NSE", "",
          "| Network | L | L+T | L_noID | L_noID+T |", "|---|---|---|---|---|"]
    tdf = pd.DataFrame(rows)
    for net in networks:
        sub = tdf[tdf["network"] == net] if not tdf.empty else pd.DataFrame()
        if sub.empty:
            continue
        def g(c):
            r = sub[sub["condition"] == c]
            return f"{r.iloc[0]['median_NSE']:+.3f}" if not r.empty else "—"
        md.append(f"| {net} | {g('L')} | {g('L_T')} | {g('L_noID')} | {g('L_noID_T')} |")
    md += ["", "## Key contrasts (paired per-basin median ΔNSE)", "",
           "| Network | topo benefit WITH one-hot | topo benefit WITHOUT one-hot | interaction | encoding cost |",
           "|---|---|---|---|---|"]
    cdf = pd.DataFrame(contrasts)
    for net in networks:
        sub = cdf[cdf["network"] == net] if not cdf.empty else pd.DataFrame()
        if sub.empty:
            continue
        def gc(lbl):
            r = sub[sub["contrast"] == lbl]
            return f"{r.iloc[0]['median']:+.3f}" if not r.empty else "—"
        md.append(f"| {net} | {gc('topo_benefit_with_ID')} | {gc('topo_benefit_without_ID')} | "
                  f"{gc('interaction')} | {gc('encoding_cost')} |")
    md += ["", "**Pre-registered prediction:** `topo benefit WITHOUT one-hot` > 0 while "
           "`topo benefit WITH one-hot` ≈ 0 → the basin one-hot subsumes topology features. "
           "If confirmed, the paper's framing is *'network structure helps streamflow LSTMs "
           "only when the model cannot memorize per-basin identity'* — a controlled, "
           "theory-grounded contribution (cf. Kipf-Welling: graph structure helps most in "
           "the can't-memorize regime).", ""]
    (OUT_DIR / "RESULTS.md").write_text("\n".join(md))
    print(f"\nWrote table + contrasts + RESULTS.md to {OUT_DIR}")


if __name__ == "__main__":
    main()
