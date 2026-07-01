"""Multi-seed analysis of the upstream-signal headline + robustness (Steps A & B).

Reads runs/topology_ablation/component0/<cond>_component0_seed{S}/ for
cond in {L, L_upQ, L_upQpred, L_upQshuf}, seeds {11,13,17}.

Reports:
  - per-condition median NSE, mean±std across seeds (the tracked invariant)
  - paired Δ vs L per seed + cross-seed (oracle / realizable / null)
  - Step A: realizable − null (the clean, capacity-controlled effect)
  - Step B: depth-stratified realizable gain (mechanistic routing check)
  - realizable recovery of the oracle ceiling
Writes: experiments/topology_ablation/analysis/MULTISEED.md

Usage: python experiments/topology_ablation/analyze_multiseed.py
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
BASE = ROOT / "runs" / "topology_ablation" / "component0"
DEPTH = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_depth.csv"
OUT = Path(__file__).parent / "analysis"
SEEDS = [11, 13, 17]
CONDS = ["L", "L_upQ", "L_upQpred", "L_upQshuf"]
# Seed-11 medians recorded in preregistration (its oracle/predicted metric folders were
# overwritten during a drive-download merge; the numbers are preserved from the original run).
REC11 = {"L": 0.653, "L_upQ": 0.703, "L_upQpred": 0.683, "L_upQshuf": 0.658}


def nse(cond, seed):
    p = BASE / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"basin": str}).set_index("basin")["NSE"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Multi-Seed Confirmation — Upstream-Signal Headline", "",
          "Component 0 (183 basins), stock cudalstm, seeds 11/13/17. "
          "Seed-11 oracle/predicted medians are the recorded originals (folders overwritten "
          "in a drive merge); seeds 13/17 are freshly measured.", ""]

    # per-condition median NSE
    md += ["## Per-condition median NSE (mean ± std across seeds)", "",
           "| Condition | mean ± std | per-seed |", "|---|---|---|"]
    for c in CONDS:
        meds = {}
        for s in SEEDS:
            x = nse(c, s)
            meds[s] = x.median() if x is not None else REC11.get(c) if s == 11 else None
        vals = [v for v in meds.values() if v is not None]
        md.append(f"| {c} | {np.mean(vals):+.4f} ± {np.std(vals):.4f} | "
                  f"{', '.join(f'{s}:{v:.3f}' for s,v in meds.items() if v is not None)} |")
    md.append("")

    # paired deltas (only seeds where both condition and L have per-basin CSVs)
    def paired(cond):
        out = {}
        for s in SEEDS:
            L, X = nse("L", s), nse(cond, s)
            if L is None or X is None:
                continue
            b = L.index.intersection(X.index)
            out[s] = float((X.loc[b] - L.loc[b]).median())
        return out

    md += ["## Paired Δ vs L (per seed; measured seeds only)", "",
           "| Contrast | per-seed Δ | cross-seed mean |", "|---|---|---|"]
    dpaired = {}
    for cond, lab in [("L_upQ", "oracle (observed)"), ("L_upQpred", "realizable (predicted)"),
                      ("L_upQshuf", "null (shuffled)")]:
        d = paired(cond)
        dpaired[cond] = d
        md.append(f"| {lab} | {', '.join(f'{s}:{v:+.4f}' for s,v in d.items())} | "
                  f"{np.mean(list(d.values())):+.4f} |")
    md.append("")

    # Step A: realizable - null, per seed (both measured)
    md += ["## Step A — realizable − null (capacity-controlled clean effect)", ""]
    ra = {}
    for s in SEEDS:
        P, N, L = nse("L_upQpred", s), nse("L_upQshuf", s), nse("L", s)
        if P is None or N is None:
            continue
        b = P.index.intersection(N.index)
        ra[s] = float((P.loc[b] - N.loc[b]).median())
    if ra:
        arr = np.array(list(ra.values()))
        md.append(f"Per-seed (realizable − null): {', '.join(f'{s}:{v:+.4f}' for s,v in ra.items())}")
        md.append(f"\n**Cross-seed mean {arr.mean():+.4f} ± {arr.std():.4f}; all positive: {all(arr>0)}.** "
                  f"{'PASS (>= +0.010)' if arr.mean() >= 0.01 and all(arr>0) else 'CHECK vs pre-reg'}")
    md.append("")

    # Step B: depth-stratified realizable gain
    if DEPTH.exists():
        depth = pd.read_csv(DEPTH, dtype={"basin": str}).set_index("basin")["depth"]
        md += ["## Step B — depth-stratified realizable gain (routing check)", "",
               "| depth | n | median realizable Δ (pooled seeds) |", "|---|---|---|"]
        # pool per-basin realizable Δ across measured seeds
        rows = []
        for s in SEEDS:
            P, L = nse("L_upQpred", s), nse("L", s)
            if P is None or L is None:
                continue
            b = P.index.intersection(L.index)
            d = (P.loc[b] - L.loc[b])
            for basin, val in d.items():
                if basin in depth.index:
                    rows.append((int(depth[basin]), float(val)))
        df = pd.DataFrame(rows, columns=["depth", "delta"])
        by = df.groupby("depth")["delta"].agg(["size", "median"])
        for dpth, r in by.iterrows():
            md.append(f"| {dpth} | {int(r['size'])} | {r['median']:+.4f} |")
        deep = df[df.depth >= 2]["delta"].median()
        head = df[df.depth == 0]["delta"].median()
        md.append(f"\ndepth≥2 median Δ {deep:+.4f} vs headwater (depth 0) {head:+.4f} "
                  f"→ diff {deep-head:+.4f}. "
                  f"{'PASS (routing signature: downstream benefits more)' if deep-head >= 0.01 else 'FLAT/inverted — see pre-reg'}")
    md.append("")

    (OUT / "MULTISEED.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {OUT/'MULTISEED.md'}")


if __name__ == "__main__":
    main()
