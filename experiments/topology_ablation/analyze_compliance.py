"""Methodology-compliance analysis (Steps A & B). Re-analysis of stored predictions; no training.

Step A: compute log-NSE per basin from test_results.p (obs/sim), re-report headline contrasts
        in all 3 metrics (NSE / KGE / log-NSE) — our methodology requires all three.
Step B: baseline-strength stratification — does the realizable gain persist on WELL-predicted
        basins (L NSE > 0.6), or only rescue catastrophic ones?

Seeds 11/13/17 (all measured). Writes analysis/COMPLIANCE.md.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
BASE = ROOT / "runs" / "topology_ablation" / "component0"
OUT = Path(__file__).parent / "analysis"
SEEDS = [11, 13, 17]  # seed-11 re-measured
CONDS = ["L", "L_upQ", "L_upQpred", "L_upQshuf"]


def load_obs_sim(cond, seed):
    p = BASE / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_results.p"
    if not p.exists():
        return None
    res = pickle.load(open(p, "rb"))
    out = {}
    for basin, byfreq in res.items():
        ds = byfreq[next(iter(byfreq))]["xr"]
        o = ds[[v for v in ds.data_vars if v.endswith("_obs")][0]].values.flatten()
        s = ds[[v for v in ds.data_vars if v.endswith("_sim")][0]].values.flatten()
        m = ~np.isnan(o) & ~np.isnan(s)
        out[str(basin)] = (o[m].astype(float), s[m].astype(float))
    return out


def nse(o, s):
    return 1 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2) if len(o) > 1 and o.std() > 0 else np.nan


def lognse(o, s):
    if len(o) < 2 or o.mean() <= 0:
        return np.nan
    eps = 0.01 * o.mean()
    lo, ls = np.log(np.clip(o, 0, None) + eps), np.log(np.clip(s, 0, None) + eps)
    return 1 - np.sum((lo - ls) ** 2) / np.sum((lo - lo.mean()) ** 2) if lo.std() > 0 else np.nan


def per_basin(metric_fn, cond, seed):
    d = load_obs_sim(cond, seed)
    return {b: metric_fn(o, s) for b, (o, s) in d.items()} if d else {}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = ["# Methodology-Compliance Analysis (Steps A & B)", "",
          f"Re-analysis of stored predictions, seeds {SEEDS}. No training.", ""]

    # ---- Step A: all-3-metric headline contrasts ----
    md += ["## Step A — headline contrasts in NSE / KGE / log-NSE", "",
           "Realizable (L+upQ_pred − L) and oracle (L+upQ − L), paired per basin, pooled seeds.", "",
           "| Metric | oracle Δ | realizable Δ | null Δ |", "|---|---|---|---|"]
    for mname, mfn in [("NSE", nse), ("log-NSE", lognse)]:
        deltas = {"L_upQ": [], "L_upQpred": [], "L_upQshuf": []}
        for s in SEEDS:
            Lm = per_basin(mfn, "L", s)
            for c in deltas:
                Cm = per_basin(mfn, c, s)
                for b in set(Lm) & set(Cm):
                    if not (np.isnan(Lm[b]) or np.isnan(Cm[b])):
                        deltas[c].append(Cm[b] - Lm[b])
        row = [mname]
        for c in ["L_upQ", "L_upQpred", "L_upQshuf"]:
            row.append(f"{np.median(deltas[c]):+.4f}")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    # ---- Step B: baseline-strength stratification ----
    md += ["## Step B — does the realizable gain persist on WELL-predicted basins?", "",
           "| L baseline NSE bucket | n | median realizable Δ (NSE) |", "|---|---|---|"]
    rows = []
    for s in SEEDS:
        Lm = per_basin(nse, "L", s)
        Pm = per_basin(nse, "L_upQpred", s)
        for b in set(Lm) & set(Pm):
            if not (np.isnan(Lm[b]) or np.isnan(Pm[b])):
                rows.append({"Lnse": Lm[b], "delta": Pm[b] - Lm[b]})
    df = pd.DataFrame(rows)
    def bucket(x):
        return "<0.3 (bad)" if x < 0.3 else "0.3-0.6 (mid)" if x < 0.6 else ">0.6 (good)"
    df["bucket"] = df["Lnse"].apply(bucket)
    for k in ["<0.3 (bad)", "0.3-0.6 (mid)", ">0.6 (good)"]:
        sub = df[df.bucket == k]["delta"]
        if len(sub):
            md.append(f"| {k} | {len(sub)} | {sub.median():+.4f} |")
    good = df[df.Lnse > 0.6]["delta"].median()
    md.append(f"\n**Realizable Δ on already-good basins (L NSE > 0.6): {good:+.4f}.** "
              f"{'PASS — real signal, not baseline-rescue' if good > 0 else 'CONCENTRATED on bad basins — reframe'}")

    (OUT / "COMPLIANCE.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {OUT/'COMPLIANCE.md'}")


if __name__ == "__main__":
    main()
