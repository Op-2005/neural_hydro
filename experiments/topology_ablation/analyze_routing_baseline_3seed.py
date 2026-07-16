"""Part 1 — 3-seed no-ML routing baseline. ZERO training.

Extends analyze_routing_baseline.py (single-seed) to all 3 seeds, using the fullspan L-sim
that exists for every seed. R1/R2 coefficients fit on TRAIN 1990-99 per seed, scored TEST.

Pre-reg: preregistration_baseline_completion_and_k2.md (Part 1).
Writes analysis/ROUTING_BASELINE_3SEED.md.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "topology_ablation" / "component0"
FEAT = ROOT / "experiments/topology_ablation/features/upstream_q_component0_lag1.p"
OUT = ROOT / "experiments/topology_ablation/analysis/ROUTING_BASELINE_3SEED.md"
TRAIN = ("1990-01-01", "1999-12-31")
TEST = ("2005-01-01", "2008-12-31")
SEEDS = [11, 13, 17]


def nse(o, s):
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    den = np.sum((o - o.mean()) ** 2)
    return (1 - np.sum((o - s) ** 2) / den) if (den > 0 and len(o) >= 10) else np.nan


def load_fullspan(seed):
    p = BASE / f"_Lfullspan_eval_seed{seed}" / "test" / "model_epoch030" / "test_results.p"
    res = pickle.load(open(p, "rb"))
    out = {}
    for b, d in res.items():
        xr = d["1D"]["xr"]
        idx = pd.to_datetime(xr["date"].values)
        out[b] = pd.DataFrame({"obs": pd.Series(xr["QObs(mm/d)_obs"].values.squeeze(), index=idx),
                               "Lsim": pd.Series(xr["QObs(mm/d)_sim"].values.squeeze(), index=idx)})
    return out


def cond_median(cond, seed, basins):
    p = BASE / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"basin": str}).set_index("basin")["NSE"].reindex(basins).dropna()


def routing_nse_for_seed(seed, feat):
    span = load_fullspan(seed)
    connected = [b for b in feat if np.abs(feat[b]["upstream_q"].values).mean() > 0]
    r1, r2 = {}, {}
    for b in connected:
        if b not in span:
            continue
        df = span[b].copy()
        upq = feat[b]["upstream_q"]; upq.index = pd.to_datetime(upq.index)
        df["upq"] = upq.reindex(df.index)
        df = df.dropna(subset=["obs", "upq", "Lsim"])
        tr, te = df.loc[TRAIN[0]:TRAIN[1]], df.loc[TEST[0]:TEST[1]]
        if len(tr) < 100 or len(te) < 100:
            continue
        c1, *_ = np.linalg.lstsq(np.column_stack([tr["upq"], np.ones(len(tr))]), tr["obs"].values, rcond=None)
        r1[b] = nse(te["obs"].values, te["upq"].values * c1[0] + c1[1])
        c2, *_ = np.linalg.lstsq(np.column_stack([tr["upq"], tr["Lsim"], np.ones(len(tr))]), tr["obs"].values, rcond=None)
        r2[b] = nse(te["obs"].values, te["upq"].values * c2[0] + te["Lsim"].values * c2[1] + c2[2])
    return pd.Series(r1).dropna(), pd.Series(r2).dropna()


def main():
    feat = pickle.load(open(FEAT, "rb"))
    rows = {"R1": [], "R2": [], "L": [], "L_upQ": [], "L_upQpred": []}
    per_seed_detail = []
    for s in SEEDS:
        r1, r2 = routing_nse_for_seed(s, feat)
        common = list(r1.index)
        L = cond_median("L", s, common)
        upQ = cond_median("L_upQ", s, common)
        upQp = cond_median("L_upQpred", s, common)
        rows["R1"].append(r1.median()); rows["R2"].append(r2.median())
        rows["L"].append(L.median() if L is not None else np.nan)
        rows["L_upQ"].append(upQ.median() if upQ is not None else np.nan)
        rows["L_upQpred"].append(upQp.median() if upQp is not None else np.nan)
        per_seed_detail.append((s, len(common), r1.median(), r2.median(),
                                L.median() if L is not None else np.nan,
                                upQ.median() if upQ is not None else np.nan,
                                upQp.median() if upQp is not None else np.nan))

    def ms(k):
        a = np.array(rows[k], float); a = a[np.isfinite(a)]
        return a.mean(), a.std()

    md = ["# Part 1 — 3-Seed No-ML Routing Baseline", "",
          "ZERO training. R1 (a·upQ+b) and R2 (a·upQ+c·L_sim+b) fit on TRAIN 1990-99, scored "
          "TEST 2005-08, per seed. Uses the fullspan L-sim available for all 3 seeds. "
          "Pre-reg: `preregistration_baseline_completion_and_k2.md` (Part 1).", "",
          "## Median test NSE, mean ± std across seeds [11,13,17] (connected basins)", "",
          "| predictor | mean ± std | per-seed (11/13/17) | note |", "|---|---|---|---|"]
    labels = {"R1": ("R1 — pure routing", "no ML, upstream only"),
              "R2": ("R2 — routing + local", "no ML, + L_sim"),
              "L": ("L (LSTM baseline)", "ML, no upstream"),
              "L_upQpred": ("L+upQ_pred (realizable)", "ML + predicted upstream"),
              "L_upQ": ("L+upQ (oracle)", "ML + observed upstream")}
    for k in ["R1", "R2", "L", "L_upQpred", "L_upQ"]:
        m, sd = ms(k)
        ps = "/".join(f"{v:+.3f}" if np.isfinite(v) else "n/a" for v in rows[k])
        md.append(f"| {labels[k][0]} | {m:+.4f} ± {sd:.4f} | {ps} | {labels[k][1]} |")
    md.append("")

    # verdict: LSTM beats R1 at every seed?
    beats = []
    for (s, n, r1m, r2m, Lm, upQm, upQpm) in per_seed_detail:
        ok = (np.isfinite(upQpm) and upQpm > r1m) and (np.isfinite(upQm) and upQm > r1m)
        beats.append(ok)
    md += ["## Pre-registered verdict", "",
           f"- realizable & oracle LSTM beat R1 (pure routing) at ALL seeds: **{all(beats)}** "
           f"(per-seed: {beats})", ""]
    r2_mean = ms("R2")[0]; upqp_mean = ms("L_upQpred")[0]
    md.append(f"- realizable ({upqp_mean:+.4f}) vs strong R2 baseline ({r2_mean:+.4f}): "
              f"margin **{upqp_mean-r2_mean:+.4f}** (multi-seed; prior single-seed was +0.010)")
    md.append("")
    md.append("**" + ("PASS — the LSTM-beats-naive-routing conclusion holds across all 3 seeds; "
                      "not seed-fragile." if all(beats) else
                      "FLIP at some seed — ML-earns-complexity is seed-sensitive; report.") + "**")

    OUT.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n-> wrote {OUT}")


if __name__ == "__main__":
    main()
