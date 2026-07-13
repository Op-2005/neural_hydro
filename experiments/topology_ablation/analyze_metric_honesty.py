"""Step C of the hardening chain — metric-honesty pass.

Zero training. Reads stored per-timestep obs/sim from test_results.p.

(1) log-NSE eps-sensitivity: recompute the realizable log-NSE Delta for
    eps in {1e-2, 1e-3, 1e-4} x (per-basin mean observed flow). Is the +0.027
    headline stable, or an artifact of one eps choice?
(2) KGE decomposition: split KGE into (r, beta, gamma) per condition and locate
    which component drives the seed-13 KGE dip (the disclosed weak spot).

Writes analysis/METRIC_HONESTY.md.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "topology_ablation" / "component0"
OUT = ROOT / "experiments" / "topology_ablation" / "analysis" / "METRIC_HONESTY.md"
SEEDS = [11, 13, 17]
CONDS = {"L": "L", "L_upQpred": "realizable", "L_upQ": "oracle", "L_upQshuf": "null"}


def load_obs_sim(cond, seed):
    p = BASE / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_results.p"
    if not p.exists():
        return None  # e.g. L_upQ seed11 test_results.p lost in the drive merge (metrics.csv survives)
    res = pickle.load(open(p, "rb"))
    out = {}
    for b, d in res.items():
        xr = d["1D"]["xr"]
        o = xr["QObs(mm/d)_obs"].values.squeeze()
        s = xr["QObs(mm/d)_sim"].values.squeeze()
        m = np.isfinite(o) & np.isfinite(s)
        out[b] = (o[m], s[m])
    return out


def lognse(o, s, eps_frac):
    # eps scaled to the basin's own mean observed flow (defensible, unit-aware)
    mo = np.mean(np.clip(o, 0, None))
    eps = eps_frac * max(mo, 1e-6)
    lo = np.log(np.clip(o, 0, None) + eps)
    ls = np.log(np.clip(s, 0, None) + eps)
    denom = np.sum((lo - lo.mean()) ** 2)
    if denom <= 0:
        return np.nan
    return 1.0 - np.sum((lo - ls) ** 2) / denom


def kge_components(o, s):
    if o.std() == 0 or s.std() == 0 or o.mean() == 0:
        return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(o, s)[0, 1]
    beta = s.mean() / o.mean()          # bias ratio
    gamma = (s.std() / s.mean()) / (o.std() / o.mean())  # variability ratio (CV ratio)
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    return kge, r, beta, gamma


def main():
    md = ["# Step C — Metric-Honesty Pass (log-NSE eps-sensitivity + KGE decomposition)", "",
          "Zero training. Per-timestep obs/sim from `test_results.p`. "
          "Pre-reg: `preregistration_hardening_chain.md`.", ""]

    # ---------- (1) log-NSE eps sweep ----------
    md += ["## C1 — log-NSE realizable Δ vs eps (stability of the +0.027 headline)", "",
           "eps = frac × (per-basin mean observed flow). Realizable Δ = median over basins of "
           "logNSE(L+upQ_pred) − logNSE(L), pooled seeds.", "",
           "| eps frac | realizable median Δ (log-NSE) | null median Δ | oracle median Δ |",
           "|---|---|---|---|"]
    eps_rows = []
    for ef in [1e-2, 1e-3, 1e-4]:
        rz, nz, oz = [], [], []
        for s in SEEDS:
            Lo = load_obs_sim("L", s)
            Pz = load_obs_sim("L_upQpred", s)
            Nz = load_obs_sim("L_upQshuf", s)
            Oz = load_obs_sim("L_upQ", s)  # may be None (seed11 pickle lost)
            if Lo is None:
                continue
            for b in Lo:
                if Pz and b in Pz:
                    rz.append(lognse(*Pz[b], ef) - lognse(*Lo[b], ef))
                if Nz and b in Nz:
                    nz.append(lognse(*Nz[b], ef) - lognse(*Lo[b], ef))
                if Oz and b in Oz:
                    oz.append(lognse(*Oz[b], ef) - lognse(*Lo[b], ef))
        rz = np.array(rz); rz = rz[np.isfinite(rz)]
        nz = np.array(nz); nz = nz[np.isfinite(nz)]
        oz = np.array(oz); oz = oz[np.isfinite(oz)]
        eps_rows.append((ef, np.median(rz), np.median(nz), np.median(oz)))
        md.append(f"| {ef:.0e} | {np.median(rz):+.4f} | {np.median(nz):+.4f} | {np.median(oz):+.4f} |")
    md.append("")
    signs = [r[1] > 0 for r in eps_rows]
    md.append(f"**log-NSE realizable Δ positive at all eps: {all(signs)}** "
              f"(range {min(r[1] for r in eps_rows):+.4f} to {max(r[1] for r in eps_rows):+.4f}). "
              + ("PASS — headline is not an eps artifact." if all(signs)
                 else "FLIPS — re-scope the log-NSE claim."))
    md.append("")

    # ---------- (2) KGE decomposition, per seed ----------
    md += ["## C2 — KGE decomposition: where does the seed-13 dip live?", "",
           "Median over basins of each KGE component per condition, per seed. "
           "KGE weakness should localize to r (timing), β (bias), or γ (variability).", ""]
    for s in SEEDS:
        md += [f"### seed {s}", "",
               "| condition | median KGE | median r | median β (bias) | median γ (var) |",
               "|---|---|---|---|---|"]
        comp = {}
        for cond, label in CONDS.items():
            data = load_obs_sim(cond, s)
            if data is None:
                md.append(f"| {label} | (results.p missing) | — | — | — |")
                continue
            ks, rs, bs, gs = [], [], [], []
            for b, (o, sm) in data.items():
                k, r, be, ga = kge_components(o, sm)
                if np.isfinite(k):
                    ks.append(k); rs.append(r); bs.append(be); gs.append(ga)
            comp[cond] = (np.median(ks), np.median(rs), np.median(bs), np.median(gs))
            md.append(f"| {label} | {np.median(ks):+.4f} | {np.median(rs):+.4f} | "
                      f"{np.median(bs):+.4f} | {np.median(gs):+.4f} |")
        # realizable - L deltas per component
        dK = comp["L_upQpred"][0] - comp["L"][0]
        dr = comp["L_upQpred"][1] - comp["L"][1]
        dbeta = comp["L_upQpred"][2] - comp["L"][2]
        dgamma = comp["L_upQpred"][3] - comp["L"][3]
        md += ["",
               f"realizable − L: ΔKGE {dK:+.4f} | Δr {dr:+.4f} | Δβ {dbeta:+.4f} | Δγ {dgamma:+.4f}",
               ""]

    OUT.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n-> wrote {OUT}")


if __name__ == "__main__":
    main()
