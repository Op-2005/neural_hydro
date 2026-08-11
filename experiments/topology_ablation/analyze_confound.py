"""Depth-gradient confound check: is the realizable upstream-Q gain driven by upstream
contribution (routing) or by basin size (area)?

Pre-registered in preregistration_confound_check.md. Zero compute — re-analyzes committed
runs. Per-basin realizable Δ = (L+upQ_pred − L) pooled over all 3 measured seeds 11/13/17.

Tests:
  T1 gain vs n_upstream buckets
  T2 gain vs area tercile
  T3 depth-2+ vs depth-0 gain within each area tercile (the partial control)
  T4 spearman corr of Δ with n_upstream vs area

Writes: experiments/topology_ablation/analysis/CONFOUND.md
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent.parent
BASE = ROOT / "runs" / "topology_ablation" / "component0"
DEPTH = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_depth.csv"
OUT = Path(__file__).parent / "analysis"
SEEDS = [11, 13, 17]  # all 3 realizable seeds measured (seed-11 re-run 2026-07-01)


def nse(cond, seed):
    p = BASE / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_metrics.csv"
    return pd.read_csv(p, dtype={"basin": str}).set_index("basin")["NSE"] if p.exists() else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    topo = pd.read_csv(DEPTH, dtype={"basin": str}).set_index("basin")
    # Eq.1 (longest-path) depth is authoritative; overlay it onto the attribute table,
    # which supplies the other columns (n_upstream, area_km2, ...).
    _eq1 = DEPTH.parent / "component0_depth_eq1.csv"
    if _eq1.is_file():
        _d = pd.read_csv(_eq1, dtype={"basin": str}).set_index("basin")["depth_eq1"]
        topo["depth"] = _d.reindex(topo.index)

    # pooled per-basin realizable Δ across seeds 13/17
    rows = []
    for s in SEEDS:
        P, L = nse("L_upQpred", s), nse("L", s)
        if P is None or L is None:
            continue
        b = P.index.intersection(L.index)
        for basin in b:
            if basin in topo.index:
                rows.append({"basin": basin, "delta": float(P[basin] - L[basin]),
                             "depth": int(topo.loc[basin, "depth"]),
                             "n_up": int(topo.loc[basin, "n_upstream"]),
                             "area": float(topo.loc[basin, "area_km2"])})
    df = pd.DataFrame(rows)

    md = ["# Depth-Gradient Confound Check — routing (n_upstream) vs size (area)", "",
          f"Per-basin realizable Δ pooled over seeds {SEEDS} (n={len(df)} basin×seed). "
          "Pre-reg: `preregistration_confound_check.md`.", ""]

    # T1 — n_upstream buckets
    def bucket_nup(n):
        return "0" if n == 0 else "1-2" if n <= 2 else "3-5" if n <= 5 else "6+"
    df["nup_b"] = df["n_up"].apply(bucket_nup)
    md += ["## T1 — gain vs n_upstream (the routing variable)", "",
           "| n_upstream | n | median Δ |", "|---|---|---|"]
    order = ["0", "1-2", "3-5", "6+"]
    t1 = df.groupby("nup_b")["delta"].agg(["size", "median"]).reindex(order).dropna()
    for k, r in t1.iterrows():
        md.append(f"| {k} | {int(r['size'])} | {r['median']:+.4f} |")
    mono = list(t1["median"]) == sorted(t1["median"])
    md.append(f"\nMonotonic increase with n_upstream: {mono}. headwaters(0) median {t1.loc['0','median']:+.4f}.")

    # T2 — area terciles
    q = df["area"].quantile([1/3, 2/3]).values
    df["area_t"] = np.where(df.area <= q[0], "small", np.where(df.area <= q[1], "mid", "large"))
    md += ["", "## T2 — gain vs area (the confound)", "",
           f"| area tercile (cuts {q[0]:.0f}/{q[1]:.0f} km²) | n | median Δ |", "|---|---|---|"]
    t2 = df.groupby("area_t")["delta"].agg(["size", "median"]).reindex(["small", "mid", "large"])
    for k, r in t2.iterrows():
        md.append(f"| {k} | {int(r['size'])} | {r['median']:+.4f} |")
    area_spread = t2["median"].max() - t2["median"].min()
    md.append(f"\nArea-tercile spread {area_spread:+.4f}.")

    # T3 — depth-2+ vs depth-0 WITHIN each area tercile
    md += ["", "## T3 — depth gradient within area terciles (the partial control)", "",
           "| area tercile | depth0 median Δ | depth≥2 median Δ | diff |", "|---|---|---|---|"]
    passes = 0
    for t in ["small", "mid", "large"]:
        sub = df[df.area_t == t]
        d0 = sub[sub.depth == 0]["delta"].median()
        d2 = sub[sub.depth >= 2]["delta"].median()
        diff = (d2 - d0) if (pd.notna(d0) and pd.notna(d2)) else np.nan
        if pd.notna(diff) and diff >= 0.01:
            passes += 1
        md.append(f"| {t} | {d0:+.4f} | {d2:+.4f} | {diff:+.4f} |")
    md.append(f"\n**depth≥2 > depth0 by ≥+0.01 in {passes}/3 area terciles.** "
              f"{'PASS — routing survives area control' if passes >= 2 else 'CONFOUNDED — depth effect tied to area'}")

    # T4 — correlations
    c_nup, _ = spearmanr(df.delta, df.n_up)
    c_area, _ = spearmanr(df.delta, df.area)
    c_depth, _ = spearmanr(df.delta, df.depth)
    md += ["", "## T4 — Spearman corr of Δ with each variable", "",
           f"- corr(Δ, n_upstream) = **{c_nup:+.3f}**",
           f"- corr(Δ, depth)      = {c_depth:+.3f}",
           f"- corr(Δ, area)       = {c_area:+.3f}",
           f"\n**{'n_upstream is the stronger predictor' if abs(c_nup) > abs(c_area) else 'area competes — flag'}** "
           f"(|{c_nup:.3f}| vs |{c_area:.3f}|)."]

    (OUT / "CONFOUND.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {OUT/'CONFOUND.md'}")


if __name__ == "__main__":
    main()
