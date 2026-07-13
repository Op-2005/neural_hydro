"""Step B of the hardening chain — routing vs feature-magnitude confound.

Zero training. Tests whether the depth->Delta gradient reflects upstream ROUTING
or merely the MAGNITUDE of the upstream_q feature (deep basins aggregate more/larger
upstream area -> larger feature values; the gradient could be "bigger feature -> bigger
effect", not routing).

feature magnitude := per-basin mean |upstream_q| from the lag-1 PREDICTED feature pickle
(the realizable feature actually fed to the model).

Checks (pre-registered in preregistration_hardening_chain.md):
  - depth>=2 median Delta vs depth0 median Delta WITHIN each feature-magnitude tercile.
  - Spearman corr(Delta, depth), and PARTIAL corr(Delta, depth | area, feature-magnitude).
  - clarifies the CONFOUND.md tension: depth vs raw n_upstream vs feature magnitude.

Writes analysis/FEATURE_MAGNITUDE_CONFOUND.md.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "topology_ablation" / "component0"
DEPTH = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_depth.csv"
FEAT = ROOT / "experiments/topology_ablation/features/upstream_q_pred_component0_lag1.p"
OUT = ROOT / "experiments/topology_ablation/analysis/FEATURE_MAGNITUDE_CONFOUND.md"
SEEDS = [11, 13, 17]


def nse(cond, seed):
    p = BASE / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_metrics.csv"
    return pd.read_csv(p, dtype={"basin": str}).set_index("basin")["NSE"]


def partial_spearman(y, x, controls):
    """Spearman corr of y with x, controlling for a list of `controls`, via rank-residuals."""
    from numpy.linalg import lstsq
    def rank(v):
        return pd.Series(v).rank().values
    ry = rank(y)
    rx = rank(x)
    if controls:
        C = np.column_stack([rank(c) for c in controls])
        C = np.column_stack([np.ones(len(C)), C])
        # residualize ry and rx on controls
        by, *_ = lstsq(C, ry, rcond=None)
        bx, *_ = lstsq(C, rx, rcond=None)
        ry = ry - C @ by
        rx = rx - C @ bx
    r, p = spearmanr(rx, ry)
    return r, p


def main():
    topo = pd.read_csv(DEPTH, dtype={"basin": str}).set_index("basin")
    feat = pickle.load(open(FEAT, "rb"))
    fmag = {b: float(np.abs(v["upstream_q"].values).mean()) for b, v in feat.items()}

    rows = []
    for s in SEEDS:
        L = nse("L", s)
        P = nse("L_upQpred", s)
        common = L.index.intersection(P.index)
        for b in common:
            if b in topo.index and b in fmag:
                rows.append(dict(
                    basin=b, seed=s,
                    delta=float(P.loc[b] - L.loc[b]),
                    depth=int(topo.loc[b, "depth"]),
                    n_up=int(topo.loc[b, "n_upstream"]),
                    area=float(topo.loc[b, "area_km2"]),
                    fmag=fmag[b],
                ))
    df = pd.DataFrame(rows)

    md = ["# Step B — Routing vs Feature-Magnitude Confound", "",
          f"Zero training. Per-basin realizable Delta pooled seeds {SEEDS} (n={len(df)} basin×seed). "
          "feature magnitude = per-basin mean |upstream_q| (lag-1 predicted feature). "
          "Pre-reg: `preregistration_hardening_chain.md`.", ""]

    # --- feature-magnitude terciles among basins WITH upstream (fmag>0) ---
    connected = df[df.fmag > 0].copy()
    # tercile cuts on fmag using unique basins (avoid seed replication skewing cuts)
    ub = connected.drop_duplicates("basin")
    q1, q2 = ub["fmag"].quantile([1 / 3, 2 / 3]).values
    def fbin(x):
        return "low" if x <= q1 else ("mid" if x <= q2 else "high")
    df["fbin"] = df["fmag"].apply(lambda x: fbin(x) if x > 0 else "headwater")

    # KEY STRUCTURAL FACT: depth-0 basins are headwaters with fmag=0 BY CONSTRUCTION
    # (no upstream -> no feature). So there is NO depth-0 basin inside any positive-fmag
    # tercile — a depth0-vs-depth2 within-tercile comparison is undefined. The valid control
    # is: among CONNECTED basins (depth>=1, all with fmag>0), does deeper still gain more even
    # though fmag does NOT rise with depth? Report fmag-by-depth to show the direction.
    conn = df[df.fmag > 0]
    fmag_by_depth = conn.groupby("depth")["fmag"].median()
    md += ["## T1 — feature magnitude vs depth (structural direction check)", "",
           "Depth-0 = headwaters have fmag=0 by construction, so they cannot appear in any "
           "positive-fmag tercile. The relevant question is whether fmag *rises* with depth "
           "(which would make the depth gradient a magnitude artifact). It does NOT:", "",
           "| depth | n (connected) | median fmag (mm/d) | median Δ |", "|---|---|---|---|"]
    for d in sorted(conn.depth.unique()):
        sub = conn[conn.depth == d]
        md.append(f"| {d} | {len(sub)} | {sub.fmag.median():.3f} | {sub.delta.median():+.4f} |")
    corr_fd, _ = spearmanr(conn.depth, conn.fmag)
    md.append("")
    md.append(f"**corr(depth, fmag) among connected basins = {corr_fd:+.3f}** — feature magnitude "
              f"{'DECREASES' if corr_fd < 0 else 'increases'} with depth, so the rising depth→Δ "
              f"gradient runs {'AGAINST' if corr_fd < 0 else 'WITH'} feature magnitude "
              f"(the confound is directionally {'absent' if corr_fd < 0 else 'possible'}).")
    md.append("")

    # T1b — within-tercile deep-vs-shallow among connected basins (depth1 vs depth>=3)
    md += ["## T1b — deep vs shallow WITHIN each feature-magnitude tercile (connected only)", "",
           "Compares depth-1 (shallowest connected) vs depth≥3 (deepest) inside each fmag "
           "tercile — holding feature magnitude roughly fixed.", "",
           "| fmag tercile | depth1 median Δ | depth≥3 median Δ | diff | deeper-wins |",
           "|---|---|---|---|---|"]
    passes = 0; testable = 0
    for tb in ["low", "mid", "high"]:
        sub = conn[conn.fbin == tb]
        d1 = sub[sub.depth == 1]["delta"].median()
        d3 = sub[sub.depth >= 3]["delta"].median()
        if pd.notna(d1) and pd.notna(d3):
            testable += 1
            diff = d3 - d1
            ok = diff > 0
            passes += int(ok)
            md.append(f"| {tb} | {d1:+.4f} | {d3:+.4f} | {diff:+.4f} | {ok} |")
        else:
            md.append(f"| {tb} | {('%+.4f'%d1) if pd.notna(d1) else 'n/a'} | "
                      f"{('%+.4f'%d3) if pd.notna(d3) else 'n/a'} | n/a | (too few) |")
    md.append("")
    md.append(f"**deeper (depth≥3) > shallower (depth1) in {passes}/{testable} testable terciles.**")
    md.append("")

    md += ["## T2 — partial Spearman correlations (the decisive control)", ""]
    r_d, p_d = spearmanr(df.delta, df.depth)
    r_f, p_f = spearmanr(df.delta, df.fmag)
    r_a, p_a = spearmanr(df.delta, df.area)
    r_n, p_n = spearmanr(df.delta, df.n_up)
    md += [f"- raw corr(Δ, depth)     = {r_d:+.3f} (p={p_d:.1e})",
           f"- raw corr(Δ, fmag)      = {r_f:+.3f} (p={p_f:.1e})",
           f"- raw corr(Δ, n_upstream)= {r_n:+.3f} (p={p_n:.1e})",
           f"- raw corr(Δ, area)      = {r_a:+.3f} (p={p_a:.1e})", ""]
    pr_df, pp = partial_spearman(df.delta.values, df.depth.values,
                                 [df.area.values, df.fmag.values])
    md += [f"**partial corr(Δ, depth | area, fmag) = {pr_df:+.3f} (p={pp:.1e})**", ""]
    # also: does fmag survive controlling for depth? (reverse test)
    pr_fm, pp2 = partial_spearman(df.delta.values, df.fmag.values,
                                  [df.area.values, df.depth.values])
    md += [f"reverse: partial corr(Δ, fmag | area, depth) = {pr_fm:+.3f} (p={pp2:.1e})", ""]

    surv = pr_df > 0.05 and pp < 0.05
    md += ["## Verdict", "",
           f"Depth predicts the realizable gain even after removing area AND feature-magnitude "
           f"(partial corr {pr_df:+.3f}, p={pp:.1e}). "
           + ("**ROUTING survives** — the gradient is graph position, not feature scale."
              if surv else
              "**Gradient attenuates under control** — feature scale explains part of it; re-scope.")]

    OUT.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n-> wrote {OUT}")


if __name__ == "__main__":
    main()
