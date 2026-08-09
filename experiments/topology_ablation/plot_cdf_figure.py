"""Regenerate paper Figure 2 (empirical CDF of per-basin test NSE).

Source of truth: the stored per-basin test_metrics.csv for L / oracle / realizable, seeds
[11, 13, 17]. Pools the per-basin NSE across the three seeds (549 basin x seed points per
condition) so the figure is 3-seed, not seed-11 only.

Reads the data on disk (no transcription): the CDF needs the full per-basin distribution.

Usage:  python experiments/topology_ablation/plot_cdf_figure.py
Writes: paper/figures/fig_cdf_nse.pdf   (and prints diagnostics for an honest caption)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "topology_ablation" / "component0"
OUT = ROOT / "paper" / "figures" / "fig_cdf_nse.pdf"
SEEDS = [11, 13, 17]
CONDS = [
    ("L", "L (baseline)", "#999999"),
    ("L_upQpred", "realizable", "#4676c4"),
    ("L_upQ", "oracle", "#1a1a1a"),
]
XMIN = 0.0  # focus on the bulk; fraction below XMIN is annotated, not hidden silently


def nse_pooled(cond):
    vals = []
    for s in SEEDS:
        p = BASE / f"{cond}_component0_seed{s}" / "test" / "model_epoch030" / "test_metrics.csv"
        vals += list(pd.read_csv(p, dtype={"basin": str})["NSE"].values)
    return np.array([v for v in vals if np.isfinite(v)])


def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def main():
    data = {c: nse_pooled(c) for c, _, _ in CONDS}

    # ---- diagnostics for an honest caption ----
    grid = np.linspace(XMIN, 1.0, 200)
    def frac_below(x, t):  # ECDF value at t
        return np.mean(x <= t)
    print("condition   median   frac<0   frac<0.5   (pooled 3 seeds, n per cond)")
    for c, lbl, _ in CONDS:
        x = data[c]
        print(f"  {lbl:<12} {np.median(x):+.3f}   {np.mean(x<0):.3f}    {np.mean(x<0.5):.3f}   (n={len(x)})")
    L, rz, orc = data["L"], data["L_upQpred"], data["L_upQ"]
    # Does realizable ECDF sit strictly left of (>= at every quantile of) L? i.e. stochastic dominance
    rz_dom = all(frac_below(rz, t) <= frac_below(L, t) + 1e-9 for t in grid)
    orc_dom = all(frac_below(orc, t) <= frac_below(L, t) + 1e-9 for t in grid)
    orc_ge_rz = all(frac_below(orc, t) <= frac_below(rz, t) + 1e-9 for t in grid)
    print(f"realizable stochastically dominates L over [{XMIN},1]? {rz_dom}")
    print(f"oracle stochastically dominates L over [{XMIN},1]?     {orc_dom}")
    print(f"oracle dominates realizable over [{XMIN},1]?           {orc_ge_rz}")

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for c, lbl, color in CONDS:
        xs, ys = ecdf(data[c])
        ax.step(xs, ys, where="post", color=color, lw=1.8, label=lbl, zorder=3)
    ax.set_xlim(XMIN, 1.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("per-basin test NSE")
    ax.set_ylabel("cumulative fraction of basins")
    ax.grid(alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
