"""Figure J: how much of the neighbour signal the basin's own discharge already carries.

Recomputes every value from the stored run metrics at plot time, so the figure cannot drift
from VERIFIED_NUMBERS.md. Style matches the flow-band figure: each seed is a point, the seed
range a vertical line, the cross-seed mean a horizontal bar.

  (a) paired median Delta NSE vs L across the 150 connected basins, for the neighbour
      selectors alone, the basin's own discharge alone, and the two combined;
  (b) the change when neighbour discharge is added on top of the basin's own discharge.

Usage:  python experiments/topology_ablation/plot_own_lag.py
Writes: paper/xai4science/figures/fig_own_lag_xai.pdf
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import apply as apply_style, LINEWIDTH, BLUE, SLATE

apply_style()

ROOT = Path(__file__).parent.parent.parent
RUNS = ROOT / "runs/topology_ablation/component0"
EDGES = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_edges.csv"
OUT = ROOT / "paper" / "xai4science" / "figures" / "fig_own_lag_xai.pdf"
SEEDS = [11, 13, 17]


def nse(cond, seed):
    p = RUNS / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_metrics.csv"
    return pd.read_csv(p, dtype={"basin": str}).set_index("basin")["NSE"]


def paired_medians(cond, ref, basins):
    out = []
    for s in SEEDS:
        a, b = nse(cond, s), nse(ref, s)
        idx = a.index.intersection(b.index).intersection(basins)
        out.append(float((a.loc[idx] - b.loc[idx]).dropna().median()))
    return np.array(out)


def draw(ax, items):
    for x, (vals, colour) in enumerate(items):
        ax.plot([x, x], [vals.min(), vals.max()], color=colour, lw=1.0, alpha=0.55,
                zorder=2, solid_capstyle="round")
        ax.scatter(np.full_like(vals, x), vals, s=13, color=colour, zorder=3, linewidths=0)
        ax.plot([x - 0.17, x + 0.17], [vals.mean()] * 2, color=colour, lw=2.0, zorder=4,
                solid_capstyle="butt")
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.set_xticks(range(len(items)))
    ax.margins(x=0.12)


def main():
    conn = set(pd.read_csv(EDGES, dtype={"child_id": str})["child_id"].str.zfill(8))

    panel_a = [("River\nnetwork", paired_medians("L_upQ", "L", conn), SLATE),
               ("Nearest\n2 gauges", paired_medians("L_upQknn2", "L", conn), SLATE),
               ("Own\ndischarge", paired_medians("L_selfQ", "L", conn), BLUE),
               ("Own +\nnetwork", paired_medians("L_selfQ_upQ", "L", conn), BLUE),
               ("Own +\nnearest 2", paired_medians("L_selfQ_knn2", "L", conn), BLUE)]
    panel_b = [("+ network", paired_medians("L_selfQ_upQ", "L_selfQ", conn), BLUE),
               ("+ nearest 2", paired_medians("L_selfQ_knn2", "L_selfQ", conn), BLUE)]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(LINEWIDTH, 2.1),
                                   gridspec_kw={"width_ratios": [2.6, 1]})
    draw(axa, [(v, c) for _, v, c in panel_a])
    axa.set_xticklabels([n for n, _, _ in panel_a])
    axa.set_ylabel(r"$\Delta$NSE vs baseline")
    axa.set_title("(a) Gain over the baseline", loc="left")

    draw(axb, [(v, c) for _, v, c in panel_b])
    axb.set_xticklabels([n for n, _, _ in panel_b])
    axb.set_ylabel(r"$\Delta$NSE vs own discharge")
    axb.set_title("(b) Added on top of own discharge", loc="left")
    lim = max(abs(v).max() for _, v, _ in panel_b) * 1.6
    axb.set_ylim(-lim, lim)

    fig.tight_layout(w_pad=1.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")

    # Diagnostics for an honest caption: printed, never hardcoded into the figure.
    print(f"wrote {OUT}")
    for name, vals, _ in panel_a + panel_b:
        name = name.replace("\n", " ")
        print(f"  {name:24s} {' / '.join(f'{v:+.4f}' for v in vals)}  mean {vals.mean():+.4f}")


if __name__ == "__main__":
    main()
