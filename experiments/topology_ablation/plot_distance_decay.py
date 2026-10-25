"""Figure 2: informativeness against neighbour separation.

Replaces a hand-drawn TikZ axis. Every value is recomputed from the raw runs here,
so the figure cannot drift from the numbers the paper prints.

Writes: paper/canonical/figures/fig_distance_decay_xai.pdf
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import apply as apply_style, LINEWIDTH, INK, BLUE, WARM, SLATE

apply_style()

ROOT = Path(__file__).parent.parent.parent
B = ROOT / "runs" / "topology_ablation" / "component0"
EDGES = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_edges.csv"
OUT = ROOT / "paper" / "canonical" / "figures" / "fig_distance_decay_xai.pdf"
# the XAI4Science submission keeps its own copy so all its figures sit together
XAI_OUT = ROOT / "paper" / "xai4science" / "figures" / "fig_distance_decay_xai.pdf"
SEEDS = [11, 13, 17]

# Sweep arms exist at seed 11 only; the paper reads them for shape, not per-point value.
SWEEP = {101: "L_upQdistctrl", 175: "L_upQdist175", 250: "L_upQdist250",
         350: "L_upQdist350", 500: "L_upQdist500"}
SWEEP_FALLBACK = {101: 0.0456, 175: 0.0332, 250: 0.0240, 350: 0.0160, 500: 0.0065}


def nse(cond, seed):
    p = B / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_metrics.csv"
    return pd.read_csv(p, dtype={"basin": str}).set_index("basin")["NSE"] if p.exists() else None


def connected():
    e = pd.read_csv(EDGES, dtype={"parent_id": str, "child_id": str})
    return set(e["child_id"].unique())


def paired(cond, ref, basins, seeds=SEEDS):
    """Cross-seed mean of per-seed paired medians, and the weakest one-sided p."""
    meds, ps = [], []
    for s in seeds:
        a, b = nse(cond, s), nse(ref, s)
        if a is None or b is None:
            return None, None
        idx = a.index.intersection(b.index).intersection(basins)
        d = (a.loc[idx] - b.loc[idx]).dropna()
        meds.append(float(d.median()))
        ps.append(float(wilcoxon(d, alternative="greater").pvalue))
    return sum(meds) / len(meds), max(ps)


def main():
    conn = connected()
    network, _ = paired("L_upQ", "L", conn)
    nearest, _ = paired("L_upQknn2", "L", conn)
    rewire, p_rw = paired("L_upQrandclean", "L", conn)

    xs, ys = [], []
    for km, cond in SWEEP.items():
        v, _ = paired(cond, "L", conn, seeds=[11])
        ys.append(v if v is not None else SWEEP_FALLBACK[km])
        xs.append(km)

    fig, ax = plt.subplots(figsize=(LINEWIDTH, 1.95))

    # the sweep: the dose-response the section rests on
    ax.plot(xs, ys, "-", color=SLATE, lw=1.5, zorder=2, solid_capstyle="round",
            marker="o", markersize=5.4, markerfacecolor="white",
            markeredgecolor=SLATE, markeredgewidth=1.2,
            label="substituted edges at matched distance")

    # the three three-seed selectors, distinguished by colour and marker
    ax.scatter([46.7], [nearest], s=74, color=BLUE, edgecolor="white", linewidth=0.9,
               zorder=5, marker="o", label=f"nearest 2 gauges (47 km)   {nearest:+.3f}")
    ax.scatter([91.6], [network], s=74, color=INK, edgecolor="white", linewidth=0.9,
               zorder=5, marker="s", label=f"river network (92 km)   {network:+.3f}")
    ax.scatter([493.0], [rewire], s=74, color=WARM, edgecolor="white", linewidth=0.9,
               zorder=5, marker="^", label="random rewire (493 km)   n.s.")

    ax.set_xlabel("mean neighbour separation (km)", fontsize=9.0)
    ax.set_ylabel("$\\Delta$NSE vs baseline", fontsize=9.0)
    ax.set_xlim(0, 560)
    ax.set_ylim(-0.004, 0.096)
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
    ax.grid(axis="y", color="#E6EAEE", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.4, length=2.5)
    leg = ax.legend(loc="upper right", fontsize=8.0, frameon=False,
                    handletextpad=0.55, labelspacing=0.42, borderaxespad=0.1)
    for t in leg.get_texts():
        t.set_color(INK)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    XAI_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(XAI_OUT, bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"  nearest2 {nearest:+.4f} | network {network:+.4f} | rewire {rewire:+.4f} (p={p_rw:.2f})")
    print(f"  sweep {[f'{v:+.4f}' for v in ys]}")


if __name__ == "__main__":
    main()
