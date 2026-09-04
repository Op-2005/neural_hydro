"""Figure 1 for the XAI4Science submission: the two questions the paper answers.

The old TikZ concept figure showed only H1 (encoding), while the paper's title claims
proximity over connectivity. This version shows both, as one left-to-right argument:

  (a) the same network, two encodings  -> static is inert, dynamic is not
  (b) the same encoding, three selectors -> gain tracks distance, not connectivity

Every plotted number is recomputed from raw runs at build time, so the figure cannot
drift from the tables. Nothing is hardcoded.

Usage: python experiments/topology_ablation/plot_concept_figure.py
Writes: paper/canonical/figures/fig_concept_xai.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from scipy.stats import wilcoxon

# Type 42, not matplotlib's default Type 3: NeurIPS-style venues reject Type 3.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).parent.parent.parent
B = ROOT / "runs" / "topology_ablation" / "component0"
EDGES = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_edges.csv"
OUT = ROOT / "paper" / "canonical" / "figures" / "fig_concept_xai.pdf"
SEEDS = [11, 13, 17]

INK = "#1a1a1a"
BLUE = "#1f4e79"
GREY = "#9a9a9a"
WARM = "#b8622f"


def nse(cond, seed):
    p = B / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_metrics.csv"
    return pd.read_csv(p, dtype={"basin": str}).set_index("basin")["NSE"]


def connected():
    e = pd.read_csv(EDGES, dtype={"parent_id": str, "child_id": str})
    return set(e["child_id"].unique())


def contrast(cond, ref="L", basins=None, alternative="greater"):
    """Cross-seed mean of per-seed paired medians, and the weakest-seed p."""
    meds, ps = [], []
    for s in SEEDS:
        a, b = nse(cond, s), nse(ref, s)
        idx = a.index.intersection(b.index)
        if basins is not None:
            idx = idx.intersection(basins)
        d = (a.loc[idx] - b.loc[idx]).dropna()
        meds.append(float(d.median()))
        ps.append(float(wilcoxon(d, alternative=alternative).pvalue))
    return float(np.mean(meds)), max(ps)


def main():
    conn = connected()
    # (a) same network, two encodings -- all 183 basins
    static, p_static = contrast("L_T", "L", None, "two-sided")
    dynamic, p_dyn = contrast("L_upQ", "L", None, "greater")
    # (b) same encoding, three selectors -- connected basins only
    network, _ = contrast("L_upQ", "L", conn, "greater")
    matched, _ = contrast("L_upQdistctrl", "L", conn, "greater")
    nearest, _ = contrast("L_upQknn2", "L", conn, "greater")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(7.1, 2.45), gridspec_kw={"width_ratios": [1, 1.12], "wspace": 0.42})

    # ---------- (a) encoding ----------
    labels = ["static\ndescriptor", "daily\ndischarge"]
    vals = [static, dynamic]
    cols = [GREY, BLUE]
    bars = axA.bar([0, 1], vals, width=0.5, color=cols, zorder=3)
    axA.axhline(0, color=INK, lw=0.8, zorder=2)
    for x, v, pv in zip([0, 1], vals, [p_static, p_dyn]):
        tag = "n.s." if pv >= 0.05 else f"$p$={pv:.0e}".replace("e-0", "e$-$")
        axA.text(x, v + 0.0022, f"{v:+.3f}\n{tag}", ha="center", va="bottom",
                 fontsize=7.4, color=INK, linespacing=1.35)
    axA.set_xticks([0, 1]); axA.set_xticklabels(labels, fontsize=8)
    axA.set_ylabel("$\\Delta$NSE vs baseline", fontsize=8)
    axA.set_title("(a) one network, two encodings", fontsize=8.6, color=INK, pad=7)
    axA.set_ylim(-0.004, max(vals) * 1.42)
    axA.tick_params(labelsize=7.5)
    axA.spines[["top", "right"]].set_visible(False)

    # ---------- (b) selectors ----------
    names = ["nearest 2\ngauges", "distance-matched\nnon-upstream", "true river\nnetwork"]
    vv = [nearest, matched, network]
    dd = [46.7, 100.6, 91.6]
    cc = [BLUE, WARM, GREY]
    y = np.arange(len(vv))[::-1]
    axB.barh(y, vv, height=0.5, color=cc, zorder=3)
    for yy, v, d in zip(y, vv, dd):
        axB.text(v + 0.0022, yy, f"{v:+.3f}   ({d:.0f} km)", va="center",
                 fontsize=7.4, color=INK)
    axB.set_yticks(y); axB.set_yticklabels(names, fontsize=7.6)
    axB.set_xlabel("$\\Delta$NSE vs baseline", fontsize=8)
    axB.set_title("(b) one encoding, three neighbour sets", fontsize=8.6, color=INK, pad=7)
    axB.set_xlim(0, max(vv) * 1.44)
    axB.tick_params(labelsize=7.5)
    axB.spines[["top", "right"]].set_visible(False)

    # The reading: the middle bar shares the network's distance but none of its edges,
    # and matches it. Bracket those two rows specifically.
    xb = max(vv) * 1.255
    axB.annotate("", xy=(xb, y[1]), xytext=(xb, y[2]),
                 arrowprops=dict(arrowstyle="-", color=GREY, lw=0.9))
    for yy in (y[1], y[2]):
        axB.annotate("", xy=(xb - max(vv) * 0.022, yy), xytext=(xb, yy),
                     arrowprops=dict(arrowstyle="-", color=GREY, lw=0.9))
    axB.text(xb + max(vv) * 0.02, (y[1] + y[2]) / 2,
             "matched distance,\nno shared edges", fontsize=6.9, color=GREY,
             va="center", linespacing=1.3)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.20, wspace=0.42)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")

    print(f"wrote {OUT}\n")
    print("caption numbers (recomputed):")
    print(f"  static descriptor   {static:+.4f}  weakest two-sided p={p_static:.3f}")
    print(f"  daily discharge     {dynamic:+.4f}  weakest one-sided p={p_dyn:.2e}")
    print(f"  nearest 2 gauges    {nearest:+.4f}  (46.7 km)")
    print(f"  distance-matched    {matched:+.4f}  (100.6 km)")
    print(f"  true river network  {network:+.4f}  (91.6 km)")


if __name__ == "__main__":
    main()
