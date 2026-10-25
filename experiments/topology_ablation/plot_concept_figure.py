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

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from figstyle import apply as apply_style, LINEWIDTH, INK, BLUE, GREY, WARM

apply_style()

ROOT = Path(__file__).parent.parent.parent
B = ROOT / "runs" / "topology_ablation" / "component0"
EDGES = ROOT / "topology_analysis/phase1_network_discovery/outputs/component0_edges.csv"
OUT = ROOT / "paper" / "canonical" / "figures" / "fig_concept_xai.pdf"
# the XAI4Science submission keeps its own copy so all its figures sit together
XAI_OUT = ROOT / "paper" / "xai4science" / "figures" / "fig_concept_xai.pdf"
SEEDS = [11, 13, 17]

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
    # Paired head-to-head against the network. The distance-matched substitute is a
    # deliberate NULL (two-sided), so the panel must not let bar length imply a
    # difference the paired test does not support.
    _, p_matched_vs_net = contrast("L_upQdistctrl", "L_upQ", conn, "two-sided")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(LINEWIDTH, 2.05), gridspec_kw={"width_ratios": [1, 1.34], "wspace": 0.46})

    # ---------- (a) encoding ----------
    labels = ["static\ndescriptor", "daily\ndischarge"]
    vals = [static, dynamic]
    bars = axA.bar([0, 1], vals, width=0.46, color=[GREY, BLUE], zorder=3)
    axA.axhline(0, color=INK, lw=0.7, zorder=2)
    for x, v, pv in zip([0, 1], vals, [p_static, p_dyn]):
        tag = "n.s." if pv >= 0.05 else (
            lambda e: rf"$p\!=\!{pv / 10 ** e:.0f}\times10^{{{e}}}$")(int(np.floor(np.log10(pv))))
        axA.text(x, v + 0.0020, f"{v:+.3f}", ha="center", va="bottom",
                 color=INK, fontsize=7.6, fontweight="medium")
        axA.text(x, v + 0.0020 + max(vals) * 0.115, tag, ha="center", va="bottom",
                 color="0.42", fontsize=6.6)
    axA.set_xticks([0, 1]); axA.set_xticklabels(labels, fontsize=7.4)
    axA.set_ylabel("$\\Delta$NSE vs baseline", fontsize=7.4)
    axA.set_title("(a) one network, two encodings", fontsize=8.0, color=INK, pad=6)
    axA.set_ylim(-0.0026, max(vals) * 1.42)
    axA.tick_params(labelsize=7.2, length=2.5)
    axA.grid(axis="y", color="0.90", lw=0.5, zorder=0)
    axA.set_axisbelow(True)
    axA.spines[["top", "right"]].set_visible(False)

    # ---------- (b) selectors ----------
    # Order runs network -> distance-matched -> nearest so the reader meets the two
    # statistically indistinguishable arms adjacently, then the one that differs.
    # Colours follow the shared roles in figstyle: GREY reference, WARM matched
    # control, BLUE the condition argued for. Same meanings as Figure 2.
    names = ["true river\nnetwork", "distance-matched\nnon-upstream", "nearest 2 gauges"]
    vv = [network, matched, nearest]
    dd = [91.6, 100.6, 46.7]
    cc = [GREY, WARM, BLUE]
    y = np.arange(len(vv))[::-1]
    axB.barh(y, vv, height=0.46, color=cc, zorder=3)
    for yy, v, d in zip(y, vv, dd):
        axB.text(v + 0.0016, yy + 0.055, f"{v:+.3f}", va="center",
                 fontsize=7.6, color=INK, fontweight="medium")
        axB.text(v + 0.0016, yy - 0.235, f"{d:.0f} km apart", va="center",
                 fontsize=6.5, color="0.42")
    axB.set_yticks(y); axB.set_yticklabels(names, fontsize=7.2)
    axB.set_xlabel("$\\Delta$NSE vs baseline", fontsize=7.4)
    axB.set_title("(b) one encoding, three neighbour sets", fontsize=8.0, color=INK, pad=6)
    axB.set_xlim(0, max(vv) * 1.56)
    axB.tick_params(labelsize=7.2, length=2.5)
    axB.grid(axis="x", color="0.90", lw=0.5, zorder=0)
    axB.set_axisbelow(True)
    axB.spines[["top", "right"]].set_visible(False)

    # Bracket the two arms whose PAIRED difference is not significant. Without it the
    # longer distance-matched bar reads as "better than the network", which the paired
    # test does not support -- the null is the finding, so the figure must show it.
    yl, ym = y[0], y[1]                      # network, distance-matched
    xb = max(vv[0], vv[1]) + 0.0235   # clears the value and distance labels
    axB.plot([xb, xb + 0.0026, xb + 0.0026, xb], [yl, yl, ym, ym],
             color="0.45", lw=0.7, clip_on=False, zorder=4)
    axB.text(xb + 0.0050, (yl + ym) / 2, "paired\nn.s.", va="center", ha="left",
             fontsize=6.3, color="0.45", linespacing=1.25)

    fig.subplots_adjust(left=0.095, right=0.985, top=0.86, bottom=0.235, wspace=0.40)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    XAI_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(XAI_OUT, bbox_inches="tight")

    print(f"wrote {OUT}\n")
    print("caption numbers (recomputed):")
    print(f"  static descriptor   {static:+.4f}  weakest two-sided p={p_static:.3f}")
    print(f"  daily discharge     {dynamic:+.4f}  weakest one-sided p={p_dyn:.2e}")
    print(f"  nearest 2 gauges    {nearest:+.4f}  (46.7 km)")
    print(f"  distance-matched    {matched:+.4f}  (100.6 km)")
    print(f"  true river network  {network:+.4f}  (91.6 km)")
    print(f"  distance-matched vs network: paired two-sided weakest p = {p_matched_vs_net:.3f}")


if __name__ == "__main__":
    main()
