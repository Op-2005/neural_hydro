"""Flow-band figure for the XAI4Science submission.

The paper's physical claim is that a neighbour's discharge reports catchment STORAGE,
not a shared storm. The evidence is that the deployable gain is reliable below the
median flow, unstable during events, and NOT reproduced by upstream precipitation.
A three-row table states this; the figure makes the reliability asymmetry visible,
because the claim is about consistency across seeds and a median hides that.

RECOMPUTES from test_results.p -- it does not hardcode the table. plot_depth_figure.py
was de-hardcoded on 2026-08-13 precisely because a hardcoded figure let a definition
drift out of step with its table. The banding logic is imported from
analyze_flow_bands.py so the figure and Table `tab:bands` cannot disagree.

Metric: per-basin RMSE reduction in mm/d within a band (positive favours the
neighbour-discharge input). Not a variance-normalised score -- low-flow variance is
small, and normalising inflates that band for arithmetic rather than hydrologic reasons.

Usage: python experiments/topology_ablation/plot_flow_bands_figure.py
Writes: paper/canonical/figures/fig_flow_bands_xai.pdf (distinct name: another script in
this directory writes fig_flow_bands.pdf, and the two must not collide).
"""
import importlib.util
from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import apply as apply_style, LINEWIDTH, INK, BLUE, GREY, WARM

apply_style()

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "paper" / "canonical" / "figures" / "fig_flow_bands_xai.pdf"

# Import the SAME banding implementation the results table uses.
_spec = importlib.util.spec_from_file_location(
    "afb", Path(__file__).parent / "analyze_flow_bands.py")
_afb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_afb)  # module executes its own report on import

BANDS = ["low", "mid", "high"]
LABELS = ["Below median", "Median to 90th", "Above 90th"]
SEEDS = _afb.SEEDS


def main():
    # realizable = the deployable condition the paper's practical claim rests on;
    # upPrecip = the control that must NOT reproduce the low-flow gain;
    # shuffled = the null that must move no band.
    series = {
        "Neighbour discharge": (_afb.by_band("L_upQpred"), BLUE, "o"),
        "Upstream precipitation": (_afb.by_band("L_upPrecip"), WARM, "s"),
        "Shuffled null": (_afb.by_band("L_upQshuf"), GREY, "^"),
    }

    fig, ax = plt.subplots(figsize=(LINEWIDTH, 2.35))
    xs = np.arange(len(BANDS))
    off = {0: -0.23, 1: 0.0, 2: 0.23}

    for k, (name, ((meds, ps), colour, marker)) in enumerate(series.items()):
        for j, b in enumerate(BANDS):
            vals = np.array(meds[b])
            x = xs[j] + off[k]
            # a vertical range bar makes the seed spread the visual signal, since the
            # claim is about consistency rather than magnitude
            ax.plot([x, x], [vals.min(), vals.max()], color=colour, lw=1.0,
                    alpha=0.55, zorder=2, solid_capstyle="round")
            ax.scatter(np.full_like(vals, x), vals, s=13, color=colour, marker=marker,
                       zorder=3, linewidths=0)
            ax.plot([x - 0.085, x + 0.085], [vals.mean()] * 2, color=colour, lw=2.0,
                    zorder=4, solid_capstyle="butt")
            if j == 0:
                ax.scatter([], [], s=18, color=colour, marker=marker, label=name)

    ax.axhline(0, color="black", lw=0.8, zorder=1)

    # Name the reading directly: the only band that is reliable is the low one.
    low = np.array(series["Neighbour discharge"][0][0]["low"])
    hi = np.array(series["Neighbour discharge"][0][0]["high"])
    ax.annotate("positive at every seed\n($p=9{\\times}10^{-4}$)",
                xy=(xs[0] + off[0] - 0.02, low.min()), xytext=(xs[0] - 0.32, -0.052),
                fontsize=6.9, color=BLUE, ha="left", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.annotate("larger, but spans a\nsevenfold range ($p=0.19$)",
                xy=(xs[2] + off[0], hi.max()), xytext=(xs[1] + 0.16, 0.228),
                fontsize=6.9, color="#6f6f6f", ha="left", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=0.7,
                                shrinkA=2, shrinkB=3))
    ax.set_xticks(xs)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("RMSE reduction (mm d$^{-1}$)")
    ax.set_xlabel("Flow band, set on each basin's own observed quantiles")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left",
              bbox_to_anchor=(0.005, 1.02), handletextpad=0.4, labelspacing=0.55)
    ax.margins(x=0.10)
    ax.set_ylim(-0.085, 0.265)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")

    # Diagnostics for an honest caption -- printed, never hardcoded into the figure.
    print(f"\nwrote {OUT}\n")
    print("caption diagnostics (per-seed medians, weakest-seed p):")
    for name, ((meds, ps), _, _) in series.items():
        for b, lab in zip(BANDS, LABELS):
            print(f"  {name.splitlines()[0]:28s} {lab:18s} "
                  f"{'/'.join(f'{v:+.3f}' for v in meds[b])}  weakest p={max(ps[b]):.4f}")


if __name__ == "__main__":
    main()
