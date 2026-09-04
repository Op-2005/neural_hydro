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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Type 42 (TrueType) rather than matplotlib's default Type 3: NeurIPS-style venues
# reject Type 3 fonts, and `make check` audits for them.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

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
        "Neighbour discharge\n(deployable)": (_afb.by_band("L_upQpred"), "#1f4e79", "o"),
        "Upstream precipitation": (_afb.by_band("L_upPrecip"), "#c0703a", "s"),
        "Shuffled null": (_afb.by_band("L_upQshuf"), "#8a8a8a", "^"),
    }

    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    xs = np.arange(len(BANDS))
    off = {0: -0.22, 1: 0.0, 2: 0.22}

    for k, (name, ((meds, ps), colour, marker)) in enumerate(series.items()):
        for j, b in enumerate(BANDS):
            vals = np.array(meds[b])
            x = xs[j] + off[k]
            # per-seed points: the claim is about consistency, so show every seed
            ax.scatter(np.full_like(vals, x), vals, s=17, color=colour, marker=marker,
                       zorder=3, linewidths=0, alpha=0.85)
            # cross-seed mean as a short bar
            ax.plot([x - 0.075, x + 0.075], [vals.mean()] * 2, color=colour, lw=1.9, zorder=4)
            if j == 0:
                ax.scatter([], [], s=22, color=colour, marker=marker, label=name)
            # mark the one band that is significant at the weakest seed
            if max(ps[b]) < 0.05:
                ax.text(x, vals.max() + 0.012, "*", ha="center", va="bottom",
                        fontsize=11, color=colour)

    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("RMSE reduction (mm d$^{-1}$)")
    ax.set_xlabel("Flow band, set on each basin's own observed quantiles")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left", handletextpad=0.4)
    ax.margins(x=0.10)
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
