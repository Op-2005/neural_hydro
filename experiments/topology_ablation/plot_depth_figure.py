"""Regenerate paper Figure 3 (realizable dNSE by graph depth).

Source of truth: analysis/PAPER_TABLE.md Table 3 (per-stratum Wilcoxon, seeds 11/13/17).

Two fixes over the previous version of this figure:
  1. Grey encodes "does not reach significance" CONSISTENTLY. Previously depth 0 was grey
     but depth 4 was blue despite p=0.34, which told the reader depth 4 was a real effect.
  2. Each bar is annotated with its stratum n, so depth 4 (n=6) is not read with the same
     weight as depth 1 (n=243).

Usage:  python experiments/topology_ablation/plot_depth_figure.py
Writes: paper/figures/fig_depth.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures" / "fig_depth.pdf"

# depth, n, median dNSE, p  -- transcribed from analysis/PAPER_TABLE.md Table 3
STRATA = [
    (0, 99, 0.002, 0.24),
    (1, 243, 0.020, 2.6e-9),
    (2, 153, 0.031, 4.7e-12),
    (3, 48, 0.044, 8.4e-4),
    (4, 6, 0.015, 0.34),
]
ALPHA = 0.05

SIG_COLOR = "#4676c4"
NS_COLOR = "#bfbfbf"


def main() -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.0))

    for depth, n, delta, p in STRATA:
        significant = p < ALPHA
        ax.bar(
            depth,
            delta,
            width=0.68,
            color=SIG_COLOR if significant else NS_COLOR,
            zorder=3,
        )
        ax.text(
            depth,
            delta + 0.0015,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#333333",
            zorder=4,
        )

    ax.set_xlabel("graph depth")
    ax.set_ylabel(r"median realizable $\Delta$NSE")
    ax.set_xticks([d for d, *_ in STRATA])
    ax.axhline(0.0, color="#333333", linewidth=0.8, zorder=2)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SIG_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=NS_COLOR),
    ]
    ax.legend(
        handles,
        [f"$p < {ALPHA}$", "not significant"],
        fontsize=7.5,
        frameon=False,
        loc="upper left",
    )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
