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

# depth, n, median dNSE, p  -- LONGEST-PATH graph_depth (matches Eq. 1 and paper Table tab:depth).
# Recomputed 2026-08-10 from graph_depth (camels_topology.txt) + stored per-basin realizable deltas,
# pooled over seeds 11/13/17. Supersedes the earlier shortest-path stratification.
def _compute_strata():
    """Recompute the depth strata from stored per-basin metrics + the Eq.1 depth file.

    Previously these were transcribed constants, which is how the depth definition came to be
    migrated in the figure but not in the analysis scripts. Computing them here keeps the figure
    tied to the data.
    """
    import numpy as np, pandas as pd
    from pathlib import Path
    from scipy.stats import wilcoxon
    root = Path(__file__).parent.parent.parent
    dep = pd.read_csv(root / "topology_analysis/phase1_network_discovery/outputs/component0_depth_eq1.csv",
                      dtype={"basin": str}).set_index("basin")["depth_eq1"]
    runs = root / "runs/topology_ablation/component0"
    rows = {}
    for seed in (11, 13, 17):
        def m(cond):
            f = runs / f"{cond}_component0_seed{seed}/test/model_epoch030/test_metrics.csv"
            return (pd.read_csv(f, dtype={"basin": str}).set_index("basin")["NSE"]
                    if f.is_file() else None)
        L, R = m("L"), m("L_upQpred")
        if L is None or R is None:
            continue
        for b in L.index.intersection(R.index):
            rows.setdefault(int(dep.get(b, 0)), []).append(R[b] - L[b])
    out = []
    for d in sorted(rows):
        v = np.array(rows[d])
        pv = wilcoxon(v, alternative="greater")[1] if len(v) > 5 else float("nan")
        out.append((d, len(v), float(np.median(v)), float(pv)))
    return out


STRATA = _compute_strata()

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
