"""STODY Figure: where in the flow regime does the neighbour-discharge input help?

Every seed is drawn individually. The seed SPREAD is the finding in the high band --
averaging it would hide the instability the paper reports, so no bars are averaged away.

Source of truth: stored per-timestep predictions (test_results.p) under
runs/topology_ablation/component0/. Bands are set per basin on that basin's own observed
flow quantiles, so every basin contributes to every band. Metric is per-basin RMSE
reduction in mm/d, not a variance-normalised score: low-flow variance is tiny and
normalising inflates that band by an order of magnitude for arithmetic reasons.

Prints every number it plots, so the caption can be verified without reading the figure.

Usage:  python experiments/topology_ablation/plot_flow_bands.py
Writes: paper/canonical/figures/fig_flow_bands.pdf
"""
from pathlib import Path
import pickle, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
# Type 42 (TrueType) rather than matplotlib's default Type 3: NeurIPS style
# checks reject Type 3 fonts, and `make check` enforces a count of zero.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "topology_ablation" / "component0"
OUT = ROOT / "paper" / "canonical" / "figures" / "fig_flow_bands.pdf"
SEEDS = [11, 13, 17]
BANDS = [("low", "below median"), ("mid", "median to 90th"), ("high", "above 90th")]


def load(cond, seed):
    p = BASE / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_results.p"
    d = pickle.load(open(p, "rb"))
    return {str(b).zfill(8): (v["1D"]["xr"]["QObs(mm/d)_obs"].values.squeeze(),
                              v["1D"]["xr"]["QObs(mm/d)_sim"].values.squeeze())
            for b, v in d.items()}


edges = pd.read_csv(glob.glob(str(ROOT / "**" / "component0_edges.csv"), recursive=True)[0])
CONN = set(edges["child_id"].astype(str).str.zfill(8))


def rmse(o, m):
    return np.sqrt(np.mean((o - m) ** 2))


def by_band(cond, base="L"):
    """Per-seed median RMSE reduction (mm/d) in each band, plus weakest-seed p."""
    meds = {b: [] for b, _ in BANDS}
    ps = {b: [] for b, _ in BANDS}
    for s in SEEDS:
        L, R = load(base, s), load(cond, s)
        per = {b: [] for b, _ in BANDS}
        for bas in CONN & set(L) & set(R):
            o, ml = L[bas]
            _, mr = R[bas]
            ok = ~np.isnan(o)
            o, ml, mr = o[ok], ml[ok], mr[ok]
            if len(o) < 100:
                continue
            med, q90 = np.median(o), np.quantile(o, 0.9)
            masks = [("low", o < med), ("mid", (o >= med) & (o < q90)), ("high", o >= q90)]
            for lbl, k in masks:
                if k.sum() >= 10:
                    per[lbl].append(rmse(o[k], ml[k]) - rmse(o[k], mr[k]))
        for lbl, _ in BANDS:
            a = np.array(per[lbl])
            meds[lbl].append(np.median(a))
            ps[lbl].append(wilcoxon(a, alternative="greater")[1])
    return meds, {b: max(v) for b, v in ps.items()}


real_m, real_p = by_band("L_upQpred")
prec_m, prec_p = by_band("L_upPrecip")
null_m, null_p = by_band("L_upQshuf")

print("=== per-basin RMSE reduction (mm/d), connected basins, per seed ===")
for cond, (m, p) in [("neighbour discharge", (real_m, real_p)),
                     ("upstream precipitation", (prec_m, prec_p)),
                     ("time-shuffled null", (null_m, null_p))]:
    print(f"\n{cond}")
    for b, lbl in BANDS:
        vals = " / ".join(f"{v:+.4f}" for v in m[b])
        print(f"  {lbl:18s} {vals}   weakest-seed p={p[b]:.4f}")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(5.4, 2.9))
x = np.arange(len(BANDS))
w = 0.20
# markers differ as well as colours so the figure survives greyscale printing
marks = ["o", "s", "^"]
greys = ["#1a1a1a", "#666666", "#a8a8a8"]

for j, s in enumerate(SEEDS):
    ax.scatter(x - w + j * w, [real_m[b][j] for b, _ in BANDS],
               marker=marks[j], s=34, color=greys[j], zorder=3,
               label=f"seed {s}", edgecolor="white", linewidth=0.5)

# precipitation arm: open markers on a slight offset, shows it does not reproduce low flow
for j, s in enumerate(SEEDS):
    ax.scatter(x - w + j * w, [prec_m[b][j] for b, _ in BANDS],
               marker=marks[j], s=26, facecolor="none",
               edgecolor="#4676c4", linewidth=0.9, zorder=2)

ax.axhline(0, color="black", linewidth=0.7, zorder=1)
for i in (0.5, 1.5):
    ax.axvline(i, color="black", alpha=0.12, linewidth=0.6, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels([f"{lbl}\n$p={real_p[b]:.3g}$" for b, lbl in BANDS])
ax.set_ylabel("RMSE reduction (mm d$^{-1}$), higher is better", fontsize=8.5)
ax.set_xlabel("flow band, set on each basin's own observed quantiles", fontsize=9)
ax.tick_params(labelsize=8)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

h, l = ax.get_legend_handles_labels()
leg = ax.legend(h, l, fontsize=7.5, frameon=False, loc="upper left",
                handletextpad=0.3, borderaxespad=0.2)
leg.set_title("neighbour discharge", prop={"size": 7.5})
leg._legend_box.align = "left"
ax.text(0.985, 0.99, "open markers: upstream precipitation",
        transform=ax.transAxes, fontsize=7, color="#4676c4",
        ha="right", va="top")

ax.margins(y=0.10)
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax + 0.055)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"\n-> wrote {OUT}")
