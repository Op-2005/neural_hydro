"""STODY Figure: the proximity dose-response, with the two selectors placed on it.

Two things are drawn together and must not be confused:
  * the SWEEP arms (grey) substitute every true edge with a non-upstream basin at a
    controlled separation. They exist at SEED 11 ONLY and are read for SHAPE.
  * the SELECTOR points (black) are the true river network and the nearest-gauge
    input. These are cross-seed means over seeds 11/13/17, drawn with per-seed spread.

Separations are recomputed here from primary data -- the edge file's own distance_km
for the network, and great-circle distance between gauge coordinates for the nearest
gauges -- not transcribed from any analysis file.

Prints every number it plots so the caption can be verified without reading the figure.

Usage:  python experiments/topology_ablation/plot_distance_decay.py
Writes: paper/canonical/figures/fig_distance_decay.pdf
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
# Type 42 (TrueType) rather than matplotlib's default Type 3: NeurIPS style
# checks reject Type 3 fonts, and `make check` enforces a count of zero.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "topology_ablation" / "component0"
OUT = ROOT / "paper" / "canonical" / "figures" / "fig_distance_decay.pdf"
SEEDS = [11, 13, 17]


def nse(cond, seed):
    p = BASE / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_metrics.csv"
    d = pd.read_csv(p, dtype={"basin": str})
    d["basin"] = d["basin"].str.zfill(8)
    return d.set_index("basin")["NSE"]


edges = pd.read_csv(glob.glob(str(ROOT / "**" / "component0_edges.csv"), recursive=True)[0])
CONN = set(edges["child_id"].astype(str).str.zfill(8))
MEAN_EDGE_KM = edges["distance_km"].mean()


def delta(cond, seed):
    a, b = nse("L", seed), nse(cond, seed)
    idx = a.index.intersection(b.index).intersection(CONN)
    return np.median((b[idx] - a[idx]).dropna())


# ---- nearest-gauge separation, recomputed from gauge coordinates -----------
topo = pd.read_csv(glob.glob(str(ROOT / "**" / "camels_topo.txt"), recursive=True)[0], sep=";")
topo["gauge_id"] = topo["gauge_id"].astype(str).str.zfill(8)
topo = topo.set_index("gauge_id")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# Match the feature construction EXACTLY (notebooks/colab_knn_baseline.ipynb, Cell 7):
# candidates are the Component-0 basin list, and each basin's TRUE PARENTS and itself are
# excluded so the arm carries zero topology. Omitting that exclusion understates the
# separation (38.96 km vs the correct 46.7 km), so it is not optional.
blist = ROOT / "topology_analysis" / "phase1_network_discovery" / "outputs" / "component0_basins.txt"
basins = [l.strip().zfill(8) for l in open(blist) if l.strip()]
parents = {}
for pp, cc in zip(edges["parent_id"].astype(str).str.zfill(8),
                  edges["child_id"].astype(str).str.zfill(8)):
    parents.setdefault(cc, set()).add(pp)

lat = {b: float(topo.loc[b, "gauge_lat"]) for b in basins if b in topo.index}
lon = {b: float(topo.loc[b, "gauge_lon"]) for b in basins if b in topo.index}

K = 2
seps = []
for c in basins:
    if c not in lat:
        continue
    forb = parents.get(c, set()) | {c}
    others = [o for o in basins if o not in forb and o in lat]
    others.sort(key=lambda o: haversine(lat[c], lon[c], lat[o], lon[o]))
    seps += [haversine(lat[c], lon[c], lat[o], lon[o]) for o in others[:K]]
knn2_sep = float(np.mean(seps))

# ---- the sweep (seed 11 only) ---------------------------------------------
sweep = [("L_upQdistctrl", 101), ("L_upQdist175km", 175), ("L_upQdist250km", 250),
         ("L_upQdist350km", 350), ("L_upQdist500km", 500)]
sx, sy = [], []
print("=== swept substitute edges (SEED 11 ONLY -- read for shape) ===")
for cond, km in sweep:
    d = delta(cond, 11)
    sx.append(km)
    sy.append(d)
    print(f"  {km:3d} km   dNSE = {d:+.4f}   ({cond})")

# ---- the selectors (cross-seed) -------------------------------------------
print("\n=== selectors (cross-seed over 11/13/17, connected basins) ===")
sel = {}
for cond, lbl, km in [("L_upQ", "river network", MEAN_EDGE_KM),
                      ("L_upQknn2", "nearest 2 gauges", knn2_sep)]:
    per = [delta(cond, s) for s in SEEDS]
    sel[lbl] = (km, float(np.mean(per)), per)
    print(f"  {lbl:18s} sep={km:5.1f} km   per-seed "
          f"{' / '.join(f'{v:+.4f}' for v in per)}   mean {np.mean(per):+.4f}")

print(f"\nmean true-edge separation  = {MEAN_EDGE_KM:.2f} km  (n={len(edges)} edges)")
print(f"mean nearest-2-gauge sep   = {knn2_sep:.2f} km  (n={len(seps)} pairs, true parents excluded)")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(5.4, 2.9))

ax.plot(sx, sy, color="#999999", linewidth=1.0, zorder=2)
ax.scatter(sx, sy, s=20, color="#999999", zorder=3)
ax.annotate("swept substitute edges\n(seed 11, shape only)",
            xy=(330, np.interp(330, sx, sy)), xytext=(395, 0.047),
            fontsize=7, color="#6f6f6f", ha="center",
            arrowprops=dict(arrowstyle="-", color="#bbbbbb", linewidth=0.6))

for lbl, mark in [("river network", "o"), ("nearest 2 gauges", "D")]:
    km, mean, per = sel[lbl]
    ax.scatter([km], [mean], s=52, marker=mark, color="#1a1a1a", zorder=5,
               edgecolor="white", linewidth=0.6)
    ax.vlines(km, min(per), max(per), color="#1a1a1a", linewidth=1.1, zorder=4)

ax.annotate(f"nearest 2 gauges, {knn2_sep:.0f} km",
            xy=(knn2_sep, sel["nearest 2 gauges"][1]), xytext=(knn2_sep + 62, 0.083),
            fontsize=7.5, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="#888888", linewidth=0.6))
ax.annotate(f"river network, {MEAN_EDGE_KM:.0f} km",
            xy=(MEAN_EDGE_KM, sel["river network"][1]), xytext=(MEAN_EDGE_KM + 68, 0.036),
            fontsize=7.5, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="#888888", linewidth=0.6))

ax.axhline(0, color="black", linewidth=0.7)
ax.set_xlabel("mean separation to the neighbours supplying the input (km)", fontsize=8.5)
ax.set_ylabel("$\\Delta$NSE vs baseline, higher is better", fontsize=8.5)
ax.set_xlim(0, 560)
ax.tick_params(labelsize=8)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.margins(y=0.12)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"\n-> wrote {OUT}")
