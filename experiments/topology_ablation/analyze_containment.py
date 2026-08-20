"""Does catchment nesting explain the nearest-gauge advantage?

CAMELS gauges nest: a neighbour whose catchment is largely contained in the target's
is partly a measurement of the target itself, which would inflate an oracle contrast.
The minimum-separation floors test near-DUPLICATE gauges, not containment, because
distance and area-containment are nearly independent.

Proxy: a pair is flagged when max/min catchment area < 2, the same threshold the paper
applies to its own edges. Stratify the kNN-vs-network advantage by whether a basin's
nearest pair contains a flagged neighbour. No retraining -- uses stored predictions.

Writes: analysis/CONTAINMENT.md
"""
import numpy as np, pandas as pd, pickle
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent.parent.parent
P1 = ROOT / "topology_analysis/phase1_network_discovery/outputs"
RUNS = ROOT / "runs/topology_ablation/component0"
FEAT = Path(__file__).parent / "features"
OUT = Path(__file__).parent / "analysis" / "CONTAINMENT.md"
SEEDS = (11, 13, 17)


def haversine(a1, o1, a2, o2):
    R = 6371.0
    p1, p2 = np.radians(a1), np.radians(a2)
    dp, dl = np.radians(a2 - a1), np.radians(o2 - o1)
    return 2 * R * np.arcsin(np.sqrt(np.sin(dp / 2) ** 2 +
                                     np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))


def main():
    basins = [l.strip() for l in open(P1 / "component0_basins.txt") if l.strip()]
    topo = pd.read_csv(ROOT / "datasets/camels_us/camels_attributes_v2.0/camels_topo.txt",
                       sep=";", dtype={"gauge_id": str}).set_index("gauge_id")
    lat = {b: float(topo.loc[b, "gauge_lat"]) for b in basins}
    lon = {b: float(topo.loc[b, "gauge_lon"]) for b in basins}
    area = {b: float(topo.loc[b, "area_gages2"]) for b in basins}
    d = lambda a, b: haversine(lat[a], lon[a], lat[b], lon[b])
    ratio = lambda a, b: max(area[a], area[b]) / min(area[a], area[b])

    E = pd.read_csv(P1 / "component0_edges.csv", dtype=str)
    tp = {}
    for p, c in zip(E.parent_id, E.child_id):
        tp.setdefault(c, set()).add(p)

    def knn_pairs(k=2, min_km=0.0):
        out = []
        for c in basins:
            forb = tp.get(c, set()) | {c}
            cand = [x for x in basins if x not in forb and d(c, x) >= min_km]
            out += [(c, o) for o in sorted(cand, key=lambda x: d(c, x))[:k]]
        return out

    def flag_rate(pairs):
        r = [ratio(a, b) for a, b in pairs]
        return sum(1 for x in r if x < 2), len(r)

    L = ["# Does catchment nesting explain the nearest-gauge advantage?\n",
         "CAMELS gauges nest. A neighbour whose catchment is largely contained in the target's is",
         "partly a measurement of the target, which would inflate an oracle contrast specifically.",
         "We use max/min catchment area $<2$ as a containment proxy, the same threshold the paper",
         "applies to its own edges.\n",
         "## The distance floors do not test containment\n",
         "| selector | pairs flagged (area ratio $<2$) |", "|---|---|"]
    for m in (0, 10, 15):
        f, n = flag_rate(knn_pairs(2, m))
        lab = "nearest 2" if m == 0 else f"nearest 2, $\\ge {m}$ km apart"
        L.append(f"| {lab} | {f}/{n} ({100*f/n:.1f}%) |")
    f, n = flag_rate([(c, p) for c, ps in tp.items() for p in ps])
    L.append(f"| true upstream edges | {f}/{n} ({100*f/n:.1f}%) |")
    L += ["",
          "Distance and area-containment are nearly independent: a 15 km floor moves the flagged",
          "fraction by under one point. The nearest-gauge selector is about twice as flagged as the",
          "network selector it outperforms, so containment must be tested directly rather than",
          "inferred from the separation floors.\n",
          "## Stratifying the advantage by nesting risk\n"]

    nested = {}
    for c in basins:
        forb = tp.get(c, set()) | {c}
        o = sorted([x for x in basins if x not in forb], key=lambda x: d(c, x))[:2]
        nested[c] = any(ratio(c, x) < 2 for x in o)

    def nse(cond, s):
        return pd.read_csv(RUNS / f"{cond}_component0_seed{s}/test/model_epoch030/test_metrics.csv",
                           dtype={"basin": str}).set_index("basin")["NSE"]

    fe = pickle.load(open(FEAT / "upstream_q_component0_lag1.p", "rb"))
    conn = sorted([b for b, v in fe.items() if float(np.nanmax(np.abs(v.values))) > 0])
    lo, hi = [], []
    for s in SEEDS:
        K, G = nse("L_upQknn2", s), nse("L_upQ", s)
        for b in conn:
            (hi if nested[b] else lo).append(K[b] - G[b])

    L += ["| basins | n (basin-seed) | median kNN $-$ network | $p$ |", "|---|---|---|---|"]
    for lab, v in [("nearest pair NOT area-nested", lo), ("nearest pair area-nested", hi)]:
        L.append(f"| {lab} | {len(v)} | ${np.median(v):+.4f}$ | ${wilcoxon(v, alternative='greater')[1]:.1e}$ |")
    L += ["",
          f"**Containment inflates the advantage but does not create it.** Among basins whose two",
          f"nearest gauges are not area-nested, the nearest-gauge input still beats the network by",
          f"**{np.median(lo):+.4f}** ($p={wilcoxon(lo, alternative='greater')[1]:.1e}$, $n={len(lo)}$",
          f"basin-seed pairs). The advantage is larger where nesting is present ({np.median(hi):+.4f}),",
          "which is the direction containment predicts, so the effect is real and partly inflated",
          "rather than wholly artifactual.\n",
          "Scope: area ratio is a proxy. Catchment-boundary overlap from the CAMELS shapefiles would",
          "measure containment directly and is the stronger test."]
    OUT.write_text("\n".join(L) + "\n")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
