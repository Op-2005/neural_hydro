"""Independent recomputation of every number used in the STODY submission.

Reads ONLY raw per-basin test_metrics.csv under runs/topology_ablation/component0/.
Does not read any analysis/*.md or any paper .tex -- the point is to catch a stale
transcription, so nothing derived is trusted.

Significance is judged by the WEAKEST of the three seeds, never pooled.

Usage: python experiments/topology_ablation/verify_stody_numbers.py
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "topology_ablation" / "component0"
SEEDS = [11, 13, 17]


def nse(cond, seed):
    p = BASE / f"{cond}_component0_seed{seed}" / "test" / "model_epoch030" / "test_metrics.csv"
    d = pd.read_csv(p, dtype={"basin": str})
    d["basin"] = d["basin"].str.zfill(8)
    return d.set_index("basin")["NSE"]


def connected():
    f = glob.glob(str(ROOT / "**" / "component0_edges.csv"), recursive=True)[0]
    e = pd.read_csv(f)
    return set(e["child_id"].astype(str).str.zfill(8))


CONN = connected()


def paired(cond, base="L", subset=None, alt="greater"):
    """Per-seed paired median delta and the weakest-seed one-sided p."""
    meds, ps = [], []
    for s in SEEDS:
        a, b = nse(base, s), nse(cond, s)
        idx = a.index.intersection(b.index)
        if subset is not None:
            idx = idx.intersection(subset)
        d = (b[idx] - a[idx]).dropna()
        meds.append(np.median(d))
        ps.append(wilcoxon(d, alternative=alt)[1])
    return meds, max(ps)


def paired_cond(c1, c2, subset=None, alt="greater"):
    """c1 minus c2, paired per basin."""
    meds, ps = [], []
    for s in SEEDS:
        a, b = nse(c2, s), nse(c1, s)
        idx = a.index.intersection(b.index)
        if subset is not None:
            idx = idx.intersection(subset)
        d = (b[idx] - a[idx]).dropna()
        meds.append(np.median(d))
        ps.append(wilcoxon(d, alternative=alt)[1])
    return meds, max(ps)


def row(label, meds, p, n=None):
    xs = " / ".join(f"{m:+.4f}" for m in meds)
    cs = np.mean(meds)
    extra = f"  n={n}" if n else ""
    print(f"  {label:38s} {xs}   cross-seed {cs:+.4f}   weakest p={p:.2e}{extra}")


print("=" * 78)
print("STODY number verification -- recomputed from raw test_metrics.csv")
print("=" * 78)

print(f"\nBasins: 183 total, {len(CONN)} connected (have >=1 upstream parent)\n")

print("[C1] Static network position (all 183 basins)")
row("L_T vs L", *paired("L_T"))
row("L_noID_T vs L_noID (no one-hot)", *paired("L_noID_T", base="L_noID"))

print("\n[C2] Neighbour discharge as a dynamic signal (all 183 basins)")
row("L_upQ (oracle) vs L", *paired("L_upQ"))
row("L_upQpred (two-stage) vs L", *paired("L_upQpred"))

print("\n[C3] Capacity control (all 183 basins)")
row("L_upQshuf (null) vs L  [two-sided]", *paired("L_upQshuf", alt="two-sided"))
row("realizable vs null", *paired_cond("L_upQpred", "L_upQshuf"))

print("\n[C4] Proximity vs connectivity (connected basins, oracle)")
for c, lbl in [("L_upQ", "forward (true upstream)"),
               ("L_upQdistctrl", "distance-matched non-parents"),
               ("L_upQrev", "reversed (downstream)"),
               ("L_upQrandclean", "random rewire (clean)")]:
    try:
        row(lbl, *paired(c, subset=CONN), n=len(CONN))
    except Exception as e:
        print(f"  {lbl:38s} MISSING ({type(e).__name__})")

print("\n[C6] Selector comparison (connected basins, oracle on both sides)")
row("L_upQknn2 vs L", *paired("L_upQknn2", subset=CONN))
row("L_upQ (network) vs L", *paired("L_upQ", subset=CONN))
row("PAIRED knn2 - network", *paired_cond("L_upQknn2", "L_upQ", subset=CONN))
row("L_upQknn4 vs L", *paired("L_upQknn4", subset=CONN))

print("\n[C6] Same, all 183 basins")
row("L_upQknn2 vs L", *paired("L_upQknn2"))
row("PAIRED knn2 - network", *paired_cond("L_upQknn2", "L_upQ"))

print("\n[C7] Two-stage: does the advantage survive prediction? (connected)")
row("L_upQpredknn2 (kNN, two-stage) vs L", *paired("L_upQpredknn2", subset=CONN))
row("L_upQpred (network, two-stage) vs L", *paired("L_upQpred", subset=CONN))
row("PAIRED kNN - network [two-sided]",
    *paired_cond("L_upQpredknn2", "L_upQpred", subset=CONN, alt="two-sided"))

print("\n[C6b] Minimum-separation floors (connected, oracle) -- nesting check")
for c, lbl in [("L_upQknn2min10km", "kNN2 with 10 km floor"),
               ("L_upQknn2min15km", "kNN2 with 15 km floor")]:
    row(lbl + " vs L", *paired(c, subset=CONN))
    row(lbl + " - network", *paired_cond(c, "L_upQ", subset=CONN))

print("\n[C11] Over-connectivity robustness (connected basins)")
row("L_upQ_k2 (in-degree<=2) vs L", *paired("L_upQ_k2", subset=CONN))

print("\n[C9] Precipitation channel (connected basins)")
row("L_upPrecip vs L", *paired("L_upPrecip", subset=CONN))
row("realizable - precip", *paired_cond("L_upQpred", "L_upPrecip", subset=CONN))

print("\n" + "=" * 78)
print("Baseline absolute skill (median NSE per seed, all 183)")
for c in ["L", "L_upQ", "L_upQpred", "L_upQshuf"]:
    vals = [np.median(nse(c, s).dropna()) for s in SEEDS]
    print(f"  {c:14s} {' / '.join(f'{v:.4f}' for v in vals)}   mean {np.mean(vals):.4f} "
          f"+/- {np.std(vals):.4f}")
print("=" * 78)
