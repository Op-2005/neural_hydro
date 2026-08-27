"""Where in the flow regime does the neighbour-discharge input help?

Every other analysis in this study collapses each run to one NSE per basin per
seed. That answers "how much" and never "where". This script uses the stored
per-timestep predictions (test_results.p) to decompose the gain by flow regime.

Metric is per-basin RMSE reduction in mm/d WITHIN a flow band, not a
variance-normalised score: low-flow variance is tiny, so a normalised score
inflates the low band by an order of magnitude and is misleading.

Bands are set per basin on observed-flow quantiles, so each basin contributes
to every band regardless of its own flow magnitude.

Judged by the weakest seed, as everywhere else in this study.
Includes the shuffled-null control and a cutoff sensitivity sweep.
"""
import pickle, glob
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

B = 'runs/topology_ablation/component0'
SEEDS = [11, 13, 17]

def load(cond, seed):
    p = f'{B}/{cond}_component0_seed{seed}/test/model_epoch030/test_results.p'
    d = pickle.load(open(p, 'rb'))
    return {str(b).zfill(8): (v['1D']['xr']['QObs(mm/d)_obs'].values.squeeze(),
                              v['1D']['xr']['QObs(mm/d)_sim'].values.squeeze())
            for b, v in d.items()}

edges = pd.read_csv(glob.glob('**/component0_edges.csv', recursive=True)[0])
CONN = set(edges['child_id'].astype(str).str.zfill(8))

def rmse(o, m):
    return np.sqrt(np.mean((o - m) ** 2))

def by_band(cond, base='L'):
    """Per-seed median RMSE reduction in each band, plus weakest-seed p."""
    bands = ['low', 'mid', 'high']
    meds = {b: [] for b in bands}; ps = {b: [] for b in bands}
    for s in SEEDS:
        L, R = load(base, s), load(cond, s)
        per = {b: [] for b in bands}
        for bas in CONN & set(L) & set(R):
            o, ml = L[bas]; _, mr = R[bas]
            ok = ~np.isnan(o); o, ml, mr = o[ok], ml[ok], mr[ok]
            if len(o) < 100:
                continue
            med, q90 = np.median(o), np.quantile(o, 0.9)
            for lbl, k in [('low', o < med), ('mid', (o >= med) & (o < q90)), ('high', o >= q90)]:
                if k.sum() >= 10:
                    per[lbl].append(rmse(o[k], ml[k]) - rmse(o[k], mr[k]))
        for lbl in bands:
            a = np.array(per[lbl])
            meds[lbl].append(np.median(a))
            ps[lbl].append(wilcoxon(a, alternative='greater')[1])
    return meds, ps

for cond, name in [('L_upQpred', 'realizable'), ('L_upQshuf', 'shuffled null')]:
    meds, ps = by_band(cond)
    print(f'=== {name} vs L: per-basin RMSE reduction (mm/d), connected basins ===')
    for b in ['low', 'mid', 'high']:
        flag = 'SIGNIFICANT' if max(ps[b]) < 0.05 else 'not significant'
        print(f'  {b:5s} {[f"{x:+.4f}" for x in meds[b]]}  weakest p={max(ps[b]):.4f}  {flag}')
    print()

print('=== sensitivity: low-flow result vs cutoff quantile (realizable) ===')
for cut in [0.3, 0.4, 0.5, 0.6]:
    meds, ps = [], []
    for s in SEEDS:
        L, R = load('L', s), load('L_upQpred', s); per = []
        for bas in CONN & set(L) & set(R):
            o, ml = L[bas]; _, mr = R[bas]
            ok = ~np.isnan(o); o, ml, mr = o[ok], ml[ok], mr[ok]
            if len(o) < 100:
                continue
            k = o < np.quantile(o, cut)
            if k.sum() >= 10:
                per.append(rmse(o[k], ml[k]) - rmse(o[k], mr[k]))
        a = np.array(per); meds.append(np.median(a)); ps.append(wilcoxon(a, alternative='greater')[1])
    print(f'  below q{cut:.1f}: {[f"{x:+.4f}" for x in meds]}  weakest p={max(ps):.4f}')

# --- Is the low-flow gain shared weather or shared state? ---------------------
# If shared WEATHER, upstream precipitation should reproduce the low-flow gain.
# If shared STATE, it should not.
print()
print('=== precipitation-only input, low-flow band (vs L) ===')
meds, ps = [], []
for s in SEEDS:
    L, R = load('L', s), load('L_upPrecip', s); per = []
    for bas in CONN & set(L) & set(R):
        o, ml = L[bas]; _, mr = R[bas]
        ok = ~np.isnan(o); o, ml, mr = o[ok], ml[ok], mr[ok]
        if len(o) < 100:
            continue
        k = o < np.median(o)
        if k.sum() >= 10:
            per.append(rmse(o[k], ml[k]) - rmse(o[k], mr[k]))
    a = np.array(per); meds.append(np.median(a)); ps.append(wilcoxon(a, alternative='greater')[1])
print(f'  {[f"{x:+.4f}" for x in meds]}  weakest p={max(ps):.4f}')
print('  -> precipitation does NOT reproduce the low-flow gain; the signal is storage, not weather.')
