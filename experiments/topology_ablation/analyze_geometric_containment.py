"""Geometric containment test for the nearest-gauge advantage.

The area-ratio proxy used in analyze_containment.py flags pairs whose areas are
similar, which is neither necessary nor sufficient for nesting. It is blind to the
case that leaks most: a small gauge sitting inside a much larger target.

This test is a necessary condition instead. Two catchments can nest only if the
smaller gauge lies within the larger catchment, which bounds their gauge separation
by the larger catchment's equivalent-circle radius r = sqrt(A/pi). Where
d(i,j) >= r_large, nesting is geometrically impossible and no containment leak can
exist regardless of catchment shape.

Uses only gauge coordinates and areas from camels_topo.txt -- no shapefiles.
"""

import pandas as pd, numpy as np, glob, math, sys
from scipy.stats import wilcoxon

t=pd.read_csv('datasets/camels_us/camels_attributes_v2.0/camels_topo.txt',sep=';')
t['gauge_id']=t['gauge_id'].astype(str).str.zfill(8)
A=dict(zip(t['gauge_id'],t['area_gages2']))
EL=dict(zip(t['gauge_id'],t['elev_mean']))
LL=dict(zip(t['gauge_id'],zip(t['gauge_lat'],t['gauge_lon'])))
def gc(a,b):
    la1,lo1=map(math.radians,LL[a]); la2,lo2=map(math.radians,LL[b])
    return 6371*2*math.asin(math.sqrt(math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2))

e=pd.read_csv(glob.glob('**/component0_edges.csv',recursive=True)[0])
e['parent_id']=e['parent_id'].astype(str).str.zfill(8); e['child_id']=e['child_id'].astype(str).str.zfill(8)
conn=sorted(set(e['child_id']))
parents={}
for _,r in e.iterrows(): parents.setdefault(r['child_id'],set()).add(r['parent_id'])

basins=sorted(set(t['gauge_id']) & set(pd.read_csv('runs/topology_ablation/component0/L_component0_seed11/test/model_epoch030/test_metrics.csv')['basin'].astype(str).str.zfill(8)))

# kNN2 selection, replicating build_distance_control: k nearest excluding self and true parents
def knn(i,k=2):
    cand=[b for b in basins if b!=i and b not in parents.get(i,set())]
    return sorted(cand,key=lambda b: gc(i,b))[:k]

# Geometric containment: nesting requires the smaller gauge to lie inside the larger catchment.
# Equivalent-circle radius r=sqrt(A/pi). Flag if d < r_large -- the ONLY regime nesting is possible in.
# This catches small-inside-large, which the area-ratio proxy passes.
def geo_flag(i,j,e=1.0):
    # A streamgauge sits at its catchment's OUTLET -- a point on the boundary, not
    # the interior. The distance from a boundary point to the far side of a disc of
    # radius r is the DIAMETER 2r, not r. So the reach is 2*sqrt(e*A/pi), where e is
    # the elongation of an equal-area ellipse (e=1 = circle, the smallest defensible
    # reach, since by the isoperimetric inequality the circle has the least extent).
    d=gc(i,j); reach=2*math.sqrt(e*max(A[i],A[j])/math.pi)
    return d < reach

rows=[]
for i in conn:
    ks=knn(i)
    rows.append((i, any(geo_flag(i,j) for j in ks),
                    any(max(A[i],A[j])/min(A[i],A[j])<2 for j in ks)))
df=pd.DataFrame(rows,columns=['basin','geo_nested','area_nested'])
print('=== kNN2 pairs on connected basins ===')
print('geometrically possible nesting:', df['geo_nested'].sum(), '/', len(df))
print('area-ratio<2 flagged        :', df['area_nested'].sum(), '/', len(df))
print('geo-clean but area-flagged  :', ((~df['geo_nested'])&df['area_nested']).sum())
print('geo-flagged but area-clean  :', (df['geo_nested']&(~df['area_nested'])).sum(), ' <-- the blind spot')

B='runs/topology_ablation/component0'
def nse(c,s):
    d=pd.read_csv(f'{B}/{c}_component0_seed{s}/test/model_epoch030/test_metrics.csv')
    d['basin']=d['basin'].astype(str).str.zfill(8); return dict(zip(d['basin'],d['NSE']))

clean=sorted(df[~df['geo_nested']]['basin']); nest=sorted(df[df['geo_nested']]['basin'])
for lbl,S in [('GEO-CLEAN (no nesting possible)',clean),('geo-flagged',nest)]:
    meds=[];ps=[]
    for s in [11,13,17]:
        K,G=nse('L_upQknn2',s),nse('L_upQ',s)
        bs=[b for b in S if b in K and b in G]
        d=np.array([K[b]-G[b] for b in bs])
        meds.append(np.median(d)); ps.append(wilcoxon(d,alternative='greater')[1])
    print(f'{lbl:34s} n={len(bs):3d} per-seed {[f"{m:+.4f}" for m in meds]} weakest p={max(ps):.3e}')

# Symmetry check: how exposed is the NETWORK arm under the same geometric instrument?
ne=0; tot=0
for _,r in e.iterrows():
    tot+=1
    if geo_flag(r['parent_id'],r['child_id']): ne+=1
print()
print(f'true edges geometrically nestable: {ne}/{tot} ({100*ne/tot:.1f}%)')
kn=sum(1 for i in conn for j in knn(i) if geo_flag(i,j)); kt=sum(1 for i in conn for j in knn(i))
print(f'kNN2 pairs geometrically nestable: {kn}/{kt} ({100*kn/kt:.1f}%)')

# --- Elongation sensitivity -------------------------------------------------
# Real catchments are dendritic and elongated, so e=1 understates reach.
print()
print('=== elongation sensitivity (e=1 is the weakest bound) ===')
for e in [1.0,2.0,3.0,4.0]:
    cl=[i for i in conn if not any(geo_flag(i,j,e) for j in knn(i))]
    meds=[];ps=[]
    for s in [11,13,17]:
        K,G=nse('L_upQknn2',s),nse('L_upQ',s)
        bs=[b for b in cl if b in K and b in G]
        d=np.array([K[b]-G[b] for b in bs])
        meds.append(np.median(d)); ps.append(wilcoxon(d,alternative='greater')[1])
    print(f'  e={e:.1f}  n={len(bs):3d}  {[f"{m:+.4f}" for m in meds]}  weakest p={max(ps):.2e}')

# --- Direction control ------------------------------------------------------
# A DOWNSTREAM neighbour integrates the target's own outflow, so its observed
# discharge partly reflects the target even when nesting is impossible. This is
# a third channel, distinct from nesting and from temporal leakage.
# 'Downstream' = Eq.(edge) applied in reverse: neighbour lower AND >=1.5x area.
print()
print('=== direction control: does a downstream neighbour drive the advantage? ===')
dfrac={}
for i in conn:
    ks=knn(i)
    dfrac[i]=sum(1 for j in ks if EL[j]<EL[i] and A[j]>=1.5*A[i])/len(ks)
for lbl,sel in [('no downstream neighbour',[i for i in conn if dfrac[i]==0]),
                ('>=1 downstream neighbour',[i for i in conn if dfrac[i]>0])]:
    meds=[];ps=[]
    for s in [11,13,17]:
        K,G=nse("L_upQknn2",s),nse("L_upQ",s)
        bs=[b for b in sel if b in K and b in G]
        d=np.array([K[b]-G[b] for b in bs])
        meds.append(np.median(d)); ps.append(wilcoxon(d,alternative='greater')[1])
    print(f'  {lbl:26s} n={len(bs):3d} {[f"{m:+.4f}" for m in meds]} weakest p={max(ps):.2e}')

# --- Overlap with the min-separation floor ----------------------------------
# The geometric flag is itself a distance criterion, so it is NOT independent of
# the 15 km separation floor. Quantify rather than assert.
flag=[(i,j) for i in conn for j in knn(i) if geo_flag(i,j)]
surv=[(i,j) for i,j in flag if gc(i,j)>=15]
print()
print(f'geometrically flagged pairs: {len(flag)}; surviving a 15 km floor: {len(surv)}')
print(f'  -> the floor removes {len(flag)-len(surv)} of them; the controls are not independent')
