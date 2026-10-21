import csv, json, sys, urllib.request, concurrent.futures as cf
EDGES = sys.argv[1]; BASINS = sys.argv[2]
edges = list(csv.DictReader(open(EDGES)))
basins = {l.strip().zfill(8) for l in open(BASINS) if l.strip()}
children = sorted({e['child_id'].zfill(8) for e in edges})
def up(g):
    url = f"https://api.water.usgs.gov/nldi/linked-data/nwissite/USGS-{g}/navigation/UT/nwissite?distance=1000"
    for _ in range(3):
        try:
            d = json.load(urllib.request.urlopen(url, timeout=60))
            return g, {f['properties']['identifier'].replace('USGS-','') for f in d.get('features',[])}
        except Exception as e:
            err = str(e)
    return g, None
res = {}
with cf.ThreadPoolExecutor(8) as ex:
    for g, s in ex.map(up, children): res[g] = s
json.dump({k:(sorted(v) if v is not None else None) for k,v in res.items()}, open('nldi_upstream.json','w'))
ok = [e for e in edges if res.get(e['child_id'].zfill(8)) is not None]
true = [e for e in ok if e['parent_id'].zfill(8) in res[e['child_id'].zfill(8)]]
print(f"children queried: {len(children)}, failed: {sum(v is None for v in res.values())}")
print(f"edges checked: {len(ok)} / {len(edges)}")
print(f"edges confirmed by NHDPlus (parent upstream of child): {len(true)}  precision={len(true)/max(1,len(ok)):.3f}")
# recall: true NHDPlus upstream pairs among study basins that the heuristic missed
inferred = {(e['parent_id'].zfill(8), e['child_id'].zfill(8)) for e in edges}
truepairs = {(p,c) for c,s in res.items() if s for p in s if p in basins and p!=c}
print(f"NHDPlus-true pairs among study basins (for queried children): {len(truepairs)}; recovered by heuristic: {len(truepairs & inferred)}")
kids_with_true = {c for (p,c) in truepairs}
print(f"connected basins with >=1 NHDPlus-true upstream study gauge: {len(kids_with_true)} / {len(children)}")
