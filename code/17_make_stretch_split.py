"""Stretch-separated partition of HighRPD.
Consecutive source groups of one capture series image overlapping pavement (results/frame_overlap_pairs.csv).
Groups are therefore assigned to subsets in contiguous blocks of the capture sequence, and a buffer removes every
training group that overlaps a validation or test group (registered pair, or index gap of at most three frames within
the same run); validation groups that overlap test groups are removed in the same way.
Among 200 candidate assignments the one whose class proportions are closest to 70/15/15 after buffering is kept.
Outputs: data/splits/stretch_{train,val,test}.txt, highrpd_stretch.yaml, highrpd_stretch_pitx3.yaml,
         column split_stretch in results/tiles_index.csv and results/tiles_gt_inventory.csv,
         results/stretch_blocks.csv, rows for the stretch partition in results/table_splits.csv"""
import re, os, numpy as np, pandas as pd, yaml

ROOT = os.path.abspath('data/HighRPD'); OUT = 'data/splits'
BLOCK = 16          # groups per block
BUF_GAP = 3         # index gap within a run that is always treated as overlapping
T = pd.read_csv('results/tiles_index.csv')
O = pd.read_csv('results/frame_overlap_pairs.csv')
O = O[(O.inliers >= 30) & ((O.scale - 1).abs() < 0.05) & (O.rot_deg.abs() < 5)]


def parse(s):
    m = re.match(r'(DJI|dji)_(\d{14})_(\d{4})', s)
    if m: return m.group(1), int(m.group(3))
    m = re.match(r'1x(\d+)', s); return '1x', int(m.group(1))


G = T.groupby('src').agg(tiles=('stem', 'size'), n0=('n0', 'sum'), n1=('n1', 'sum'), n2=('n2', 'sum')).reset_index()
G[['series', 'idx']] = G.src.apply(lambda s: pd.Series(parse(s)))
G = G.sort_values(['series', 'idx']).reset_index(drop=True)

# runs: a new run starts where the index gap exceeds BUF_GAP; blocks: chunks of BLOCK groups within a run
G['run'] = ''; G['block'] = ''
for ser, d in G.groupby('series'):
    if ser == 'dji':                                  # 16 scattered frames from a separate flight, no registered overlap
        G.loc[d.index, 'run'] = ser + '_' + d.idx.astype(str); G.loc[d.index, 'block'] = 'dji'; continue
    r = (d.idx.diff().fillna(1) > BUF_GAP).cumsum()
    G.loc[d.index, 'run'] = ser + '_r' + r.astype(str)
    for rr, dd in d.groupby(r):
        k = np.arange(len(dd)) // BLOCK
        G.loc[dd.index, 'block'] = ser + '_r' + str(rr) + '_b' + pd.Series(k, index=dd.index).astype(str)
G.loc[G.series == 'dji', 'block'] = 'dji_' + G.loc[G.series == 'dji', 'idx'].astype(str)

pos = {s: (ser, run, i) for s, ser, run, i in zip(G.src, G.series, G.run, G.idx)}
edges = set()
for a, b in zip(O.src_a, O.src_b): edges.add((a, b)); edges.add((b, a))
for run, d in G.groupby('run'):
    s = list(d.src); ix = list(d.idx)
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if ix[j] - ix[i] <= BUF_GAP: edges.add((s[i], s[j])); edges.add((s[j], s[i]))
nbr = {}
for a, b in edges: nbr.setdefault(a, set()).add(b)

tot = T[['n0', 'n1', 'n2']].sum().values; n_all = len(T)
blocks = G.groupby('block').agg(tiles=('tiles', 'sum')).reset_index()


def assign(seed):
    rng = np.random.default_rng(seed); order = rng.permutation(blocks.block.values)
    lab, cum = {}, {'test': 0, 'val': 0}
    for b in order:
        t = int(blocks.loc[blocks.block == b, 'tiles'].iloc[0])
        if cum['test'] < 0.16 * n_all: lab[b] = 'test'; cum['test'] += t
        elif cum['val'] < 0.16 * n_all: lab[b] = 'val'; cum['val'] += t
        else: lab[b] = 'train'
    sp = G.block.map(lab).values.copy()
    S = dict(zip(G.src, sp))
    for i, s in enumerate(G.src):        # buffer: drop training groups next to val/test, val groups next to test
        nb = nbr.get(s, ())
        if sp[i] == 'train' and any(S[x] in ('val', 'test') for x in nb): sp[i] = 'drop'
        elif sp[i] == 'val' and any(S[x] == 'test' for x in nb): sp[i] = 'drop'
    return sp


best = None
for seed in range(200):
    sp = assign(seed); G['sp'] = sp; m = T.src.map(dict(zip(G.src, sp)))
    kept = T[m != 'drop']; tk = kept[['n0', 'n1', 'n2']].sum().values; nk = len(kept)
    dev = 0
    for name, target in [('train', .7), ('val', .15), ('test', .15)]:
        sub = T[m == name]; dev += np.abs(sub[['n0', 'n1', 'n2']].sum().values / tk - target).sum() + abs(len(sub) / nk - target)
    dev += 0.5 * (m == 'drop').mean()          # penalise data loss
    if best is None or dev < best[0]: best = (dev, seed, sp.copy())
G['split_stretch'] = best[2]
print('stretch split seed', best[1], 'objective', round(best[0], 4))
lab = dict(zip(G.src, G.split_stretch))
T['split_stretch'] = T.src.map(lab)
# audit: no registered or adjacent pair between train and val/test, or between val and test
S = lab; bad = [(a, b) for a, b in edges if (S[a], S[b]) in {('train', 'test'), ('train', 'val'), ('val', 'test')}]
print('overlapping pairs across subsets after buffering:', len(bad))
print(T.groupby('split_stretch').agg(tiles=('stem', 'size'), groups=('src', 'nunique'), line=('n0', 'sum'), block=('n1', 'sum'), pit=('n2', 'sum')))
T.to_csv('results/tiles_index.csv', index=False)
GI = pd.read_csv('results/tiles_gt_inventory.csv'); GI['split_stretch'] = GI.src.map(lab); GI.to_csv('results/tiles_gt_inventory.csv', index=False)
G[['src', 'series', 'idx', 'run', 'block', 'tiles', 'split_stretch']].to_csv('results/stretch_blocks.csv', index=False)

for part in ['train', 'val', 'test']:
    with open(f'{OUT}/stretch_{part}.txt', 'w') as f:
        for s in T.loc[T.split_stretch == part, 'stem']: f.write(f'{ROOT}/images/{s}.jpg\n')
with open(f'{OUT}/stretch_train_pitx3.txt', 'w') as f:
    for r in T[T.split_stretch == 'train'].itertuples():
        for _ in range(3 if r.n2 > 0 else 1): f.write(f'{ROOT}/images/{r.stem}.jpg\n')
y = dict(path=os.path.abspath(OUT), train='stretch_train.txt', val='stretch_val.txt', test='stretch_test.txt', names={0: 'line', 1: 'block', 2: 'pit'})
yaml.safe_dump(y, open(f'{OUT}/highrpd_stretch.yaml', 'w'))
y['train'] = 'stretch_train_pitx3.txt'; yaml.safe_dump(y, open(f'{OUT}/highrpd_stretch_pitx3.yaml', 'w'))

# partition table: overlap of val/test groups with training groups for all three partitions
rows = []
for col, name in [('split_random', 'Random'), ('split_source', 'Source-separated'), ('split_stretch', 'Stretch-separated')]:
    gl = T.groupby('src')[col].agg(lambda x: set(x))
    trs = set(T[T[col] == 'train'].src)
    for part in ['train', 'val', 'test', 'drop']:
        s = T[T[col] == part]
        if not len(s): continue
        groups = set(s.src)
        ov = np.mean([any(('train' in gl[x]) for x in nbr.get(g, ())) for g in groups]) if part in ('val', 'test') else np.nan
        rows.append(dict(partition=name, subset=part, tiles=len(s), frames=len(groups), line=int(s.n0.sum()), block=int(s.n1.sum()), pit=int(s.n2.sum()),
                         pit_tiles=int((s.n2 > 0).sum()), frames_shared_with_train=(s.src.isin(trs).mean() if part in ('val', 'test') else np.nan),
                         groups_overlapping_train=ov))
pd.DataFrame(rows).to_csv('results/table_splits.csv', index=False); print(pd.DataFrame(rows).round(3).to_string())
