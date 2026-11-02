"""Sequence-separated partition of UAV-PDD2023 for fine-tuning.
The view audit (code/32_uavpdd_view_audit.py) found overlap only between views whose identifiers differ by at most
three. Views are ordered by identifier and cut into blocks of eight consecutive views, which are assigned to training,
validation and test with nominal 70/15/15 proportions; among 200 assignments the one with class proportions closest to
the whole set is kept. Training views within three identifiers of a validation or test view, and validation views within
three identifiers of a test view, are dropped. All cross-subset image pairs with identifiers up to ten apart are then
registered as in the view audit to confirm the separation.
Outputs: data/uavpdd_tiles_seq/{train,val,test}.txt and uavpdd_tiles_seq.yaml, results/uavpdd_tiles_index.csv (column
split_seq), results/revision/uavpdd_seq_split.json"""
import os, json, numpy as np, pandas as pd, cv2

U = pd.read_csv('results/uavpdd_tiles_index.csv'); OUT = 'data/uavpdd_tiles_seq'; os.makedirs(OUT, exist_ok=True)
views = np.array(sorted(U.view.unique())); blocks = np.arange(len(views)) // 8
V = U.groupby('view').agg(n0=('n0', 'sum'), n1=('n1', 'sum'), n2=('n2', 'sum'), tiles=('stem', 'size')).loc[views]
target = (V[['n0', 'n1', 'n2']].sum() / V[['n0', 'n1', 'n2']].sum().sum()).values
nb = blocks.max() + 1; rng = np.random.default_rng(2026); best = None


def assign(order):
    lab = np.empty(nb, dtype=object); k_te = int(round(0.15 * nb)); k_va = int(round(0.15 * nb))
    lab[order[:k_te]] = 'test'; lab[order[k_te:k_te + k_va]] = 'val'; lab[order[k_te + k_va:]] = 'train'
    sp = pd.Series(lab[blocks], index=views)
    for a, others in [('train', ['val', 'test']), ('val', ['test'])]:             # buffer of three identifiers
        near = [v for v in sp.index[sp == a] if any((abs(v - w) <= 3) for w in sp.index[sp.isin(others)])]
        sp[near] = 'drop'
    return sp


for k in range(200):
    sp = assign(rng.permutation(nb)); d = 0
    for part in ['train', 'val', 'test']:
        c = V.loc[sp.index[sp == part], ['n0', 'n1', 'n2']].sum().values; d += np.abs(c / c.sum() - target).sum()
    d += (sp == 'drop').mean()
    if best is None or d < best[0]: best = (d, sp)
sp = best[1]; U['split_seq'] = U.view.map(sp); U.to_csv('results/uavpdd_tiles_index.csv', index=False)
for part in ['train', 'val', 'test']:
    open(f'{OUT}/{part}.txt', 'w').write(''.join(os.path.abspath(f'data/uavpdd_tiles/images/{s}.jpg') + '\n' for s in U[U.split_seq == part].stem))
with open(f'{OUT}/uavpdd_tiles_seq.yaml', 'w') as f:
    f.write(f'names:\n  0: line\n  1: block\n  2: pit\npath: {os.path.abspath(OUT)}\ntrain: train.txt\nval: val.txt\ntest: test.txt\n')
summ = U.groupby('split_seq').agg(tiles=('stem', 'size'), views=('view', 'nunique'), line=('n0', 'sum'), block=('n1', 'sum'), pit=('n2', 'sum'), empty=('n', lambda x: int((x == 0).sum())))
print(summ)

# registration check of nearby cross-subset image pairs
I = U.groupby('image').agg(view=('view', 'first'), split=('split_seq', 'first')); I = I[I.split != 'drop']
sift = cv2.SIFT_create(nfeatures=4000); flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48)); D = {}


def feats(img):
    if img not in D:
        im = cv2.resize(cv2.imread(f'data/external/JPEGImages/{img}.jpg', cv2.IMREAD_GRAYSCALE), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        kp, des = sift.detectAndCompute(im, None); D[img] = (np.float32([k.pt for k in kp]), des if des is not None else np.zeros((0, 128), np.float32))
    return D[img]


rows = []
imgs = list(I.index)
for i, a in enumerate(imgs):
    for b in imgs[i + 1:]:
        if I.split[a] == I.split[b] or abs(I.view[a] - I.view[b]) > 10: continue
        (pa, da), (pb, db) = feats(a), feats(b); r = dict(img_a=a, img_b=b, split_a=I.split[a], split_b=I.split[b], gap=abs(I.view[a] - I.view[b]), inliers=0)
        if len(da) > 20 and len(db) > 20:
            m = [x for x, y in (p for p in flann.knnMatch(da, db, k=2) if len(p) == 2) if x.distance < 0.75 * y.distance]
            if len(m) >= 12:
                M, inl = cv2.estimateAffinePartial2D(pa[[x.queryIdx for x in m]], pb[[x.trainIdx for x in m]], method=cv2.RANSAC, ransacReprojThreshold=4.0)
                if M is not None: r.update(inliers=int(inl.sum()), scale=float(np.hypot(M[0, 0], M[1, 0])))
        rows.append(r)
P = pd.DataFrame(rows); F = P[(P.inliers >= 30) & P.scale.between(0.8, 1.25)] if len(P) else P
out = dict(block_size=8, buffer_ids=3, subsets=summ.reset_index().to_dict('records'), checked_pairs=len(P), flagged_pairs=len(F),
           max_inliers=int(P.inliers.max()) if len(P) else 0, flagged=F.to_dict('records'))
json.dump(out, open('results/revision/uavpdd_seq_split.json', 'w'), indent=1, default=int)
print({k: v for k, v in out.items() if k not in ('flagged', 'subsets')})
