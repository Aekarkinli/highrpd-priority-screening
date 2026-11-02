"""Threshold selection repeated without the validation frames that the exhaustive audit (code/24_overlap_audit.py) found
to overlap training frames. Output: results/revision/val_overlap_check.json"""
import sys, json, numpy as np, pandas as pd
from importlib import import_module
sys.path.insert(0, 'code'); from inventory import read_yolo_txt
RA = import_module('18_revision_analyses')
A = json.load(open('results/revision/overlap_audit.json')); fl = A['flagged']
G = pd.read_csv('results/tiles_gt_inventory.csv'); V = G[G.split_stretch == 'val']
bad = sorted({f['src_a'] for f in fl if f['split_a'] == 'val'} | {f['src_b'] for f in fl if f['split_b'] == 'val'})
out = dict(flagged_pairs=[(f['src_a'], f['src_b'], f['split_a'], f['split_b'], f['inliers'], round(f['rot_deg'], 2)) for f in fl], val_frames=bad,
           val_tiles_removed=int(V.src.isin(bad).sum()), val_tiles=len(V))
for det in ['y11s', 'y8s']:
    runs = [f'{det}_stretch_s{s}' for s in (42, 123, 3407)]
    for name, d in [('all', V), ('clean', V[~V.src.isin(bad)])]:
        stems = list(d.stem); Fg = RA.features({s: [b + (1.0,) for b in read_yolo_txt(f'data/HighRPD/labels/{s}.txt')] for s in stems}, stems); gr, g = RA.score(Fg)
        curve = []
        for t in RA.GRID:
            q = [RA.agree(gr, g, *RA.score(RA.features(RA.predict_cache(r, [], '', 'highrpd', '0'), stems, t)))['qwk'] for r in runs]
            curve.append((t, float(np.mean(q))))
        best = max(curve, key=lambda x: x[1]); out[f'{det}_{name}'] = dict(t_sel=best[0], qwk=best[1], curve=curve)
        print(det, name, best)
json.dump(out, open('results/revision/val_overlap_check.json', 'w'), indent=1); print({k: v for k, v in out.items() if 'y' not in k[:2]})
