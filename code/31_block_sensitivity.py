"""Sensitivity of the frame-level block bootstrap to the definition of the capture blocks.
Two of the five 16-frame test blocks (DJI 214-230 and 231-246) are adjacent in the capture sequence, and two of the
single frames (dji 43 and 46) are three frames apart. The intervals of the frame rank correlation and of its difference
to the count of tiles with detections are recomputed with these blocks merged.
Output: results/revision/block_sensitivity.json"""
import sys, json, numpy as np, pandas as pd
from importlib import import_module
from scipy.stats import spearmanr
sys.path.insert(0, 'code'); from inventory import *
RA = import_module('18_revision_analyses')
GT = pd.read_csv('results/tiles_gt_inventory.csv'); d = GT[GT.split_stretch == 'test'].reset_index(drop=True)
stems = list(d.stem); src = d.src.values
blocks = pd.read_csv('results/stretch_blocks.csv').set_index('src').block
Fg = RA.features({s: [b + (1.0,) for b in read_yolo_txt(f'data/HighRPD/labels/{s}.txt')] for s in stems}, stems); _, g = RA.score(Fg)
Gref = RA.group_table(Fg, g, stems, src)
DEF = {'eight blocks': {}, 'adjacent blocks merged': {'DJI_r2_b7': 'DJI_r2_b6'},
       'adjacent blocks and close single frames merged': {'DJI_r2_b7': 'DJI_r2_b6', 'dji_46': 'dji_43'}}
out = {}
for det in ['y11s', 'y8s']:
    rv = json.load(open(f'results/revision/{det}_stretch/revision.json')); T = rv['t_sel']; Gp = []
    for r in rv['runs']:
        P = json.load(open(f'results/runs/{r}/pred_highrpd_conf005.json'))
        Fp = RA.features(P, stems, T); _, p = RA.score(Fp); Gp.append(RA.group_table(Fp, p, stems, src).loc[Gref.index])

    def stats(idx):
        r = lambda a, b: spearmanr(a[idx], b[idx]).correlation; S = Gref.S_VMPS.values
        o = dict(rho=np.mean([r(S, x.S_VMPS.values) for x in Gp]), rho_ndet=np.mean([r(S, x.n_det.values) for x in Gp]))
        o['diff'] = o['rho'] - o['rho_ndet']; return o
    out[det] = {}
    for name, m in DEF.items():
        gblk = Gref.index.map(lambda s: m.get(blocks[s], blocks[s])).values
        ci = RA.boot(gblk, stats, 1000, seed=7)
        rng = np.random.default_rng(7); bl = pd.unique(gblk); pos = {b: np.where(gblk == b)[0] for b in bl}
        D = [stats(np.concatenate([pos[b] for b in rng.choice(bl, len(bl), replace=True)]))['diff'] for _ in range(1000)]
        out[det][name] = dict(n_blocks=len(bl), point=stats(np.arange(len(Gref))), ci=ci, diff_le0=float(np.mean(np.array(D) <= 0)))
        print(det, name, json.dumps(out[det][name], default=float), flush=True)
json.dump(out, open('results/revision/block_sensitivity.json', 'w'), indent=1, default=float)
