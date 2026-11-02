"""Controlled contrast of the partitions on common tiles. The tiles in the frame-separated test set that are also in the
stretch-separated validation or test set were used for training by neither model; the frame-separated model has
overlapping neighbouring frames in its training set, the stretch-separated model has none. A third YOLO11s model is
trained on a frame-separated training subset matched in size and class composition (code/23_matched_split.py).
Output: results/revision/leakage_contrast.json"""
import sys, os, json, numpy as np, pandas as pd
from importlib import import_module
sys.path.insert(0, 'code'); from inventory import *
from ultralytics import YOLO
RA = import_module('18_revision_analyses')
T = json.load(open('results/revision/y11s_stretch/revision.json'))['t_sel']
G = pd.read_csv('results/tiles_gt_inventory.csv')
C = G[(G.split_source == 'test') & (G.split_stretch.isin(['val', 'test']))].reset_index(drop=True)
stems = list(C.stem); src = C.src.values
with open('data/splits/common_contrast.txt', 'w') as f: f.write('\n'.join(os.path.abspath(f'data/HighRPD/images/{s}.jpg') for s in stems) + '\n')
with open('data/splits/highrpd_common_contrast.yaml', 'w') as f:
    f.write(f'names:\n  0: line\n  1: block\n  2: pit\npath: {os.path.abspath("data/splits")}\ntrain: common_contrast.txt\nval: common_contrast.txt\ntest: common_contrast.txt\n')
Fg = RA.features({s: [b + (1.0,) for b in read_yolo_txt(f'data/HighRPD/labels/{s}.txt')] for s in stems}, stems); gr, g = RA.score(Fg)
Gref = RA.group_table(Fg, g, stems, src)
out = dict(n_tiles=len(stems), n_frames=int(C.src.nunique()), threshold=T)
for det in ['y11s', 'y8s']:
    for part in ['source', 'stretch', 'sourcematched']:
        if det == 'y8s' and part == 'sourcematched': continue
        L = []
        for seed in [42, 123, 3407]:
            run = f'{det}_{part}_s{seed}'
            v = YOLO(f'runs/{run}/weights/best.pt').val(data='data/splits/highrpd_common_contrast.yaml', split='test', imgsz=640, batch=8, plots=False, verbose=False, project='runs/_val', name=f'{run}_common', exist_ok=True, half=True, workers=0)
            P = RA.predict_cache(run, stems, 'data/HighRPD/images', 'common', '0')
            Fp = RA.features(P, stems, T); pr, p = RA.score(Fp); Gp = RA.group_table(Fp, p, stems, src).loc[Gref.index]
            at = RA.agree(gr, g, pr, p); ag = RA.agree(Gref.S_VMPS.values, Gref.S_VMPS.values, Gp.S_VMPS.values, Gp.S_VMPS.values)
            L.append(dict(map50=float(v.box.map50), map5095=float(v.box.map), tile_rho=at['spearman'], tile_qwk=at['qwk'], tile_mae=at['mae'],
                          frame_rho=ag['spearman'], frame_qwk=ag['qwk'], frame_mae=ag['mae']))
        out[f'{det}_{part}'] = {k: [float(np.mean([x[k] for x in L])), float(np.std([x[k] for x in L], ddof=1))] for k in L[0]}
        print(det, part, {k: np.round(v, 3).tolist() for k, v in out[f'{det}_{part}'].items()}, flush=True)
json.dump(out, open('results/revision/leakage_contrast.json', 'w'), indent=1)
