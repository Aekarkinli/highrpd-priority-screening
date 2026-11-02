"""Effect of the three validation frames that overlap training frames on checkpoint selection.
For every stretch-separated main run the best checkpoint (selected on the full validation set) and the last checkpoint
(epoch 50, selected without any validation data) are compared on the validation set without the three frames and on
the test set. Early stopping did not end any of these runs, since all trained for the full 50 epochs.
Output: results/revision/checkpoint_check.json"""
import sys, os, json, numpy as np, pandas as pd
from importlib import import_module
sys.path.insert(0, 'code'); from inventory import *
from ultralytics import YOLO
RA = import_module('18_revision_analyses')
V = json.load(open('results/revision/val_overlap_check.json'))
GT = pd.read_csv('results/tiles_gt_inventory.csv')
val = GT[(GT.split_stretch == 'val') & ~GT.src.isin(V['val_frames'])]
with open('data/splits/stretch_val_clean.txt', 'w') as f: f.write('\n'.join(os.path.abspath(f'data/HighRPD/images/{s}.jpg') for s in val.stem) + '\n')
with open('data/splits/highrpd_stretch_cleanval.yaml', 'w') as f:
    f.write(f'names:\n  0: line\n  1: block\n  2: pit\npath: {os.path.abspath("data/splits")}\ntrain: stretch_train.txt\nval: stretch_val_clean.txt\ntest: stretch_test.txt\n')
d = GT[GT.split_stretch == 'test'].reset_index(drop=True); stems = list(d.stem); src = d.src.values
Fg = RA.features({s: [b + (1.0,) for b in read_yolo_txt(f'data/HighRPD/labels/{s}.txt')] for s in stems}, stems); gr, g = RA.score(Fg)
Gref = RA.group_table(Fg, g, stems, src)
out = dict(clean_val_frames=int(val.src.nunique()), clean_val_tiles=len(val))
for det in ['y11s', 'y8s']:
    T = json.load(open(f'results/revision/{det}_stretch/revision.json'))['t_sel']
    for seed in [42, 123, 3407]:
        run = f'{det}_stretch_s{seed}'; res = {}
        for ck in ['best', 'last']:
            m = YOLO(f'runs/{run}/weights/{ck}.pt'); kw = dict(imgsz=640, batch=8, plots=False, verbose=False, project='runs/_val', exist_ok=True, half=True, workers=0)
            vc = m.val(data='data/splits/highrpd_stretch_cleanval.yaml', split='val', name=f'{run}_{ck}_cleanval', **kw)
            vt = m.val(data='data/splits/highrpd_stretch_cleanval.yaml', split='test', name=f'{run}_{ck}_test', **kw)
            if ck == 'best': P = json.load(open(f'results/runs/{run}/pred_highrpd_conf005.json'))
            else:
                P = {}; paths = [os.path.abspath(f'data/HighRPD/images/{s}.jpg') for s in stems]
                for i in range(0, len(paths), 16):
                    for r in m.predict(paths[i:i + 16], imgsz=640, conf=0.05, iou=0.7, verbose=False, half=True):
                        b = r.boxes; P[os.path.splitext(os.path.basename(r.path))[0]] = [[int(c), *map(float, x), float(p)] for c, x, p in zip(b.cls.cpu().numpy(), b.xywhn.cpu().numpy(), b.conf.cpu().numpy())]
            Fp = RA.features(P, stems, T); pr, p = RA.score(Fp); Gp = RA.group_table(Fp, p, stems, src).loc[Gref.index]
            at = RA.agree(gr, g, pr, p); ag = RA.agree(Gref.S_VMPS.values, Gref.S_VMPS.values, Gp.S_VMPS.values, Gp.S_VMPS.values)
            res[ck] = dict(cleanval_fitness=float(vc.fitness), cleanval_map50=float(vc.box.map50), test_map50=float(vt.box.map50),
                           tile_rho=float(at['spearman']), tile_qwk=at['qwk'], frame_rho=float(ag['spearman']), frame_qwk=ag['qwk'])
        out[run] = res; print(run, json.dumps(res), flush=True)
for det in ['y11s', 'y8s']:
    for ck in ['best', 'last']:
        L = [out[f'{det}_stretch_s{s}'][ck] for s in [42, 123, 3407]]
        out[f'{det}_{ck}_mean'] = {k: [float(np.mean([x[k] for x in L])), float(np.std([x[k] for x in L], ddof=1))] for k in L[0]}
json.dump(out, open('results/revision/checkpoint_check.json', 'w'), indent=1)
