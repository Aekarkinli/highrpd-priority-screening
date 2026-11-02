"""Evaluate fine-tuned detectors on the UAV-PDD2023 test split (detection + score agreement). Runs whose name contains
'_seq_' use the sequence-separated partition of code/33_uavpdd_seq_split.py, the others the random view partition."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, 'code'); from inventory import *
from ultralytics import YOLO
from importlib import import_module
ev = import_module('07_evaluate')
U = pd.read_csv('results/uavpdd_tiles_index.csv')
REF = json.load(open('results/reference_scale_p95.json'))
CONF_SEL = json.load(open('results/revision/y11s_stretch/revision.json'))['t_sel']   # threshold selected on HighRPD validation tiles
def run(name):
    outdir = f'results/runs/{name}'; os.makedirs(outdir, exist_ok=True); seq = '_seq_' in name
    T = U[U['split_seq' if seq else 'split'] == 'test'].reset_index(drop=True)
    yml = 'data/uavpdd_tiles_seq/uavpdd_tiles_seq.yaml' if seq else 'data/uavpdd_tiles/uavpdd_tiles.yaml'
    model = YOLO(f'runs/{name}/weights/best.pt'); meta = json.load(open(f'runs/{name}/meta.json'))
    v = model.val(data=yml, split='test', imgsz=640, batch=8, plots=False, verbose=False, project='runs/_val', name=name, exist_ok=True, half=True)
    S = dict(name=name, **meta, conf=CONF_SEL, det=dict(map50=float(v.box.map50), map5095=float(v.box.map), ap50_cls=[float(x) for x in v.box.ap50], p_cls=[float(x) for x in v.box.p], r_cls=[float(x) for x in v.box.r]))
    paths = [os.path.abspath(f'data/uavpdd_tiles/images/{s}.jpg') for s in T.stem]
    pu = ev.predict_tiles(model, paths, conf=CONF_SEL)
    UG = pd.DataFrame([tile_features(read_yolo_txt(f'data/uavpdd_tiles/labels/{s}.txt')) for s in T.stem]); UP = pd.DataFrame([tile_features(pu[s]) for s in T.stem])
    gr, g100 = ev.score(UG); pr, p100 = ev.score(UP)
    S['tile'] = ev.agreement(gr.values, g100.values, pr.values, p100.values, UG, UP)
    S['band_gt'] = band(g100).value_counts(normalize=True).to_dict(); S['band_pred'] = band(p100).value_counts(normalize=True).to_dict()
    vg = section_aggregate(UG.assign(VMPS=g100.values, stem=T.stem.values, src=T.view.values)); vp = section_aggregate(UP.assign(VMPS=p100.values, stem=T.stem.values, src=T.view.values)).loc[vg.index]
    S['section'] = ev.agreement(vg.S_VMPS.values, vg.S_VMPS.values, vp.S_VMPS.values, vp.S_VMPS.values, k_frac=0.1)
    E = T[['stem', 'view']].copy(); E['VMPS_gt'] = g100.values; E['VMPS_pred'] = p100.values; E.to_csv(f'{outdir}/ext_test_tile_inventory.csv', index=False)
    json.dump(S, open(f'{outdir}/summary_ft.json', 'w'), indent=1, default=float)
    print(name, 'mAP50 %.3f cls %s | tile rho %.3f qwk %.3f | view rho %.3f' % (S['det']['map50'], [round(x, 3) for x in S['det']['ap50_cls']], S['tile']['spearman'], S['tile']['qwk'], S['section']['spearman']))
if __name__ == '__main__':
    for n in sys.argv[1:]: run(n)
