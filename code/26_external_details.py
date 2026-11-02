"""Details of the UAV-PDD2023 tilings: boxes lost at the crop border or as fragments narrower than four pixels, boxes
split between tiles, tiles without annotation, and detections on tiles without annotation (zero-shot and fine-tuned
models at the selected threshold).
Output: results/revision/external_details.json"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, 'code'); from inventory import read_yolo_txt
from importlib import import_module
RA = import_module('18_revision_analyses')
SRC = 'data/uavpdd_yolo'; idx = pd.read_csv('results/uavpdd_index.csv'); out = {}
T = json.load(open('results/revision/y11s_stretch/revision.json'))['t_sel']
for name, X0, Y0, S, F, nx, ny in [('native', 16, 32, 640, 1.0, 4, 3), ('gsd_matched', 0, 0, 640, 3.0 / 6.3, 2, 2)]:
    n_src = n_tile = lost = split = 0
    for r in idx.itertuples():
        W, H = 2592, 1944; w2, h2 = round(W * F), round(H * F)
        for c, x, y, w, h in read_yolo_txt(f'{SRC}/labels/{r.stem}.txt'):
            x1, y1, x2, y2 = (x - w / 2) * w2, (y - h / 2) * h2, (x + w / 2) * w2, (y + h / 2) * h2; n_src += 1; k = 0
            for i in range(ny):
                for j in range(nx):
                    x0, y0 = X0 + j * S, Y0 + i * S
                    a, b, cc, d = max(x1, x0), max(y1, y0), min(x2, x0 + S, w2), min(y2, y0 + S, h2)
                    if cc - a >= 4 and d - b >= 4: k += 1
            n_tile += k; lost += (k == 0); split += (k > 1)
    U = pd.read_csv('results/uavpdd_tiles_index.csv' if name == 'native' else 'results/uavpdd_tiles_gsd_index.csv')
    out[name] = dict(source_boxes=n_src, tile_boxes=n_tile, source_boxes_lost=lost, source_boxes_split=split, tiles=len(U), tiles_without_annotation=int((U.n == 0).sum()))
    # detections on tiles without annotation, zero-shot YOLO11s seeds
    e = U[U.n == 0].stem.tolist(); fp = []
    for s in (42, 123, 3407):
        P = RA.predict_cache(f'y11s_stretch_s{s}', [], '', f'uavpdd_{name}', '0')
        fp.append(np.mean([any(b[5] >= T for b in P.get(st, [])) for st in e]))
    out[name]['zero_shot_share_empty_tiles_with_detection'] = [float(np.mean(fp)), float(np.std(fp, ddof=1))]
U = pd.read_csv('results/uavpdd_tiles_index.csv')
for col, tag in [('split', ''), ('split_seq', 'seq_')]:
    Te = U[(U[col] == 'test') & (U.n == 0)].stem.tolist()
    out[f'finetune_{tag}test_tiles_without_annotation'] = len(Te); out[f'finetune_{tag}test_tiles'] = int((U[col] == 'test').sum())
    for run in sorted(os.listdir('runs')):
        if run.startswith('ft_uav_') and (('_seq_' in run) == (tag == 'seq_')) and os.path.exists(f'runs/{run}/weights/best.pt'):
            P = RA.predict_cache(run, U[U[col] == 'test'].stem.tolist(), 'data/uavpdd_tiles/images', 'uavpdd_ft_test', '0')
            out[run + '_share_empty_tiles_with_detection'] = float(np.mean([any(b[5] >= T for b in P.get(st, [])) for st in Te]))
json.dump(out, open('results/revision/external_details.json', 'w'), indent=1); print(json.dumps(out, indent=1))
