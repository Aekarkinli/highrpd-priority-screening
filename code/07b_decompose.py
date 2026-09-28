"""Decompose score error into class confusion, missed tiles and residual (main model runs). Saves predicted boxes."""
import sys, os, json, glob, numpy as np, pandas as pd
sys.path.insert(0, 'code'); from inventory import *
from ultralytics import YOLO
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
GT = pd.read_csv('results/tiles_gt_inventory.csv'); REF = json.load(open('results/reference_scale_p95.json'))
def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1]-a[3]/2, a[2]-a[4]/2, a[1]+a[3]/2, a[2]+a[4]/2; bx1, by1, bx2, by2 = b[1]-b[3]/2, b[2]-b[4]/2, b[1]+b[3]/2, b[2]+b[4]/2
    iw, ih = max(0, min(ax2, bx2)-max(ax1, bx1)), max(0, min(ay2, by2)-max(ay1, by1)); inter = iw*ih
    return inter/(a[3]*a[4]+b[3]*b[4]-inter+1e-9)
def run(name):
    model = YOLO(f'runs/{name}/weights/best.pt'); split_col = 'split_random' if 'random' in name else 'split_source'
    test = GT[GT[split_col] == 'test'].reset_index(drop=True); paths = [os.path.abspath(f'data/HighRPD/images/{s}.jpg') for s in test.stem]
    pred = {}
    for i in range(0, len(paths), 16):
        for r in model.predict(paths[i:i+16], imgsz=640, conf=0.25, iou=0.7, verbose=False, half=True):
            st = os.path.splitext(os.path.basename(r.path))[0]; b = r.boxes
            pred[st] = [(int(c), *map(float, x), float(cf)) for c, x, cf in zip(b.cls.cpu().numpy(), b.xywhn.cpu().numpy(), b.conf.cpu().numpy())]
    json.dump(pred, open(f'results/runs/{name}/pred_boxes_conf025.json', 'w'))
    full, oracle, conf_mat = [], [], np.zeros((3, 4), int)  # rows GT class, cols pred class or unmatched(3)
    n_pred_unmatched = np.zeros(3, int)
    for st in test.stem:
        g = read_yolo_txt(f'data/HighRPD/labels/{st}.txt'); p = [b[:5] for b in pred[st]]
        matched_g = set(); oc = []
        for pb in p:
            best, bj = 0.5, -1
            for j, gb in enumerate(g):
                if j in matched_g: continue
                v = iou(pb, gb)
                if v >= best: best, bj = v, j
            if bj >= 0: matched_g.add(bj); conf_mat[g[bj][0], pb[0]] += 1; oc.append((g[bj][0],) + tuple(pb[1:]))
            else: n_pred_unmatched[pb[0]] += 1; oc.append(pb)
        for j, gb in enumerate(g):
            if j not in matched_g: conf_mat[gb[0], 3] += 1
        full.append(tile_features(p)); oracle.append(tile_features(oc))
    F = pd.DataFrame(full); O = pd.DataFrame(oracle)
    gr, g100 = vmps_raw(test, REF), vmps(test, REF)
    out = dict(name=name, gt_pred_confusion=conf_mat.tolist(), pred_unmatched=n_pred_unmatched.tolist())
    for lab, X in [('full', F), ('oracle_class', O)]:
        r, s = vmps_raw(X, REF), vmps(X, REF); gb, pb = band(g100).astype(str), band(s).astype(str)
        out[lab] = dict(mae=float(np.mean(np.abs(g100-s))), spearman=float(spearmanr(gr, r).correlation), qwk=float(cohen_kappa_score(gb, pb, weights='quadratic', labels=['Low','Moderate','High','Critical'])), band_acc=float((gb == pb).mean()))
    empty = (F[['N_line', 'N_block', 'N_pit']].sum(1) == 0)
    out['empty_tiles_frac'] = float(empty.mean()); out['empty_tiles_gt_score_mean'] = float(g100[empty].mean()); out['mae_nonempty'] = float(np.mean(np.abs(g100[~empty]-vmps(F, REF)[~empty])))
    out['empty_tiles_band'] = band(g100[empty]).value_counts(normalize=True).to_dict()
    json.dump(out, open(f'results/runs/{name}/decomposition.json', 'w'), indent=1); print(name, {k: v for k, v in out.items() if k != 'name'})
if __name__ == '__main__':
    for n in sys.argv[1:]: run(n)
