"""Evaluate finished runs: detection metrics, inventory error, VMPS error propagation, section-level agreement, external data."""
import sys, os, json, glob, numpy as np, pandas as pd
sys.path.insert(0, 'code'); from inventory import *
from ultralytics import YOLO
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix, precision_recall_fscore_support
CONF = 0.25; IOU = 0.7; BANDS_L = ['Low', 'Moderate', 'High', 'Critical']
GT = pd.read_csv('results/tiles_gt_inventory.csv'); REF = json.load(open('results/reference_scale_p95.json'))


def score(F):
    return vmps_raw(F, REF), vmps(F, REF)


def predict_tiles(model, paths, imgsz=640, conf=CONF):
    out = {}
    for i in range(0, len(paths), 16):
        for res in model.predict(paths[i:i + 16], imgsz=imgsz, conf=conf, iou=IOU, verbose=False, half=True):
            st = os.path.splitext(os.path.basename(res.path))[0]
            b = res.boxes; out[st] = [(int(c), *xywhn) for c, xywhn in zip(b.cls.cpu().numpy(), b.xywhn.cpu().numpy())]
    return out


def agreement(gt_raw, gt_100, pr_raw, pr_100, gt_F=None, pr_F=None, k_frac=0.05):
    gb = band(pd.Series(gt_100)).astype(str); pb = band(pd.Series(pr_100)).astype(str)
    cm = confusion_matrix(gb, pb, labels=BANDS_L)
    gi = np.array([BANDS_L.index(x) for x in gb]); pi = np.array([BANDS_L.index(x) for x in pb])
    d = dict(n=len(gt_raw), mae=np.mean(np.abs(gt_100 - pr_100)), rmse=np.sqrt(np.mean((gt_100 - pr_100) ** 2)), bias=np.mean(pr_100 - gt_100),
             spearman=spearmanr(gt_raw, pr_raw).correlation, kendall=kendalltau(gt_raw, pr_raw).correlation,
             band_acc=(gb == pb).mean(), qwk=cohen_kappa_score(gb, pb, weights='quadratic', labels=BANDS_L),
             wf1=f1_score(gb, pb, average='weighted', labels=BANDS_L),
             crit_recall=cm[3, 3] / max(cm[3].sum(), 1), crit_prec=cm[3, 3] / max(cm[:, 3].sum(), 1),
             highcrit_recall=cm[2:, 2:].sum() / max(cm[2:].sum(), 1), adjacent_acc=np.mean(np.abs(gi - pi) <= 1),
             under2=np.mean(pi <= gi - 2))
    k = max(1, int(k_frac * len(gt_raw))); tg = set(np.argsort(-np.asarray(gt_raw))[:k]); tp = set(np.argsort(-np.asarray(pr_raw))[:k])
    d['topk_overlap'] = len(tg & tp) / k; d['k'] = k; d['cm'] = cm.tolist()
    if gt_F is not None:
        for c in ['N_line', 'N_block', 'N_pit']:
            d['mae_' + c] = np.mean(np.abs(gt_F[c].values - pr_F[c].values)); d['bias_' + c] = np.mean(pr_F[c].values - gt_F[c].values)
        p, r, f, _ = precision_recall_fscore_support(gt_F.N_pit > 0, pr_F.N_pit > 0, average='binary', zero_division=0)
        d.update(pit_presence_p=p, pit_presence_r=r, pit_presence_f1=f)
    return d


def eval_run(name, ext=True):
    outdir = f'results/runs/{name}'; os.makedirs(outdir, exist_ok=True)
    if os.path.exists(f'{outdir}/summary.json'): return json.load(open(f'{outdir}/summary.json'))
    meta = json.load(open(f'runs/{name}/meta.json')); model = YOLO(f'runs/{name}/weights/best.pt')
    split_col = 'split_random' if 'random' in meta['data'] else 'split_source'
    S = dict(name=name, **meta, split=split_col)
    # 1 detection metrics on the test split
    v = model.val(data=meta['data'], split='test', imgsz=640, batch=8, plots=False, verbose=False, project='runs/_val', name=name, exist_ok=True, half=True)
    S['det'] = dict(map50=float(v.box.map50), map5095=float(v.box.map), mp=float(v.box.mp), mr=float(v.box.mr),
                    ap50_cls=[float(x) for x in v.box.ap50], ap_cls=[float(x) for x in v.box.ap], p_cls=[float(x) for x in v.box.p], r_cls=[float(x) for x in v.box.r])
    # 2 predictions at deployment threshold -> inventory -> VMPS
    test = GT[GT[split_col] == 'test'].reset_index(drop=True)
    paths = [os.path.abspath(f'data/HighRPD/images/{s}.jpg') for s in test.stem]
    pred = predict_tiles(model, paths); PF = pd.DataFrame([tile_features(pred[s]) for s in test.stem]); PF['stem'] = test.stem
    gr, g100 = score(test); pr, p100 = score(PF)
    PF['VMPS_pred'] = p100.values; PF['raw_pred'] = pr.values; PF['VMPS_gt'] = g100.values; PF['raw_gt'] = gr.values; PF['src'] = test.src.values
    PF.to_csv(f'{outdir}/tile_pred_inventory.csv', index=False)
    S['tile'] = agreement(gr.values, g100.values, pr.values, p100.values, test, PF)
    # 3 section level
    tg = test.assign(VMPS=g100.values); tp = PF.assign(VMPS=p100.values)
    sg = section_aggregate(tg); sp = section_aggregate(tp).loc[sg.index]
    S['section'] = agreement(sg.S_VMPS.values, sg.S_VMPS.values, sp.S_VMPS.values, sp.S_VMPS.values, k_frac=0.1)
    sec = S['section']
    sec['spearman_mean'] = spearmanr(sg.mean_VMPS, sp.mean_VMPS).correlation; sec['spearman_p90'] = spearmanr(sg.p90_VMPS, sp.p90_VMPS).correlation
    sec['spearman_sumL'] = spearmanr(sg.sum_L_line, sp.sum_L_line).correlation; sec['spearman_sumAb'] = spearmanr(sg.sum_A_block, sp.sum_A_block).correlation
    sec['spearman_Npit'] = spearmanr(sg.N_pit, sp.N_pit).correlation; sec['mae_Npit'] = float(np.mean(np.abs(sg.N_pit - sp.N_pit))); sec['n_sections'] = len(sg)
    sg.join(sp, lsuffix='_gt', rsuffix='_pred').to_csv(f'{outdir}/section_pred_inventory.csv')
    # 4 confidence threshold sensitivity (tile level)
    sens = []
    for cf in [0.1, 0.15, 0.25, 0.35, 0.5]:
        pd_ = predict_tiles(model, paths, conf=cf); F = pd.DataFrame([tile_features(pd_[s]) for s in test.stem]); r_, s_ = score(F)
        a = agreement(gr.values, g100.values, r_.values, s_.values, test, F)
        sens.append(dict(conf=cf, **{k: a[k] for k in ['mae', 'spearman', 'qwk', 'band_acc', 'crit_recall', 'mae_N_line', 'mae_N_block', 'mae_N_pit', 'pit_presence_f1']}))
    S['conf_sens'] = sens
    # 5 external UAV-PDD2023 tiles (zero-shot)
    if ext:
        U = pd.read_csv('results/uavpdd_tiles_index.csv')
        vu = model.val(data='data/uavpdd_tiles/uavpdd_tiles_all.yaml', split='test', imgsz=640, batch=8, plots=False, verbose=False, project='runs/_val', name=name + '_ext', exist_ok=True, half=True)
        S['ext_det'] = dict(map50=float(vu.box.map50), map5095=float(vu.box.map), ap50_cls=[float(x) for x in vu.box.ap50], p_cls=[float(x) for x in vu.box.p], r_cls=[float(x) for x in vu.box.r])
        up = [os.path.abspath(f'data/uavpdd_tiles/images/{s}.jpg') for s in U.stem]
        pu = predict_tiles(model, up)
        UG = pd.DataFrame([tile_features(read_yolo_txt(f'data/uavpdd_tiles/labels/{s}.txt')) for s in U.stem]); UP = pd.DataFrame([tile_features(pu[s]) for s in U.stem])
        gr_, g100_ = score(UG); pr_, p100_ = score(UP)
        E = U[['stem', 'image', 'view', 'quadrant']].copy(); E['VMPS_gt'] = g100_.values; E['VMPS_pred'] = p100_.values; E['raw_gt'] = gr_.values; E['raw_pred'] = pr_.values
        for c in ['N_line', 'N_block', 'N_pit', 'L_line', 'A_block', 'A_pit']: E[c + '_gt'] = UG[c].values; E[c + '_pred'] = UP[c].values
        E.to_csv(f'{outdir}/ext_tile_inventory.csv', index=False)
        S['ext_tile'] = agreement(gr_.values, g100_.values, pr_.values, p100_.values, UG, UP)
        S['ext_band_gt'] = band(pd.Series(g100_)).value_counts(normalize=True).to_dict(); S['ext_band_pred'] = band(pd.Series(p100_)).value_counts(normalize=True).to_dict()
        vg = section_aggregate(UG.assign(VMPS=g100_.values, stem=U.stem.values, src=U.view.values)); vp = section_aggregate(UP.assign(VMPS=p100_.values, stem=U.stem.values, src=U.view.values)).loc[vg.index]
        S['ext_section'] = agreement(vg.S_VMPS.values, vg.S_VMPS.values, vp.S_VMPS.values, vp.S_VMPS.values, k_frac=0.1)
    json.dump(S, open(f'{outdir}/summary.json', 'w'), indent=1, default=float); return S


if __name__ == '__main__':
    names = sys.argv[1:] or [os.path.basename(os.path.dirname(p)) for p in sorted(glob.glob('runs/*/meta.json'))]
    for n in names:
        if n.startswith('_'): continue
        S = eval_run(n)
        print(n, 'mAP50 %.3f mAP %.3f | tile rho %.3f qwk %.3f critR %.3f | sect rho %.3f' % (S['det']['map50'], S['det']['map5095'], S['tile']['spearman'], S['tile']['qwk'], S['tile']['crit_recall'], S['section']['spearman']))
