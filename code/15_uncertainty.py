"""Uncertainty budget of the detection-based priority score (tile and section level), YOLO11s, source-separated test set.
Components (standard uncertainties in score units, 0-100):
  A1 repeatability     : seed-to-seed standard deviation of the detection-based score (three seeds), RMS over tiles / sections
  B1 operating point   : half the change of the score between confidence thresholds 0.15 and 0.25, RMS over tiles / sections
  B2 proxy approximation: dispersion of the SAM-mask-to-box ratio per class propagated through the score sensitivities
  B3 reference ambiguity: effect of swapping the line/block label of matched boxes at the observed confusion rate (reference side)
  bias                 : mean difference detection minus reference (reported separately, not folded into the combined uncertainty)
  residual             : standard deviation of (seed-mean detection score - reference - bias), i.e. the observed dispersion
The combined standard uncertainty u_c = sqrt(A1^2 + B1^2 + B2^2 + B3^2) is compared with the observed residual (closure)."""
import sys, os, json, glob, numpy as np, pandas as pd
sys.path.insert(0, 'code'); from inventory import *
from ultralytics import YOLO
RUNS = ['y11s_source_s42', 'y11s_source_s123', 'y11s_source_s3407']
GT = pd.read_csv('results/tiles_gt_inventory.csv'); REF = json.load(open('results/reference_scale_p95.json'))
test = GT[GT.split_source == 'test'].reset_index(drop=True); stems = list(test.stem)
ref = vmps(test, REF).values; ref_sec = section_aggregate(test.assign(VMPS=ref)).S_VMPS
rng = np.random.default_rng(0)


def tile_scores(boxes_by_stem):
    F = pd.DataFrame([tile_features(boxes_by_stem[s]) for s in stems]); return vmps(F, REF).values


def sec_scores(score):
    return section_aggregate(test.assign(VMPS=score)).S_VMPS.loc[ref_sec.index].values


# --- scores per seed at conf 0.25 (saved boxes) and at conf 0.15 (re-predicted)
S25, S15 = [], []
for r in RUNS:
    pb = json.load(open(f'results/runs/{r}/pred_boxes_conf025.json')); S25.append(tile_scores({s: [tuple(b[:5]) for b in pb[s]] for s in stems}))
    cache = f'results/runs/{r}/pred_boxes_conf015.json'
    if not os.path.exists(cache):
        m = YOLO(f'runs/{r}/weights/best.pt'); out = {}
        paths = [os.path.abspath(f'data/HighRPD/images/{s}.jpg') for s in stems]
        for i in range(0, len(paths), 16):
            for res in m.predict(paths[i:i + 16], imgsz=640, conf=0.15, iou=0.7, verbose=False, half=True):
                st = os.path.splitext(os.path.basename(res.path))[0]; b = res.boxes
                out[st] = [(int(c), *map(float, x)) for c, x in zip(b.cls.cpu().numpy(), b.xywhn.cpu().numpy())]
        json.dump(out, open(cache, 'w'))
    pb = json.load(open(cache)); S15.append(tile_scores({s: [tuple(b[:5]) for b in pb[s]] for s in stems}))
S25, S15 = np.array(S25), np.array(S15)
rows = []
for level, R, A, B in [('tile', ref, S25, S15), ('section', ref_sec.values, np.array([sec_scores(s) for s in S25]), np.array([sec_scores(s) for s in S15]))]:
    mean25 = A.mean(0)
    rep = np.sqrt(np.mean(A.std(0, ddof=1) ** 2))
    op = np.sqrt(np.mean((0.5 * (B.mean(0) - mean25)) ** 2))
    bias = np.mean(mean25 - R); resid = np.std(mean25 - R - bias, ddof=1); rmse = np.sqrt(np.mean((mean25 - R) ** 2))
    rows.append(dict(level=level, n=len(R), repeatability=rep, operating_point=op, bias=bias, residual=resid, rmse=rmse))
U = pd.DataFrame(rows).set_index('level')

# --- B2 proxy approximation from the SAM comparison: relative dispersion of mask/box ratio per class, propagated linearly
I = pd.read_csv('results/sam_instances.csv')
rel = {}
for c, key in [(0, 'L_line'), (1, 'A_block'), (2, 'A_pit')]:
    s = I[(I.cls == c) & (I.mask_area > 0)]
    ratio = (s.skel_len / s.bbox_diag) if c == 0 else (s.mask_area / s.bbox_area)
    rel[key] = float(np.std(np.log(ratio), ddof=1))   # relative standard dispersion of the proxy-to-reference ratio
w = WEIGHTS; P95 = REF['P95_raw']
def proxy_u(df_feat):
    c = {k: np.minimum(df_feat[k] / REF[k], 1.0) for k in ['L_line', 'A_block', 'A_pit']}
    var = sum((w[k] * c[k] * rel[k]) ** 2 for k in c)   # clipped components contribute their scaled value times relative dispersion
    return 100 / P95 * np.sqrt(var)
u_proxy_tile = np.sqrt(np.mean(proxy_u(test) ** 2))
sec_feat = section_aggregate(test.assign(VMPS=ref)); u_proxy_sec = np.sqrt(np.mean((0.4 * 0 + 1) ** 2 * (section_aggregate(test.assign(VMPS=proxy_u(test) * 0 + proxy_u(test))).mean_VMPS.loc[ref_sec.index].values) ** 2)) * 0.4
# conservative: proxy errors are correlated within a group, so the section term is 0.4 * mean tile term (mean component) plus 0.4 * P90 term approximated by the same
u_proxy_sec = np.sqrt(np.mean((0.8 * section_aggregate(test.assign(VMPS=proxy_u(test))).mean_VMPS.loc[ref_sec.index].values) ** 2))

# --- B3 reference ambiguity: swap line<->block labels of a fraction p of boxes in the reference, recompute score (50 draws)
D = [json.load(open(f'results/runs/{r}/decomposition.json')) for r in RUNS]
cm = np.array([d['gt_pred_confusion'] for d in D]).mean(0); p_swap = (cm[0, 1] + cm[1, 0]) / cm[:2, :3].sum()
gtb = {s: read_yolo_txt(f'data/HighRPD/labels/{s}.txt') for s in stems}
draws_t, draws_s = [], []
for k in range(50):
    sw = {}
    for s in stems:
        bb = []
        for b in gtb[s]:
            if b[0] in (0, 1) and rng.random() < p_swap: bb.append((1 - b[0],) + tuple(b[1:]))
            else: bb.append(tuple(b))
        sw[s] = bb
    sc = tile_scores(sw); draws_t.append(sc); draws_s.append(sec_scores(sc))
draws_t, draws_s = np.array(draws_t), np.array(draws_s)
u_ref_tile = np.sqrt(np.mean(draws_t.std(0, ddof=1) ** 2)); u_ref_sec = np.sqrt(np.mean(draws_s.std(0, ddof=1) ** 2))
U.loc['tile', 'proxy'] = u_proxy_tile; U.loc['section', 'proxy'] = u_proxy_sec
U.loc['tile', 'reference_ambiguity'] = u_ref_tile; U.loc['section', 'reference_ambiguity'] = u_ref_sec
U['combined_reproduction'] = np.sqrt(U.repeatability ** 2 + U.operating_point ** 2 + U.reference_ambiguity ** 2)   # detector vs annotation-based score
U['closure'] = U.combined_reproduction / U.residual
U['combined_with_proxy'] = np.sqrt(U.combined_reproduction ** 2 + U.proxy ** 2)   # relative to a pixel-level extent measurement
U['expanded_k2'] = 2 * U.combined_with_proxy
U['p_swap'] = p_swap
for k, v in rel.items(): U['rel_' + k] = v
U.round(3).to_csv('results/uncertainty_budget.csv'); print(U.round(2).T)
