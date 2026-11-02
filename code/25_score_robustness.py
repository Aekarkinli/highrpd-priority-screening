"""Robustness of the agreement results to the design of the priority score, paired comparisons with the size baselines,
and the dependence of frame-level results on single capture blocks (YOLO11s and YOLOv8s, stretch-separated partition,
selected threshold, three seeds; cached predictions from code/18_revision_analyses.py).
1  Score design: class weights (equal, no multi-class term, pit area or pit count only, pit weights halved, one distress
   type removed), reference percentile of the scaling (P90, P99 instead of P95, refitted on the training tiles),
   frame aggregation weights and band limits. Every variant is applied to annotations and detections alike, and the
   reference ranking of each variant is compared with that of the base design.
2  Paired block bootstrap (same resampled capture blocks for both statistics) of the difference in frame-level Spearman
   correlation between the detection-based score and the size-only predictors.
3  Leave-one-block-out frame agreement; number of frames and tiles per capture block; constant-band baselines.
Outputs: results/revision/score_robustness.json, results/revision/score_robustness.csv"""
import sys, json, numpy as np, pandas as pd
from importlib import import_module
sys.path.insert(0, 'code'); from inventory import WEIGHTS, read_yolo_txt
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
RA = import_module('18_revision_analyses')

GT = pd.read_csv('results/tiles_gt_inventory.csv'); blocks = pd.read_csv('results/stretch_blocks.csv').set_index('src').block
TR = GT[GT.split_stretch == 'train']; TE = GT[GT.split_stretch == 'test'].reset_index(drop=True)
stems = list(TE.stem); src = TE.src.values
T = json.load(open('results/revision/y11s_stretch/revision.json'))['t_sel']
COMP = ['L_line', 'A_block', 'A_pit', 'N_pit']
DET = dict(YOLO11s=[f'y11s_stretch_s{s}' for s in (42, 123, 3407)], YOLOv8s=[f'y8s_stretch_s{s}' for s in (42, 123, 3407)])


def fit(F, w, q):
    ref = {k: float(np.percentile(F[k][F[k] > 0], q)) for k in COMP}
    ref['P_raw'] = float(np.percentile(raw(F, ref, w), q)); return ref


def raw(F, ref, w):
    return sum(w[k] * np.minimum(F[k].values / ref[k], 1.0) for k in COMP) + w['C_multi'] * F['C_multi'].values


def tile_score(F, ref, w):
    r = raw(F, ref, w); return r, 100 * np.minimum(r / ref['P_raw'], 1.0)


def bands(v, lim): return np.digitize(v, lim)        # left-closed bands


def frames(s, lim, a):
    d = pd.DataFrame(dict(s=s, src=src)); g = d.groupby('src', sort=True).s
    mean, p90, crit = g.mean(), g.apply(lambda x: np.percentile(x, 90)), g.apply(lambda x: np.mean(x >= lim[-1]))
    return (a[0] * mean + a[1] * p90 + a[2] * 100 * crit), g.size()


def agree(gr, g, pr, p, lim):
    gi, pi = bands(g, lim), bands(p, lim); n = len(g); k = max(1, int(round(0.1 * n)))
    order = lambda v: np.lexsort((np.arange(n), -np.asarray(v)))
    return dict(rho=spearmanr(gr, pr).correlation, qwk=cohen_kappa_score(gi, pi, weights='quadratic', labels=range(4)), mae=float(np.mean(np.abs(p - g))),
                within1=float(np.mean(np.abs(gi - pi) <= 1)), crit_recall=float(np.mean(pi[gi == 3] == 3)) if (gi == 3).any() else np.nan,
                n_crit=int((gi == 3).sum()), top10=len(set(order(gr)[:k]) & set(order(pr)[:k])) / k)


Fg = RA.features({s: [b + (1.0,) for b in read_yolo_txt(f'data/HighRPD/labels/{s}.txt')] for s in stems}, stems)
P = {d: [RA.features(RA.predict_cache(r, [], '', 'highrpd', '0'), stems, T) for r in runs] for d, runs in DET.items()}

BASE = dict(w=dict(WEIGHTS), q=95, a=(0.4, 0.4, 0.2), lim=[25, 50, 75])
V = {'Base design': {}}
V['Equal score-term weights'] = dict(w=dict(L_line=1, A_block=1, A_pit=1, N_pit=1, C_multi=1))
V['No multi-class term'] = dict(w={**WEIGHTS, 'C_multi': 0})
V['Pit area only'] = dict(w={**WEIGHTS, 'N_pit': 0})
V['Pit count only'] = dict(w={**WEIGHTS, 'A_pit': 0})
V['Pit weights halved'] = dict(w={**WEIGHTS, 'A_pit': 1.25, 'N_pit': 0.25})
V['Without line cracks'] = dict(w={**WEIGHTS, 'L_line': 0})
V['Without block cracks'] = dict(w={**WEIGHTS, 'A_block': 0})
V['Without pits'] = dict(w={**WEIGHTS, 'A_pit': 0, 'N_pit': 0})
V['Scaling at P90'] = dict(q=90)
V['Scaling at P99'] = dict(q=99)
V['Frame: mean only'] = dict(a=(1, 0, 0))
V['Frame: P90 only'] = dict(a=(0, 1, 0))
V['Frame: equal thirds'] = dict(a=(1 / 3, 1 / 3, 1 / 3))
V['Frame: mean and P90'] = dict(a=(0.5, 0.5, 0))
V['Bands 20/45/70'] = dict(lim=[20, 45, 70])
V['Bands 30/55/80'] = dict(lim=[30, 55, 80])

ref0 = fit(TR, BASE['w'], 95); g0r, g0 = tile_score(Fg, ref0, BASE['w']); G0, _ = frames(g0, BASE['lim'], BASE['a'])
rows = []
for name, v in V.items():
    c = {**BASE, **v}; ref = fit(TR, c['w'], c['q']); gr, g = tile_score(Fg, ref, c['w']); Gs, n = frames(g, c['lim'], c['a'])
    k = max(1, int(round(0.1 * len(Gs)))); top = lambda x: set(np.argsort(-np.asarray(x), kind='stable')[:k])
    row = dict(variant=name, ref_rho_tile_vs_base=spearmanr(gr, g0r).correlation, ref_rho_frame_vs_base=spearmanr(Gs, G0).correlation,
               ref_top10_vs_base=len(top(Gs) & top(G0)) / k)
    for d in DET:
        L_t, L_f = [], []
        for Fp in P[d]:
            pr, p = tile_score(Fp, ref, c['w']); Ps, _ = frames(p, c['lim'], c['a'])
            L_t.append(agree(gr, g, pr, p, c['lim'])); L_f.append(agree(Gs.values, Gs.values, Ps.values, Ps.values, c['lim']))
        for lev, L in [('tile', L_t), ('frame', L_f)]:
            for m in ['rho', 'qwk', 'mae', 'within1', 'crit_recall', 'top10', 'n_crit']:
                if lev == 'tile' and m in ('top10',): continue
                row[f'{d}_{lev}_{m}'] = float(np.nanmean([x[m] for x in L]))
    rows.append(row); print(name, {k_: round(v_, 3) for k_, v_ in row.items() if isinstance(v_, float) and 'YOLO11s' in k_ and ('rho' in k_ or 'qwk' in k_)}, flush=True)
R = pd.DataFrame(rows); R.to_csv('results/revision/score_robustness.csv', index=False)

# ---- 2 paired bootstrap of rho differences against size-only predictors (base design)
Gref = RA.group_table(Fg, g0, stems, src); n95 = json.load(open('results/revision/y11s_stretch/revision.json'))['n95']
Gref['S_ext'] = RA.S_ext(Gref, n95, 'n_tiles'); gblk = Gref.index.map(blocks).values
out = dict(variants=rows, paired={}, lobo={}, blocks={})
for d, runs in DET.items():
    Gp = []
    for Fp in P[d]:
        pr, p = tile_score(Fp, ref0, BASE['w']); x = RA.group_table(Fp, p, stems, src).loc[Gref.index].copy(); x['S_ext'] = RA.S_ext(x, n95, 'n_det'); Gp.append(x)

    def stats(idx):
        r = lambda a, b: spearmanr(a[idx], b[idx]).correlation
        S_r, E_r, nt = Gref.S_VMPS.values, Gref.S_ext.values, Gref.n_tiles.values
        o = dict(S=np.mean([r(S_r, x.S_VMPS.values) for x in Gp]), Sext=np.mean([r(E_r, x.S_ext.values) for x in Gp]),
                 ndet_S=np.mean([r(S_r, x.n_det.values) for x in Gp]), ndet_Sext=np.mean([r(E_r, x.n_det.values) for x in Gp]), ntiles_S=r(S_r, nt))
        o['S_minus_ndet'] = o['S'] - o['ndet_S']; o['Sext_minus_ndet'] = o['Sext'] - o['ndet_Sext']; o['S_minus_ntiles'] = o['S'] - o['ntiles_S']
        return o
    point = stats(np.arange(len(Gref))); ci = RA.boot(gblk, stats, 1000, seed=7)
    out['paired'][d] = {k: dict(point=float(point[k]), ci=ci[k], p_le0=None) for k in point}
    # share of bootstrap replicates with a non-positive difference
    rng = np.random.default_rng(7); bl = pd.unique(gblk); pos = {b: np.where(gblk == b)[0] for b in bl}; D_ = []
    for _ in range(1000):
        idx = np.concatenate([pos[b] for b in rng.choice(bl, len(bl), replace=True)]); s = stats(idx); D_.append((s['S_minus_ndet'], s['Sext_minus_ndet'], s['S_minus_ntiles']))
    D_ = np.array(D_)
    for j, k in enumerate(['S_minus_ndet', 'Sext_minus_ndet', 'S_minus_ntiles']): out['paired'][d][k]['p_le0'] = float(np.mean(D_[:, j] <= 0))
    # ---- 3 leave-one-block-out
    L = []
    for b in bl:
        idx = np.where(gblk != b)[0]; s = [agree(Gref.S_VMPS.values[idx], Gref.S_VMPS.values[idx], x.S_VMPS.values[idx], x.S_VMPS.values[idx], BASE['lim']) for x in Gp]
        L.append(dict(block=b, n_frames=int((gblk == b).sum()), rho=float(np.mean([z['rho'] for z in s])), qwk=float(np.mean([z['qwk'] for z in s])), mae=float(np.mean([z['mae'] for z in s]))))
    out['lobo'][d] = L
    print(d, 'paired', {k: (round(v['point'], 3), np.round(v['ci'], 3).tolist(), v['p_le0']) for k, v in out['paired'][d].items()}, flush=True)
    print(d, 'LOBO rho', round(min(x['rho'] for x in L), 3), round(max(x['rho'] for x in L), 3), 'qwk', round(min(x['qwk'] for x in L), 3), round(max(x['qwk'] for x in L), 3), flush=True)
bt = TE.assign(block=TE.src.map(blocks)).groupby('block').agg(frames=('src', 'nunique'), tiles=('stem', 'size'))
out['blocks'] = dict(n_blocks=len(bt), frames_min=int(bt.frames.min()), frames_median=float(bt.frames.median()), frames_max=int(bt.frames.max()),
                     tiles_min=int(bt.tiles.min()), tiles_max=int(bt.tiles.max()), table=bt.reset_index().to_dict('records'))
gi, Gi = bands(g0, BASE['lim']), bands(G0.values, BASE['lim'])
out['constant_baseline'] = dict(tile_mode=int(np.bincount(gi).argmax()), tile_exact=float(np.mean(gi == np.bincount(gi).argmax())),
                                tile_within1=float(np.mean(np.abs(gi - np.bincount(gi).argmax()) <= 1)), frame_mode=int(np.bincount(Gi).argmax()),
                                frame_exact=float(np.mean(Gi == np.bincount(Gi).argmax())), frame_within1=float(np.mean(np.abs(Gi - np.bincount(Gi).argmax()) <= 1)),
                                frame_band_counts=np.bincount(Gi, minlength=4).tolist(), tile_band_counts=np.bincount(gi, minlength=4).tolist())
print('blocks', {k: v for k, v in out['blocks'].items() if k != 'table'}); print('constant', out['constant_baseline'])
json.dump(out, open('results/revision/score_robustness.json', 'w'), indent=1, default=float)
