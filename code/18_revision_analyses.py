"""Analyses of the revised protocol for one detector configuration evaluated over several training seeds.
Usage: python code/18_revision_analyses.py <tag> <run1,run2,...> [--split split_stretch] [--device 0|cpu] [--ext]

1  Predictions on the validation and test tiles are cached once per run at confidence 0.05; any higher operating
   threshold is obtained by filtering, which reproduces a prediction at that threshold because non-maximum suppression
   is greedy in confidence.
2  Operating threshold: chosen on the validation tiles (grid 0.05-0.50) by the tile-level quadratic weighted kappa of
   the screening bands, averaged over seeds; the test tiles are evaluated at the selected threshold and at 0.25.
3  Tile- and group-level agreement with block bootstrap intervals (capture blocks resampled with replacement,
   1000 replicates; metrics averaged over seeds within each replicate), constant-band baseline, confusion matrices.
4  Group summaries: mean, P90, share of Critical tiles, combined score, maximum, an extent-aware score and the number of
   tiles with detections as a size-only predictor; totals against per-tile densities; pseudo-groups of equal size.
5  Line-box merging (intersection over the smaller box > 0.6) as a sensitivity of the line-crack proxy.
6  Error decomposition: bias, single-seed error dispersion, training variability and its variance share, share of the
   squared error on tiles without detections; sensitivities to the threshold range 0.15-0.25 (score evaluated on 11
   thresholds, uniform weighting), to hypothetical line/block label changes (5 and 10 %) and to the proxy approximation
   (Monte Carlo with within-group correlation 0, 0.5 and 1).
7  (--ext) Zero-shot UAV-PDD2023 at native resolution and resampled to the HighRPD ground sampling distance.
Outputs: results/revision/<tag>/*.json / *.csv"""
import sys, os, json, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'code'); from inventory import *
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import cohen_kappa_score, confusion_matrix

BL = ['Low', 'Moderate', 'High', 'Critical']
GRID = [round(x, 2) for x in np.arange(0.05, 0.501, 0.05)]
REF = json.load(open('results/reference_scale_p95.json'))
TILE_M = 640 * 0.0063            # tile side in metres
rng0 = np.random.default_rng(0)


# ----------------------------------------------------------------------------------------------------------- helpers
def predict_cache(run, stems, img_dir, key, device):
    path = f'results/runs/{run}/pred_{key}_conf005.json'
    if os.path.exists(path): return json.load(open(path))
    from ultralytics import YOLO
    m = YOLO(f'runs/{run}/weights/best.pt'); out = {}
    paths = [os.path.abspath(f'{img_dir}/{s}.jpg') for s in stems]
    for i in range(0, len(paths), 16):
        for r in m.predict(paths[i:i + 16], imgsz=640, conf=0.05, iou=0.7, verbose=False, device=device, half=(device != 'cpu')):
            b = r.boxes; st = os.path.splitext(os.path.basename(r.path))[0]
            out[st] = [[int(c), *map(float, x), float(p)] for c, x, p in zip(b.cls.cpu().numpy(), b.xywhn.cpu().numpy(), b.conf.cpu().numpy())]
    os.makedirs(os.path.dirname(path), exist_ok=True); json.dump(out, open(path, 'w')); return out


def merge_lines(bs):
    """Merge line boxes whose intersection exceeds 60 % of the smaller box into their union rectangle (repeated)."""
    L = [list(b[:5]) for b in bs if b[0] == 0]; other = [tuple(b[:5]) for b in bs if b[0] != 0]
    changed = True
    while changed and len(L) > 1:
        changed = False
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                a, b = L[i], L[j]
                ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
                bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
                iw, ih = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
                if iw > 0 and ih > 0 and iw * ih > 0.6 * min(a[3] * a[4], b[3] * b[4]):
                    x1, y1, x2, y2 = min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2)
                    L[i] = [0, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]; L.pop(j); changed = True; break
            if changed: break
    return [tuple(b) for b in L] + other


def features(boxes_by_stem, stems, t=None, merge=False):
    rows = []
    for s in stems:
        bs = boxes_by_stem.get(s, [])
        if t is not None: bs = [b for b in bs if b[5] >= t]
        bs = [tuple(b[:5]) for b in bs]
        if merge: bs = merge_lines(bs)
        f = tile_features(bs); f['n_boxes'] = len(bs); rows.append(f)
    return pd.DataFrame(rows)


def score(F): return vmps_raw(F, REF).values, vmps(F, REF).values


def bands_idx(v): return np.array([BL.index(x) for x in band(pd.Series(v)).astype(str)])


def agree(gr, g, pr, p, k_frac=0.1):
    gi, pi = bands_idx(g), bands_idx(p); n = len(g); k = max(1, int(round(k_frac * n)))
    order = lambda v: np.lexsort((np.arange(n), -np.asarray(v)))          # ties broken by position (group identifier order)
    tg, tp = set(order(gr)[:k]), set(order(pr)[:k])
    cm = confusion_matrix(gi, pi, labels=range(4))
    return dict(n=n, spearman=spearmanr(gr, pr).correlation, mae=float(np.mean(np.abs(p - g))), bias=float(np.mean(p - g)),
                exact=float(np.mean(gi == pi)), adjacent=float(np.mean(np.abs(gi - pi) <= 1)),
                qwk=float(cohen_kappa_score(gi, pi, weights='quadratic', labels=range(4))),
                crit_recall=float(cm[3, 3] / max(cm[3].sum(), 1)), under2=float(np.mean(pi <= gi - 2)),
                topk_recall=len(tg & tp) / k, k=k, cm=cm.tolist())


def group_table(F, s100, stems, src, ndet=None):
    df = F.assign(VMPS=s100, stem=stems, src=src)
    g = section_aggregate(df)
    g['max_VMPS'] = df.groupby('src').VMPS.max()
    g['n_det'] = (df.assign(d=(F.n_boxes > 0) if ndet is None else ndet).groupby('src').d.sum()) if 'n_boxes' in F else g.n_tiles
    return g


def S_ext(g, n95, n_col):
    return 0.35 * g.mean_VMPS + 0.35 * g.p90_VMPS + 0.15 * 100 * g.critical_ratio + 0.15 * 100 * np.minimum(g[n_col] / n95, 1)


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def boot(units_block, fn, reps=1000, seed=1):
    """Block bootstrap: units_block is the block label of every unit; fn(idx) returns a dict of metrics."""
    rng = np.random.default_rng(seed); blocks = pd.unique(units_block)
    pos = {b: np.where(units_block == b)[0] for b in blocks}; out = []
    for _ in range(reps):
        pick = rng.choice(blocks, len(blocks), replace=True); idx = np.concatenate([pos[b] for b in pick]); out.append(fn(idx))
    D = pd.DataFrame(out); return {c: [float(D[c].quantile(.025)), float(D[c].quantile(.975))] for c in D}


# -------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('tag'); ap.add_argument('runs'); ap.add_argument('--split', default='split_stretch')
    ap.add_argument('--device', default='0'); ap.add_argument('--ext', action='store_true'); ap.add_argument('--reps', type=int, default=1000)
    A = ap.parse_args(); runs = A.runs.split(','); out = f'results/revision/{A.tag}'; os.makedirs(out, exist_ok=True)
    GT = pd.read_csv('results/tiles_gt_inventory.csv')
    blocks = pd.read_csv('results/stretch_blocks.csv').set_index('src').block
    R = {}
    for part in ['val', 'test']:
        d = GT[GT[A.split] == part].reset_index(drop=True); R[part] = d
    gtb = {s: read_yolo_txt(f'data/HighRPD/labels/{s}.txt') for s in pd.concat([R['val'], R['test']]).stem}
    P = {r: predict_cache(r, list(R['val'].stem) + list(R['test'].stem), 'data/HighRPD/images', 'highrpd', A.device) for r in runs}
    res = dict(tag=A.tag, runs=runs, split=A.split)

    # ---- reference scores
    ref = {}
    for part in ['val', 'test']:
        d = R[part]; Fg = features({s: [b + (1.0,) for b in gtb[s]] for s in d.stem}, d.stem); gr, g = score(Fg)
        ref[part] = dict(F=Fg, raw=gr, s=g)

    # ---- 2 operating threshold on validation
    curve = []
    for t in GRID:
        q = [];
        for r in runs:
            Fp = features(P[r], R['val'].stem, t); pr, p = score(Fp)
            q.append(agree(ref['val']['raw'], ref['val']['s'], pr, p)['qwk'])
        curve.append(dict(t=t, val_qwk=float(np.mean(q)), val_qwk_sd=float(np.std(q, ddof=1))))
    curve = pd.DataFrame(curve); t_sel = float(curve.loc[curve.val_qwk.idxmax(), 't']); res['t_sel'] = t_sel
    res['val_curve'] = curve.to_dict('records'); print('selected threshold', t_sel, flush=True)

    d = R['test']; stems = list(d.stem); src = d.src.values; ublk = d.src.map(blocks).values
    rg, g = ref['test']['raw'], ref['test']['s']
    Gref = group_table(ref['test']['F'], g, stems, src); Gref['n_det'] = Gref.n_tiles
    gblk = Gref.index.map(blocks).values
    tr = GT[GT[A.split] == 'train']; n95 = float(np.percentile(tr.groupby('src').size(), 95)); res['n95'] = n95
    Gref['S_ext'] = S_ext(Gref, n95, 'n_tiles')

    # ---- test-set curves (tile and group) for every threshold, per seed
    test_curve = []; per = {}
    for t in sorted(set(GRID) | {t_sel, 0.25}):
        rows = []
        for r in runs:
            Fp = features(P[r], stems, t); pr, p = score(Fp); Gp = group_table(Fp, p, stems, src).loc[Gref.index]
            at, ag = agree(rg, g, pr, p), agree(Gref.S_VMPS.values, Gref.S_VMPS.values, Gp.S_VMPS.values, Gp.S_VMPS.values)
            rows.append(dict(tile_qwk=at['qwk'], tile_rho=at['spearman'], tile_crit=at['crit_recall'], tile_mae=at['mae'],
                             grp_qwk=ag['qwk'], grp_rho=ag['spearman'], grp_crit=ag['crit_recall'], grp_mae=ag['mae'], grp_top=ag['topk_recall']))
            per[(t, r)] = (Fp, pr, p, Gp)
        test_curve.append(dict(t=t, **pd.DataFrame(rows).mean().to_dict()))
    res['test_curve'] = test_curve

    # ---- 3 agreement at the selected and default thresholds, bootstrap intervals, baselines
    for t in [t_sel, 0.25]:
        key = f't{t:.2f}'; T_ = []; Gm = []
        for r in runs:
            Fp, pr, p, Gp = per[(t, r)]
            T_.append(agree(rg, g, pr, p)); Gm.append(agree(Gref.S_VMPS.values, Gref.S_VMPS.values, Gp.S_VMPS.values, Gp.S_VMPS.values))
        Fg_t = ref['test']['F']
        for x, r in zip(T_, runs):                         # inventory errors per tile
            Fp = per[(t, r)][0]
            for c in ['N_line', 'N_block', 'N_pit']:
                x['mae_' + c] = float(np.mean(np.abs(Fp[c] - Fg_t[c]))); x['bias_' + c] = float(np.mean(Fp[c] - Fg_t[c]))
            for c in ['L_line', 'A_block', 'A_pit']:
                x['mae_' + c] = float(np.mean(np.abs(Fp[c] - Fg_t[c]))); x['bias_' + c] = float(np.mean(Fp[c] - Fg_t[c]))
            gp, pp = Fg_t.N_pit.values > 0, Fp.N_pit.values > 0; tp = np.sum(gp & pp)
            pr_, rc_ = tp / max(pp.sum(), 1), tp / max(gp.sum(), 1); x['pit_presence_p'] = pr_; x['pit_presence_r'] = rc_
            x['pit_presence_f1'] = 2 * pr_ * rc_ / max(pr_ + rc_, 1e-9); x['share_no_detection'] = float(np.mean(Fp.n_boxes == 0))
        num = lambda L: {k: [float(np.mean([x[k] for x in L])), float(np.std([x[k] for x in L], ddof=1))] for k in L[0] if k not in ('cm', 'n', 'k')}
        res[key] = dict(tile=num(T_), group=num(Gm), tile_cm=np.sum([x['cm'] for x in T_], 0).tolist(), group_cm=np.sum([x['cm'] for x in Gm], 0).tolist(),
                        n_tile=len(g), n_group=len(Gref), k_group=Gm[0]['k'])
        def fn_t(idx):
            L = [agree(rg[idx], g[idx], per[(t, r)][1][idx], per[(t, r)][2][idx]) for r in runs]
            return {k: np.mean([x[k] for x in L]) for k in ['spearman', 'qwk', 'mae', 'exact', 'crit_recall']}
        def fn_g(idx):
            L = [agree(Gref.S_VMPS.values[idx], Gref.S_VMPS.values[idx], per[(t, r)][3].S_VMPS.values[idx], per[(t, r)][3].S_VMPS.values[idx]) for r in runs]
            return {k: np.mean([x[k] for x in L]) for k in ['spearman', 'qwk', 'mae', 'exact', 'adjacent', 'crit_recall']}
        res[key]['tile_ci'] = boot(ublk, fn_t, A.reps); res[key]['group_ci'] = boot(gblk, fn_g, A.reps)
        print(key, 'tile', {k: round(v[0], 3) for k, v in res[key]['tile'].items()}, flush=True)
        print(key, 'group', {k: round(v[0], 3) for k, v in res[key]['group'].items()}, flush=True)
    gi, Gi = bands_idx(g), bands_idx(Gref.S_VMPS.values)
    res['ref_band_share_tile'] = np.bincount(gi, minlength=4).tolist(); res['ref_band_share_group'] = np.bincount(Gi, minlength=4).tolist()
    res['baseline_constant'] = dict(tile_exact=float(np.mean(gi == np.bincount(gi).argmax())), group_exact=float(np.mean(Gi == np.bincount(Gi).argmax())), qwk=0.0)

    # ---- 4 group summaries, size baselines, densities, pseudo-groups (selected threshold)
    t = t_sel; V = []
    for r in runs:
        Fp, pr, p, Gp = per[(t, r)]; Gp = Gp.copy(); Gp['S_ext'] = S_ext(Gp, n95, 'n_det')
        row = {}
        for c in ['mean_VMPS', 'p90_VMPS', 'critical_ratio', 'S_VMPS', 'max_VMPS', 'S_ext']:
            row['rho_' + c] = spearmanr(Gref[c], Gp[c]).correlation
            gb, pb = bands_idx(np.clip(Gref[c] * (100 if c == 'critical_ratio' else 1), 0, 100)), bands_idx(np.clip(Gp[c] * (100 if c == 'critical_ratio' else 1), 0, 100))
            row['qwk_' + c] = cohen_kappa_score(gb, pb, weights='quadratic', labels=range(4))
        row['rho_size_only_vs_S'] = spearmanr(Gref.S_VMPS, Gp.n_det).correlation
        row['rho_size_only_vs_Sext'] = spearmanr(Gref.S_ext, Gp.n_det).correlation
        row['rho_ntiles_vs_S'] = spearmanr(Gref.S_VMPS, Gref.n_tiles).correlation
        for c, lab in [('sum_L_line', 'L'), ('sum_A_block', 'Ab'), ('N_pit', 'Npit')]:
            row['rho_total_' + lab] = spearmanr(Gref[c], Gp[c]).correlation
            row['rho_density_' + lab] = spearmanr(Gref[c] / Gref.n_tiles, Gp[c] / Gp.n_tiles).correlation
            row['rho_size_only_' + lab] = spearmanr(Gref[c], Gref.n_tiles).correlation
            row['partial_total_' + lab] = partial_spearman(Gref[c], Gp[c], Gref.n_tiles)
        # pseudo-groups: test tiles permuted into groups of the observed sizes
        sizes = Gref.n_tiles.values; ps = []
        for k in range(200):
            perm = rng0.permutation(len(stems)); lab = np.empty(len(stems), dtype=object); pos = 0
            for i, n in enumerate(sizes): lab[perm[pos:pos + n]] = f'p{i}'; pos += n
            a = section_aggregate(ref['test']['F'].assign(VMPS=g, stem=stems, src=lab)); b = section_aggregate(Fp.assign(VMPS=p, stem=stems, src=lab)).loc[a.index]
            ps.append(spearmanr(a.S_VMPS, b.S_VMPS).correlation)
        row['rho_pseudo_groups'] = float(np.mean(ps)); row['rho_pseudo_groups_p05_p95'] = [float(np.percentile(ps, 5)), float(np.percentile(ps, 95))]
        V.append(row)
    res['group_variants'] = {k: ([float(np.mean([v[k][0] for v in V])), float(np.mean([v[k][1] for v in V]))] if isinstance(V[0][k], list)
                                 else [float(np.mean([v[k] for v in V])), float(np.std([v[k] for v in V], ddof=1))]) for k in V[0]}
    print('group variants', {k: np.round(v, 3).tolist() for k, v in res['group_variants'].items()}, flush=True)

    # ---- 5 line-box merging
    Fg_m = features({s: [b + (1.0,) for b in gtb[s]] for s in stems}, stems, merge=True); rgm, gm = score(Fg_m)
    M = []
    for t in sorted({0.10, t_sel, 0.25}):
        for merge in [False, True]:
            L = []
            for r in runs:
                Fp = features(P[r], stems, t, merge=merge); pr, p = score(Fp)
                Fr, rr_, rs = (Fg_m, rgm, gm) if merge else (ref['test']['F'], rg, g)
                L.append(dict(mae_N_line=np.mean(np.abs(Fp.N_line - Fr.N_line)), bias_N_line=np.mean(Fp.N_line - Fr.N_line),
                              mae_L_line_m=np.mean(np.abs(Fp.L_line - Fr.L_line)) * TILE_M, bias_L_line_m=np.mean(Fp.L_line - Fr.L_line) * TILE_M,
                              tile_rho=spearmanr(rr_, pr).correlation, tile_qwk=agree(rr_, rs, pr, p)['qwk']))
            M.append(dict(t=t, merge=merge, **pd.DataFrame(L).mean().to_dict()))
    res['line_merge'] = M; res['line_merge_ref_change'] = dict(N_line_before=float(ref['test']['F'].N_line.mean()), N_line_after=float(Fg_m.N_line.mean()),
                                                              L_line_m_before=float(ref['test']['F'].L_line.mean() * TILE_M), L_line_m_after=float(Fg_m.L_line.mean() * TILE_M))
    print('line merge', pd.DataFrame(M).round(3).to_string(), flush=True)

    # ---- 6 error decomposition at the selected threshold
    t = t_sel; dec = {}
    for level in ['tile', 'group']:
        if level == 'tile':
            Rv = g; Dv = np.array([per[(t, r)][2] for r in runs]); empty = np.array([(per[(t, r)][0].n_boxes == 0).values for r in runs])
        else:
            Rv = Gref.S_VMPS.values; Dv = np.array([per[(t, r)][3].S_VMPS.values for r in runs]); empty = None
        E = Dv - Rv; bias = float(E.mean()); obs = float(np.sqrt(np.mean([np.var(e - bias, ddof=1) for e in E])))
        trn = float(np.sqrt(np.mean(Dv.var(0, ddof=1)))); common = float(np.sqrt(max(obs ** 2 - trn ** 2, 0)))
        x = dict(n=len(Rv), bias=bias, rmse_single_seed=float(np.sqrt(np.mean(E ** 2))), sd_single_seed=obs, sd_training=trn,
                 var_share_training=trn ** 2 / obs ** 2, sd_common=common, sd_seed_mean=float(np.std(E.mean(0) - bias, ddof=1)))
        if empty is not None: x['sq_error_share_empty_tiles'] = float(np.sum((E ** 2) * empty) / np.sum(E ** 2)); x['share_empty_tiles'] = float(empty.mean())
        # threshold sensitivity, uniform over 0.15-0.25
        TT = np.round(np.linspace(0.15, 0.25, 11), 3); sd = []
        for r in runs:
            Sc = []
            for tt in TT:
                Fp = features(P[r], stems, tt); _, p = score(Fp)
                Sc.append(p if level == 'tile' else group_table(Fp, p, stems, src).loc[Gref.index].S_VMPS.values)
            sd.append(np.array(Sc).std(0))
        x['sd_threshold_0.15_0.25'] = float(np.sqrt(np.mean(np.array(sd) ** 2)))
        dec[level] = x
    # hypothetical line/block label changes in the reference
    for pswap in [0.05, 0.10]:
        dt, dg = [], []
        for k in range(50):
            sw = {s: [((1 - b[0],) + tuple(b[1:]) if b[0] in (0, 1) and rng0.random() < pswap else tuple(b)) + (1.0,) for b in gtb[s]] for s in stems}
            Fs = features(sw, stems); _, s_ = score(Fs); dt.append(s_); dg.append(group_table(Fs, s_, stems, src).loc[Gref.index].S_VMPS.values)
        dec['tile'][f'sd_label_change_{pswap:.2f}'] = float(np.sqrt(np.mean(np.array(dt).std(0, ddof=1) ** 2)))
        dec['group'][f'sd_label_change_{pswap:.2f}'] = float(np.sqrt(np.mean(np.array(dg).std(0, ddof=1) ** 2)))
    # proxy approximation: lognormal factors from the SAM comparison, centred (scale offsets absorbed by the reference scales)
    I = pd.read_csv('results/sam_instances.csv'); rel = {}
    for c, key in [(0, 'L_line'), (1, 'A_block'), (2, 'A_pit')]:
        s = I[(I.cls == c) & (I.mask_area > 0)]; ratio = (s.skel_len / s.bbox_diag) if c == 0 else (s.mask_area / s.bbox_area)
        rel[key] = float(np.std(np.log(ratio), ddof=1))
    dec['proxy_rel_dispersion'] = rel
    gid = pd.factorize(src)[0]
    for rho in [0.0, 0.5, 1.0]:
        dt, dg = [], []
        for k in range(200):
            Fm = ref['test']['F'].copy()
            for key, sg in rel.items():
                zg = rng0.standard_normal(gid.max() + 1)[gid]; zt = rng0.standard_normal(len(stems))
                Fm[key] = Fm[key] * np.exp(sg * (np.sqrt(rho) * zg + np.sqrt(1 - rho) * zt))
            _, s_ = score(Fm); dt.append(s_); dg.append(group_table(Fm, s_, stems, src).loc[Gref.index].S_VMPS.values)
        dec['tile']['sd_proxy'] = float(np.sqrt(np.mean(np.array(dt).std(0, ddof=1) ** 2)))
        dec['group'][f'sd_proxy_rho{rho:.1f}'] = float(np.sqrt(np.mean(np.array(dg).std(0, ddof=1) ** 2)))
    res['decomposition'] = dec; print('decomposition', json.dumps(dec, indent=0, default=float)[:1500], flush=True)

    # ---- 7 external data
    if A.ext:
        from ultralytics import YOLO
        ext = {}
        for name, idxf, imdir, lbdir, yml in [('native', 'results/uavpdd_tiles_index.csv', 'data/uavpdd_tiles/images', 'data/uavpdd_tiles/labels', 'data/uavpdd_tiles/uavpdd_tiles_all.yaml'),
                                              ('gsd_matched', 'results/uavpdd_tiles_gsd_index.csv', 'data/uavpdd_tiles_gsd/images', 'data/uavpdd_tiles_gsd/labels', 'data/uavpdd_tiles_gsd/uavpdd_tiles_gsd_all.yaml')]:
            U = pd.read_csv(idxf); us = list(U.stem)
            UG = features({s: [b + (1.0,) for b in read_yolo_txt(f'{lbdir}/{s}.txt')] for s in us}, us); ugr, ug = score(UG)
            UGg = group_table(UG, ug, us, U.view.values)
            L = []
            for r in runs:
                v = YOLO(f'runs/{r}/weights/best.pt').val(data=yml, split='test', imgsz=640, batch=8, plots=False, verbose=False, project='runs/_val', name=f'{r}_{name}', exist_ok=True, half=(A.device != 'cpu'), device=A.device)
                Pp = predict_cache(r, us, imdir, f'uavpdd_{name}', A.device)
                Fp = features(Pp, us, t_sel); pr, p = score(Fp); Gp = group_table(Fp, p, us, U.view.values).loc[UGg.index]
                at, ag = agree(ugr, ug, pr, p), agree(UGg.S_VMPS.values, UGg.S_VMPS.values, Gp.S_VMPS.values, Gp.S_VMPS.values)
                L.append(dict(map50=float(v.box.map50), map5095=float(v.box.map), ap50_line=float(v.box.ap50[0]), ap50_block=float(v.box.ap50[1]), ap50_pit=float(v.box.ap50[2]),
                              tile_rho=at['spearman'], tile_qwk=at['qwk'], tile_mae=at['mae'], group_rho=ag['spearman'], group_qwk=ag['qwk'], group_mae=ag['mae']))
            ext[name] = dict(n_tiles=len(us), n_views=int(U.view.nunique()), **{k: [float(np.mean([x[k] for x in L])), float(np.std([x[k] for x in L], ddof=1))] for k in L[0]})
            print('external', name, {k: np.round(v, 3).tolist() if isinstance(v, list) else v for k, v in ext[name].items()}, flush=True)
        res['external'] = ext
    json.dump(res, open(f'{out}/revision.json', 'w'), indent=1, default=float); print('written', out)


if __name__ == '__main__':
    main()
