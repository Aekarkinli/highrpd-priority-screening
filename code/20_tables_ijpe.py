"""LaTeX tables of the revised manuscript (paper_ijpe/tables). Values are means over seeds with the standard deviation
in parentheses in units of the last digit; bracketed ranges are 95 % block-bootstrap intervals."""
import json, os, glob, numpy as np, pandas as pd
OUT = 'paper_ijpe/tables'; os.makedirs(OUT, exist_ok=True)
DET = {'YOLOv8s': 'y8s', 'YOLO11s': 'y11s'}
SEEDS = [42, 123, 3407]


def ms(v, f='%.3f'):
    v = np.asarray([x for x in v if x is not None and not np.isnan(x)], float)
    if len(v) == 0: return '--'
    if len(v) == 1: return f % v[0]
    dec = int(f[2]); return (f % v.mean()) + '(%d)' % int(round(v.std(ddof=1) * 10 ** dec))


def pair(x, f='%.3f'):
    """[mean, sd] from revision.json."""
    m, s = x; dec = int(f[2])
    return (f % m) + ('' if s is None or np.isnan(s) else '(%d)' % int(round(s * 10 ** dec)))


def ci(x, f='%.2f'): return '[' + (f % x[0]) + ', ' + (f % x[1]) + ']'


def table(name, caption, colspec, header, rows, note=None, size='\\small', sep=None):
    body = '\n'.join(r if r.startswith(('\\addlinespace', '\\midrule', '\\cmidrule')) else r + ' \\\\' for r in rows)
    t = ('\\begin{table}[t]\n\\centering\n' + size + '\n' + (f'\\setlength{{\\tabcolsep}}{{{sep}}}\n' if sep else '') + '\\caption{' + caption + '}\\label{tab:' + name + '}\n'
         '\\begin{tabular}{' + colspec + '}\n\\toprule\n' + header + ' \\\\\n\\midrule\n' + body + '\n\\bottomrule\n\\end{tabular}\n')
    if note: t += '\\par\\smallskip\\parbox{\\linewidth}{\\footnotesize ' + note + '}\n'
    t += '\\end{table}\n'
    open(f'{OUT}/{name}.tex', 'w', encoding='utf8').write(t)


def summ(name):
    p = f'results/runs/{name}/summary.json'
    return json.load(open(p)) if os.path.exists(p) else None


def rev(tag):
    p = f'results/revision/{tag}/revision.json'
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------- partitions
AUD = json.load(open('results/revision/val_overlap_check.json'))
T = pd.read_csv('results/table_splits.csv'); T['partition'] = T.partition.replace({'Source-separated': 'Frame-separated'})
rows = []; last = None
for r in T.itertuples():
    name = r.partition if r.partition != last else ''; last = r.partition
    sub = {'train': 'Training', 'val': 'Validation', 'test': 'Test', 'drop': 'Buffer (removed)'}[r.subset]
    ov = '--' if np.isnan(r.groups_overlapping_train) else '%.1f' % (100 * r.groups_overlapping_train)
    if r.partition == 'Stretch-separated' and r.subset == 'val': ov = '%.1f' % (100 * len(AUD['val_frames']) / r.frames)   # exhaustive audit
    rows.append(f'{name} & {sub} & {r.tiles:,} & {r.frames} & {r.line:,} & {r.block:,} & {r.pit:,} & {ov}'.replace(',', '{,}'))
    if r.subset in ('test', 'drop') and r.partition != 'Stretch-separated' or r.subset == 'drop': rows.append('\\addlinespace')
table('splits', 'Partitions of HighRPD. The last column gives the share of validation or test frames that overlap a training frame, from the exhaustive audit for the stretch-separated partition.',
      'llrrrrrr', 'Partition & Subset & Tiles & Frames & Line & Block & Pit & Overlap (\\%)', [x for x in rows if x][:-1] if rows[-1] == '\\addlinespace' else rows)

# ---------------------------------------------------------------- detection by partition
rows = []
for det, p in DET.items():
    for part, key in [('Random', 'random'), ('Frame-separated', 'source'), ('Stretch-separated', 'stretch')]:
        S = [summ(f'{p}_{key}_s{s}') for s in SEEDS]; S = [s for s in S if s]
        if not S: continue
        rows.append(f'{det} & {part} & {len(S)} & ' + ' & '.join([ms([s["det"]["map50"] for s in S]), ms([s["det"]["map5095"] for s in S])] +
                                                          [ms([s["det"]["ap50_cls"][c] for s in S]) for c in range(3)]))
    rows.append('\\addlinespace')
for label, names in [('YOLO11s, pit oversampling', [f'y11s_stretchpit_s{s}' for s in SEEDS]), ('YOLO11s, 150 epochs', ['y11s_stretch_e150_s42'])]:
    S = [summ(n) for n in names]; S = [s for s in S if s]
    if S: rows.append(f'\\multicolumn{{2}}{{l}}{{{label}}} & {len(S)} & ' + ' & '.join([ms([s["det"]["map50"] for s in S]), ms([s["det"]["map5095"] for s in S])] + [ms([s["det"]["ap50_cls"][c] for s in S]) for c in range(3)]))
if rows and rows[-1] == '\\addlinespace': rows.pop()
table('detection', 'Detection accuracy on the test set of each partition.', 'llrrrrrr',
      'Detector & Partition & Seeds & AP$_{50}$ & AP$_{50\\text{--}95}$ & AP$_{50}$ line & AP$_{50}$ block & AP$_{50}$ pit', rows,
      'Pit oversampling and the 150-epoch schedule refer to the stretch-separated partition.')

# ---------------------------------------------------------------- proxy check
P = pd.read_csv('results/sam_proxy_summary.csv').set_index('cls')
rows = []
for c, lab in [('line', 'Line crack'), ('block', 'Block crack'), ('pit', 'Pit')]:
    r = P.loc[c]; tl = P.loc[c + '_tile']
    sk_a = '%.2f' % r.rho_area_skel if c == 'line' else '--'; sk_d = '%.2f' % r.rho_diag_skel if c == 'line' else '--'
    rot = ('%.2f' % r.rot_area_maxmin + ' / ' + '%.2f' % r.rot_diag_maxmin) if c == 'line' else '--'
    rows.append(f'{lab} & {int(r.n)} & {r.rho_area:.2f} & {r.rho_diag:.2f} & {sk_a} & {sk_d} & {rot}')
table('proxy', 'Rank correlation between box proxies and box-prompted pixel-level pseudo-references, and orientation sensitivity of line-crack boxes.',
      'lrrrrrr', 'Class & Instances & \\multicolumn{2}{c}{$\\rho$ with mask area} & \\multicolumn{2}{c}{$\\rho$ with skeleton length} & Max/min under rotation \\\\\n & & box area & box diagonal & box area & box diagonal & area / diagonal', rows,
      'Max/min is the median ratio of the largest to the smallest value when the mask is rotated from 0 to 90 degrees.')

# ---------------------------------------------------------------- agreement at tile and frame level
R = {d: rev(f'{p}_stretch') for d, p in DET.items()}
if all(R.values()):
    rows = []
    M = [('mae', 'Mean absolute error', '%.1f', True), ('bias', 'Bias', '%.1f', False), ('spearman', 'Spearman $\\rho$', '%.2f', True),
         ('qwk', 'Weighted kappa', '%.2f', True), ('exact', 'Exact band', '%.2f', True), ('adjacent', 'Within one band', '%.3f', True),
         ('crit_recall', 'Critical recall', '%.2f', False), ('under2', 'Under by two or more bands', '%.2f', False), ('topk_recall', 'Top 10\\% recall', '%.2f', False)]
    for k, l, f, has_ci in M:
        cells = []
        for d, r in R.items():
            q = r[f"t{r['t_sel']:.2f}"]
            for lev, cik in [('tile', 'tile_ci'), ('group', 'group_ci')]:
                if k == 'topk_recall' and lev == 'tile': cells.append('--'); continue
                if k == 'crit_recall' and lev == 'group': cells.append('--'); continue
                cells.append(pair(q[lev][k], f))
        rows.append(l + ' & ' + ' & '.join(cells))
        if has_ci:
            cells = []
            for d, r in R.items():
                q = r[f"t{r['t_sel']:.2f}"]
                for lev, cik in [('tile', 'tile_ci'), ('group', 'group_ci')]:
                    cells.append(ci(q[cik][k], f) if k in q[cik] else '')
            rows.append('\\quad \\textit{95\\% interval} & ' + ' & '.join(cells))
    RB = json.load(open('results/revision/score_robustness.json'))
    r0 = R['YOLO11s']; b = r0['baseline_constant']; q0 = r0[f"t{r0['t_sel']:.2f}"]
    table('agreement', f"Agreement between detection-based and annotation-based priority scores on the stretch-separated test set at the selected threshold of {r0['t_sel']:.2f}.", 'lrrrr',
          'Measure & \\multicolumn{2}{c}{YOLOv8s} & \\multicolumn{2}{c}{YOLO11s} \\\\\n & Tile & Frame & Tile & Frame', rows,
          (f"Test set: {q0['n_tile']:,} tiles and {q0['n_group']} frames. Mean absolute error and bias are in score units (0--100). A constant prediction of the most frequent band agrees exactly on {b['tile_exact']:.2f} of the tiles and {b['group_exact']:.2f} of the frames, and within one band on {RB['constant_baseline']['tile_within1']:.2f} and {RB['constant_baseline']['frame_within1']:.3f}. "
           f"Critical recall is not reported for frames because only {r0['ref_band_share_group'][3]} test frames are Critical in the reference, and top 10\\% recall refers to the {q0['k_group']} frames with the highest reference scores. Intervals are block-bootstrap intervals of the seed mean.").replace(',', '{,}').replace('{,} ', ', '), size='\\footnotesize')

    # ---------------------------------------------------------------- inventory errors
    rows = []
    for d, r in R.items():
        for t in [r['t_sel'], 0.25]:
            a = r[f't{t:.2f}']['tile']; lab = f'{d}, {t:.2f}'
            rows.append(lab + ' & ' + ' & '.join([pair(a['share_no_detection'], '%.2f'), pair(a['mae_N_line'], '%.2f'), pair(a['bias_N_line'], '%.2f'), pair(a['mae_N_block'], '%.2f'),
                        pair(a['bias_N_block'], '%.2f'), pair(a['mae_N_pit'], '%.2f'), pair(a['bias_N_pit'], '%.2f'), pair(a['pit_presence_r'], '%.2f'), pair(a['pit_presence_f1'], '%.2f')]))
    rp = rev('y11s_stretchpit')
    if rp:
        for t in [rp['t_sel']]:
            a = rp[f't{t:.2f}']['tile']
            rows.append(f'YOLO11s, pits $\\times$3, {t:.2f} & ' + ' & '.join([pair(a['share_no_detection'], '%.2f'), pair(a['mae_N_line'], '%.2f'), pair(a['bias_N_line'], '%.2f'), pair(a['mae_N_block'], '%.2f'),
                        pair(a['bias_N_block'], '%.2f'), pair(a['mae_N_pit'], '%.2f'), pair(a['bias_N_pit'], '%.2f'), pair(a['pit_presence_r'], '%.2f'), pair(a['pit_presence_f1'], '%.2f')]))
    table('inventory', 'Inventory errors per test tile at the selected threshold and at 0.25.', 'lrrrrrrrrr',
          'Configuration & No detection & \\multicolumn{2}{c}{Line count} & \\multicolumn{2}{c}{Block count} & \\multicolumn{2}{c}{Pit count} & \\multicolumn{2}{c}{Pit presence} \\\\\n'
          ' & share & MAE & bias & MAE & bias & MAE & bias & recall & F1', rows, 'Counts are boxes per tile; the threshold follows the detector name.', size='\\footnotesize', sep='4pt')

    # ---------------------------------------------------------------- frame summaries and baselines
    rows = []
    lab = {'mean_VMPS': 'Mean tile score', 'p90_VMPS': '90th percentile', 'critical_ratio': 'Share of Critical tiles', 'max_VMPS': 'Maximum tile score',
           'S_VMPS': 'Combined score $S$', 'S_ext': 'Extent-aware score $S_{\\mathrm{ext}}$'}
    for k, l in lab.items():
        rows.append(l + ' & ' + ' & '.join(pair(R[d]['group_variants']['rho_' + k], '%.2f') + ' & ' + pair(R[d]['group_variants']['qwk_' + k], '%.2f') for d in R))
    rows.append('\\addlinespace')
    rows.append('Annotated tiles per frame, against $S$ & ' + ' & '.join('%.2f & ' % R[d]['group_variants']['rho_ntiles_vs_S'][0] for d in R))
    rows.append('Tiles with detections per frame, against $S$ & ' + ' & '.join(pair(R[d]['group_variants']['rho_size_only_vs_S'], '%.2f') + ' & ' for d in R))
    rows.append('Tiles with detections per frame, against $S_{\\mathrm{ext}}$ & ' + ' & '.join(pair(R[d]['group_variants']['rho_size_only_vs_Sext'], '%.2f') + ' & ' for d in R))
    rows.append('Pseudo-frames of equal size, $S$ & ' + ' & '.join(pair(R[d]['group_variants']['rho_pseudo_groups'], '%.2f') + ' & ' for d in R))
    rows.append('\\addlinespace')
    for c, l in [('L', 'line length'), ('Ab', 'block area'), ('Npit', 'pit count')]:
        rows.append(f'Summed {l}: total / density / partial & ' + ' & '.join(
            '%.2f / %.2f / %.2f & ' % (R[d]['group_variants']['rho_total_' + c][0], R[d]['group_variants']['rho_density_' + c][0], R[d]['group_variants']['partial_total_' + c][0]) for d in R))
    rows.append('Summed quantities from frame size only (line / block / pit) & ' + ' & '.join(
        '%.2f / %.2f / %.2f & ' % tuple(R[d]['group_variants']['rho_size_only_' + c][0] for c in ['L', 'Ab', 'Npit']) for d in R))
    table('frames', 'Frame-level agreement of alternative summaries and baselines at the selected threshold (Spearman $\\rho$ and QWK of bands).', 'lrrrr',
          'Summary & \\multicolumn{2}{c}{YOLOv8s} & \\multicolumn{2}{c}{YOLO11s} \\\\\n & $\\rho$ & QWK & $\\rho$ & QWK', rows,
          'The annotated-tile count is taken from the reference and the count of tiles with detections from the detector, so both size predictors carry information that a survey without annotations would not have in full. Density divides each total by the number of tiles of the frame, and partial is the rank correlation of the totals controlling for that number.', size='\\footnotesize')

    # ---------------------------------------------------------------- score design
    RS = pd.read_csv('results/revision/score_robustness.csv'); rows = []
    for r in RS.itertuples():
        if r.variant in ('Scaling at P90', 'Frame: mean only', 'Bands 20/45/70'): rows.append('\\addlinespace')
        rows.append(f'{r.variant} & {r.ref_rho_frame_vs_base:.2f} & {r.ref_top10_vs_base:.2f} & {r.YOLO11s_tile_rho:.2f} & {r.YOLO11s_tile_qwk:.2f} & '
                    f'{r.YOLO11s_frame_rho:.2f} & {r.YOLO11s_frame_qwk:.2f} & {r.YOLOv8s_frame_rho:.2f} & {int(r.YOLO11s_frame_n_crit)}')
    table('design', 'Sensitivity of the agreement to the design of the priority score at the selected threshold. Every variant is applied to annotations and detections alike.', 'lrrrrrrrr',
          'Variant & \\multicolumn{2}{c}{Reference vs base} & \\multicolumn{4}{c}{YOLO11s} & YOLOv8s & Critical \\\\\n'
          ' & $\\rho$ & top 10\\% & tile $\\rho$ & tile QWK & frame $\\rho$ & frame QWK & frame $\\rho$ & frames', rows,
          'Reference vs base compares the annotation-based frame ranking of the variant with that of the base design. Detector columns are means over three seeds. '
          'Scaling refits the component and score percentiles on the training tiles. Critical frames is the number of test frames in the Critical band of the reference.', size='\\footnotesize', sep='4pt')

    # ---------------------------------------------------------------- error decomposition
    rows = []
    XD = json.load(open('results/revision/decomposition_exact.json'))
    items = [('bias', 'Bias $b$', '%.1f'), ('rmse_single_run', 'Pooled RMSE of individual runs', '%.1f'), ('B', 'Dispersion of the three-run mean error $B$', '%.1f'),
             ('T', 'Variation between runs $T$', '%.1f'), ('share_T', 'Share of $T^2$ in $B^2 + T^2$', '%.2f'),
             ('rmse_seed_mean', 'RMSE of the three-run mean score', '%.1f'),
             ('sq_error_share_empty_tiles', 'Share of squared error on tiles without detections', '%.2f'),
             ('sd_threshold_0.15_0.25', 'Sensitivity: threshold 0.15 to 0.25', '%.1f'), ('sd_label_change_0.05', 'Sensitivity: 5\\% line/block label changes', '%.1f'),
             ('sd_label_change_0.10', 'Sensitivity: 10\\% line/block label changes', '%.1f'), ('sd_proxy', 'Sensitivity: proxy approximation', '%.1f')]
    for k, l, f in items:
        cells = []
        for d in R:
            D = R[d]['decomposition']; X = XD[f'{DET[d]}_stretch']
            for lev in ['tile', 'group']:
                if k == 'sd_proxy' and lev == 'group':
                    cells.append('%.1f--%.1f' % (D['group']['sd_proxy_rho0.0'], D['group']['sd_proxy_rho1.0']))
                else:
                    v = X['tile' if lev == 'tile' else 'frame'].get(k, D[lev].get(k)); cells.append('' if v is None else f % v)
        rows.append(l + ' & ' + ' & '.join(cells))
        if k in ('sq_error_share_empty_tiles',): rows.append('\\addlinespace')
    table('decomposition', 'Decomposition of the error of the detection-based score and sensitivities, in score units (0--100) at the selected threshold.', 'lrrrr',
          'Component & \\multicolumn{2}{c}{YOLOv8s} & \\multicolumn{2}{c}{YOLO11s} \\\\\n & Tile & Frame & Tile & Frame', rows,
          'The mean squared error pooled over the three runs equals $b^2 + B^2 + T^2$ (Section~\\ref{sec:decomp}). The proxy sensitivity at frame level is given for within-frame correlations of 0 and 1.', size='\\footnotesize')

    # ---------------------------------------------------------------- external data
    rows = []
    for d, r in R.items():
        if 'external' not in r: continue
        for name, l in [('native', 'native, 3 mm'), ('gsd_matched', '6.3 mm')]:
            e = r['external'][name]
            rows.append(f'{d} & {l} & {e["n_tiles"]:,} & '.replace(',', '{,}') + ' & '.join([pair(e['map50'], '%.3f'), pair(e['ap50_line'], '%.3f'), pair(e['ap50_block'], '%.3f'), pair(e['ap50_pit'], '%.3f'),
                        pair(e['tile_rho'], '%.2f'), pair(e['tile_qwk'], '%.2f'), pair(e['group_rho'], '%.2f'), pair(e['group_qwk'], '%.2f')]))
    FT = {('Tuned, random views', 'HighRPD start'): 'ft_uav_stretch', ('Tuned, random views', 'COCO start'): 'ft_uav_coco',
          ('Tuned, sequence', 'HighRPD start'): 'ft_uav_seq_stretch', ('Tuned, sequence', 'COCO start'): 'ft_uav_seq_coco'}
    ms = lambda v: [float(np.mean(v)), float(np.std(v, ddof=1))]
    for (lp, l), n in FT.items():
        if not os.path.exists(f'results/runs/{n}_s3407/summary_ft.json'): continue
        S = [json.load(open(f'results/runs/{n}_s{sd}/summary_ft.json')) for sd in (42, 123, 3407)]
        vals = [ms([s['det']['map50'] for s in S])] + [ms([s['det']['ap50_cls'][c] for s in S]) for c in range(3)]
        agr = [ms([s[k][m] for s in S]) for k, m in [('tile', 'spearman'), ('tile', 'qwk'), ('section', 'spearman'), ('section', 'qwk')]]
        rows.append(f'{lp} & {l} & ' + f'{S[0]["tile"]["n"]:,} & '.replace(',', '{,}') + ' & '.join([pair(v, '%.3f') for v in vals] + [pair(v, '%.2f') for v in agr]))
    table('external', 'Zero-shot application to UAV-PDD2023 and diagnostic fine-tuning.', 'llrrrrrrrrr',
          'Model & GSD or start & Tiles & AP$_{50}$ & line & block & pit & \\multicolumn{2}{c}{Tile} & \\multicolumn{2}{c}{View} \\\\\n & & & & & & & $\\rho$ & QWK & $\\rho$ & QWK', rows,
          'Zero-shot rows give YOLOv8s and YOLO11s, tuned rows YOLO11s. All rows are means over three seeds with the standard deviation in parentheses. Fine-tuning used the native tiles of a partition by random views (36 test views) or by sequence blocks with a buffer (32 test views), starting from the stretch-separated HighRPD weights or from the COCO weights.', size='\\footnotesize', sep='4pt')
print('tables written:', sorted(os.listdir(OUT)))
