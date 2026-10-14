"""Build LaTeX tables for the manuscript from results/ (writes paper/tables/*.tex)."""
import json, glob, os, re, numpy as np, pandas as pd
os.makedirs('paper/tables', exist_ok=True)
S = {os.path.basename(os.path.dirname(p)): json.load(open(p)) for p in glob.glob('results/runs/*/summary.json')}


CAP = {
 'splits': r'Composition of the random and source-separated partitions of HighRPD; the last column is the share of tiles whose source group also contributes tiles to the training subset.',
 'classes': r'Annotation statistics of HighRPD by class: box sides in pixels (\SI{6.3}{mm} per pixel), box area as a fraction of the tile, and share of boxes spanning the full tile width or height.',
 'detection': r'Detection performance on the test subset of each partition; recall at the F1-optimal threshold.',
 'tile_error': r'Tile-level inventory error on the test subsets: count MAE and bias (detections minus reference) per class, and recognition of tiles containing pits.',
 'tile_score': r'Agreement between annotation-based and detection-based tile scores: score error, rank correlations, screening-band agreement, Critical recall and top-5\% overlap.',
 'section_error': r'Agreement of section-level results for the 87 test groups: section score $S$, summed inventory quantities, pit count, bands of $S$ and top-10\% overlap.',
 'proxy': r'Rank correlation (Spearman) between box proxies and SAM pseudo-references on 150 tiles, and orientation sensitivity of the line proxies (median ratio of largest to smallest value under rotation).',
 'sensitivity': r'Sensitivity of the annotation-based score to the component weights (line / block / pit area / pit count / multi-class) and to the reference percentiles, relative to the base configuration.',
 'conf': r'Effect of the deployment confidence threshold on the tile-level inventory and score agreement (YOLO11s, source-separated test subset).',
 'summary': r'Measurement-to-decision summary of the source-separated configurations.',
 'external': r'Zero-shot application of the HighRPD-trained detectors to the 9{,}756 UAV-PDD2023 tiles and to the 236 source views.',
}


def w(name, txt):
    if name in CAP: txt = re.sub(r'\\caption\{.*?\}\\label', lambda m: '\\caption{' + CAP[name] + '}\\label', txt, count=1, flags=re.S)
    txt = txt.replace('Frames shared with training', 'Groups shared with training').replace('& Frames &', '& Groups &')
    width = '\\textwidth' if 'table*' in txt else '\\columnwidth'
    txt = re.sub(r'\\begin\{tabular\*\}\{\\tblwidth\}\{@\{\}(.*?)@\{\}\}', lambda m: '\\resizebox{' + width + '}{!}{\\begin{tabular}{' + m.group(1) + '}', txt)
    txt = txt.replace('\\end{tabular*}', '\\end{tabular}}')
    open(f'paper/tables/{name}.tex', 'w', encoding='utf8').write(txt)


def ms(vals, f='%.3f', pm=True):
    v = np.asarray(vals, float)
    if len(v) == 1 or not pm: return f % v.mean()
    dec = int(f[2]) if f.startswith('%.') else 3
    sd = int(round(v.std(ddof=1) * 10 ** dec))
    return (f % v.mean()) + '(%d)' % sd


# ---- Table: partitions
T = pd.read_csv('results/table_splits.csv')
rows = []
for r in T.itertuples():
    sh = '--' if r.subset == 'train' else '%.1f' % (100 * r.frames_shared_with_train)
    rows.append(f"{r.partition if r.subset=='train' else ''} & {r.subset} & {r.tiles:,} & {r.frames} & {r.line:,} & {r.block:,} & {r.pit:,} & {r.pit_tiles} & {sh} \\\\")
w('splits', """\\begin{table*}[t]
\\caption{Composition of the random and source-separated partitions of HighRPD. The last column gives the percentage of tiles in the subset whose source frame also contributes tiles to the training subset of the same partition.}\\label{tab:splits}
\\begin{tabular*}{\\tblwidth}{@{}llrrrrrrr@{}}
\\toprule
Partition & Subset & Tiles & Frames & Line & Block & Pit & Tiles with pits & Frames shared with training (\\%) \\\\
\\midrule
""" + '\n'.join(rows) + """
\\bottomrule
\\end{tabular*}
\\end{table*}
""")

# ---- Table: class statistics
C = pd.read_csv('results/table_class_stats.csv', index_col=0)
rows = [f"{c} & {int(C.loc[c,'n']):,} & {100*C.loc[c,'share_instances']:.1f} & {int(C.loc[c,'tiles_with']):,} & {C.loc[c,'long_med']*640:.0f} & {C.loc[c,'short_med']*640:.0f} & {100*C.loc[c,'area_med']:.1f} & {100*C.loc[c,'full_extent']:.1f} \\\\" for c in ['line', 'block', 'pit']]
w('classes', """\\begin{table}[t]
\\caption{Annotation statistics of HighRPD by distress class. Box sides are medians in pixels (\\SI{6.3}{mm} per pixel); the box area is the median fraction of the tile; the last column is the share of boxes that span the full tile width or height.}\\label{tab:classes}
\\begin{tabular*}{\\tblwidth}{@{}lrrrrrrr@{}}
\\toprule
Class & Boxes & Share (\\%) & Tiles & Long side & Short side & Area (\\%) & Full extent (\\%) \\\\
\\midrule
""" + '\n'.join(rows) + """
\\bottomrule
\\end{tabular*}
\\end{table}
""")

# ---- helper to group runs
def group(prefix): return [S[k] for k in sorted(S) if k.startswith(prefix)]
CFG = [('y8s_random', 'YOLOv8s', 'Random'), ('y11s_random', 'YOLO11s', 'Random'), ('y8s_source', 'YOLOv8s', 'Source'),
       ('y11s_source', 'YOLO11s', 'Source'), ('y11s_pit', 'YOLO11s + pit', 'Source')]
# ---- Table: detection
rows = []
for pre, model, part in CFG:
    g = group(pre)
    if not g: continue
    d = [x['det'] for x in g]; n = len(g)
    rows.append(f"{model} & {part} & {n} & {ms([x['map50'] for x in d])} & {ms([x['map5095'] for x in d])} & " +
                ' & '.join(ms([x['ap50_cls'][c] for x in d]) for c in range(3)) + ' & ' + ' & '.join(ms([x['r_cls'][c] for x in d]) for c in range(3)) + ' \\\\')
w('detection', """\\begin{table*}[t]
\\caption{Detection performance on the test subset of each partition. Source-separated runs are means over three seeds with the standard deviation in parentheses in units of the last digit; random-partition runs use one seed. Recall is reported at the confidence threshold maximising F1.}\\label{tab:detection}
\\begin{tabular*}{\\tblwidth}{@{}llcccccccccc@{}}
\\toprule
Model & Partition & Seeds & mAP$_{50}$ & mAP$_{50\\text{--}95}$ & AP$_{50}$ line & AP$_{50}$ block & AP$_{50}$ pit & Recall line & Recall block & Recall pit \\\\
\\midrule
""" + '\n'.join(rows) + """
\\bottomrule
\\end{tabular*}
\\end{table*}
""")

# ---- Table: tile-level inventory error
rows = []
for pre, model, part in CFG:
    g = group(pre)
    if not g: continue
    t = [x['tile'] for x in g]
    rows.append(f"{model} & {part} & {ms([x['mae_N_line'] for x in t],'%.2f')} & {ms([x['mae_N_block'] for x in t],'%.2f')} & {ms([x['mae_N_pit'] for x in t],'%.2f')} & {ms([x['bias_N_line'] for x in t],'%.2f')} & {ms([x['bias_N_block'] for x in t],'%.2f')} & {ms([x['bias_N_pit'] for x in t],'%.2f')} & {ms([x['pit_presence_p'] for x in t])} & {ms([x['pit_presence_r'] for x in t])} & {ms([x['pit_presence_f1'] for x in t])} \\\\")
w('tile_error', r"""\begin{table*}[t]
\caption{Tile-level inventory error on the test subsets (mean over seeds, standard deviation in parentheses in units of the last digit). MAE and bias refer to the number of boxes per tile (detections minus reference); pit presence refers to the recognition of tiles that contain at least one pit.}\label{tab:tile}
\begin{tabular*}{\tblwidth}{@{}llccccccccc@{}}
\toprule
 & & \multicolumn{3}{c}{Count MAE} & \multicolumn{3}{c}{Count bias} & \multicolumn{3}{c}{Pit presence} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
Model & Partition & line & block & pit & line & block & pit & precision & recall & F1 \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
""")

# ---- Table: tile-level score and band agreement
rows = []
for pre, model, part in CFG:
    g = group(pre)
    if not g: continue
    t = [x['tile'] for x in g]
    rows.append(f"{model} & {part} & {ms([x['mae'] for x in t],'%.1f')} & {ms([x['bias'] for x in t],'%.1f')} & {ms([x['spearman'] for x in t])} & {ms([x['kendall'] for x in t])} & {ms([x['qwk'] for x in t])} & {ms([x['band_acc'] for x in t])} & {ms([x['adjacent_acc'] for x in t])} & {ms([x['crit_recall'] for x in t])} & {ms([x['highcrit_recall'] for x in t])} & {ms([x['topk_overlap'] for x in t])} \\\\")
w('tile_score', r"""\begin{table*}[t]
\caption{Agreement between reference and detection-based tile scores on the test subsets (mean over seeds, standard deviation in parentheses in units of the last digit). Score MAE and bias are in score units (0--100); $\rho$ and $\tau$ are Spearman and Kendall rank correlations; $\kappa_w$ is the quadratic weighted kappa of the four screening bands; exact and $\pm 1$ are the band accuracies; recall refers to the Critical band and to the High and Critical bands together; top 5\% is the overlap of the highest-ranked 5\% of tiles.}\label{tab:tilescore}
\begin{tabular*}{\tblwidth}{@{}llcccccccccc@{}}
\toprule
 & & \multicolumn{2}{c}{Score error} & \multicolumn{2}{c}{Rank} & \multicolumn{5}{c}{Screening bands} & Top 5\% \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-11}
Model & Partition & MAE & bias & $\rho$ & $\tau$ & $\kappa_w$ & exact & $\pm 1$ & Critical recall & High+Critical recall & overlap \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
""")

# ---- Table: section level
rows = []
for pre, model, part in CFG:
    g = group(pre)
    if not g or part == 'Random': continue
    s = [x['section'] for x in g]
    rows.append(f"{model} & {ms([x['mae'] for x in s],'%.1f')} & {ms([x['spearman'] for x in s])} & {ms([x['kendall'] for x in s])} & {ms([x['spearman_sumL'] for x in s])} & {ms([x['spearman_sumAb'] for x in s])} & {ms([x['spearman_Npit'] for x in s])} & {ms([x['mae_Npit'] for x in s],'%.2f')} & {ms([x['qwk'] for x in s])} & {ms([x['band_acc'] for x in s])} & {ms([x['topk_overlap'] for x in s])} \\\\")
w('section_error', """\\begin{table*}[t]
\\caption{Agreement between reference and detection-based section-level results for the 87 test frames of the source-separated partition (mean over three seeds, standard deviation in parentheses in units of the last digit). Rank correlations are given for the section score $S$ and for the summed line chord length, block area and pit count of the section; $\\kappa_w$ and exact refer to the screening bands applied to $S$; top 10\\% is the overlap of the nine highest-ranked sections.}\\label{tab:section}
\\begin{tabular*}{\\tblwidth}{@{}lcccccccccc@{}}
\\toprule
 & \\multicolumn{3}{c}{Section score $S$} & \\multicolumn{3}{c}{Inventory ($\\rho$)} & Pit count & \\multicolumn{2}{c}{Bands} & Top 10\\% \\\\
\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){9-10}
Model & MAE & $\\rho$ & $\\tau$ & line & block & pit & MAE & $\\kappa_w$ & exact & overlap \\\\
\\midrule
""" + '\n'.join(rows) + """
\\bottomrule
\\end{tabular*}
\\end{table*}
""")

# ---- Table: proxy validation (SAM)
P = pd.read_csv('results/sam_proxy_summary.csv').set_index('cls')
rows = []
for c, lab in [('line', 'Line (instances)'), ('block', 'Block (instances)'), ('pit', 'Pit (instances)'), ('line_tile', 'Line (tiles)'), ('block_tile', 'Block (tiles)'), ('pit_tile', 'Pit (tiles)')]:
    r = P.loc[c]
    sk = f"{r.rho_area_skel:.2f} & {r.rho_diag_skel:.2f}" if not np.isnan(r.rho_area_skel) else '-- & --'
    rot = f"{r.rot_area_maxmin:.2f} & {r.rot_diag_maxmin:.2f}" if c == 'line' else '-- & --'
    rows.append(f"{lab} & {int(r.n)} & {r.rho_area:.2f} & {r.rho_diag:.2f} & {sk} & {rot} \\\\")
w('proxy', """\\begin{table}[t]
\\caption{Rank correlation (Spearman's $\\rho$) between bounding-box proxies and pixel-level pseudo-references from SAM masks on 150 tiles, and orientation sensitivity of the line-crack proxies expressed as the median ratio between the largest and the smallest value of the proxy when the crack mask is rotated between 0 and \\SI{90}{\\degree}.}\\label{tab:proxy}
\\begin{tabular*}{\\tblwidth}{@{}lrcccccc@{}}
\\toprule
 & & \\multicolumn{2}{c}{vs. mask area} & \\multicolumn{2}{c}{vs. skeleton length} & \\multicolumn{2}{c}{Rotation max/min} \\\\
\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}
Class & $n$ & area & diagonal & area & diagonal & area & diagonal \\\\
\\midrule
""" + '\n'.join(rows) + """
\\bottomrule
\\end{tabular*}
\\end{table}
""")

# ---- Table: score sensitivity (weights and reference percentiles, GT based)
Wt = pd.read_csv('results/vmps_weight_sensitivity.csv'); Rf = pd.read_csv('results/vmps_reference_sensitivity.csv')
wts = {'Base': '1.0 / 1.5 / 2.5 / 0.5 / 0.5', 'Pit-light': '1.0 / 1.5 / 2.0 / 0.3 / 0.5', 'Pit-heavy': '1.0 / 1.5 / 3.0 / 0.8 / 0.5', 'Crack-heavy': '1.3 / 1.8 / 2.5 / 0.5 / 0.5', 'Equal': '1.0 / 1.0 / 1.0 / 1.0 / 1.0'}
rows = [f"Weights & {r.scenario} ({wts[r.scenario]}) & {r.spearman:.3f} & {r.band_agreement:.3f} & {r.top5pct_overlap:.3f} & {100*r.critical_share:.1f} \\\\" for r in Wt.itertuples()]
for r in Rf.itertuples():
    lab = f"component P{int(r.component_percentile)}" if np.isnan(r.score_percentile) else f"score P{int(r.score_percentile)}"
    rows.append(f"Reference & {lab} & {r.spearman:.3f} & {r.band_agreement:.3f} & -- & {100*r.Critical:.1f} \\\\")
w('sensitivity', """\\begin{table}[t]
\\caption{Sensitivity of the reference-annotation score to the component weights (line / block / pit area / pit count / multi-class) and to the reference percentiles, evaluated on all 11{,}696 tiles against the base configuration: rank correlation of the unclipped score, agreement of the screening band, overlap of the top 5\\% of tiles and share of tiles in the Critical band.}\\label{tab:sens}
\\begin{tabular*}{\\tblwidth}{@{}llcccc@{}}
\\toprule
Factor & Setting & $\\rho$ & Band agreement & Top 5\\% overlap & Critical (\\%) \\\\
\\midrule
""" + '\n'.join(rows) + """
\\bottomrule
\\end{tabular*}
\\end{table}
""")

# ---- Table: confidence threshold sensitivity (main model, mean over seeds)
g = group('y11s_source')
if g:
    rows = []
    for i, cf in enumerate([0.1, 0.15, 0.25, 0.35, 0.5]):
        v = [x['conf_sens'][i] for x in g]
        rows.append(f"{cf:.2f} & {ms([x['mae_N_line'] for x in v],'%.2f')} & {ms([x['mae_N_block'] for x in v],'%.2f')} & {ms([x['mae_N_pit'] for x in v],'%.2f')} & {ms([x['pit_presence_f1'] for x in v])} & {ms([x['mae'] for x in v],'%.1f')} & {ms([x['spearman'] for x in v])} & {ms([x['qwk'] for x in v])} & {ms([x['crit_recall'] for x in v])} \\\\")
    w('conf', """\\begin{table*}[t]
\\caption{Effect of the deployment confidence threshold on the tile-level inventory and score agreement for YOLO11s on the source-separated test subset (mean over three seeds, standard deviation in parentheses in units of the last digit).}\\label{tab:conf}
\\begin{tabular*}{\\tblwidth}{@{}lcccccccc@{}}
\\toprule
Threshold & MAE line & MAE block & MAE pit & Pit F1 & Score MAE & $\\rho$ & $\\kappa_w$ & Critical recall \\\\
\\midrule
""" + '\n'.join(rows) + """
\\bottomrule
\\end{tabular*}
\\end{table*}
""")

# ---- Table: external
rows = []
for pre, model, part in CFG:
    g = [x for x in group(pre) if 'ext_det' in x]
    if not g or part == 'Random': continue
    d = [x['ext_det'] for x in g]; t = [x['ext_tile'] for x in g]; s = [x['ext_section'] for x in g]
    rows.append(f"{model} & {ms([x['map50'] for x in d])} & " + ' & '.join(ms([x['ap50_cls'][c] for x in d]) for c in range(3)) + ' & ' + ' & '.join(ms([x['r_cls'][c] for x in d]) for c in range(3)) +
                f" & {ms([x['mae_N_line'] for x in t],'%.2f')} & {ms([x['mae'] for x in t],'%.1f')} & {ms([x['spearman'] for x in t])} & {ms([x['qwk'] for x in t])} & {ms([x['spearman'] for x in s])} & {ms([x['qwk'] for x in s])} \\\\")
w('external', """\\begin{table*}[t]
\\caption{Zero-shot application of the HighRPD-trained detectors to the 9{,}756 UAV-PDD2023 tiles (mean over three seeds, standard deviation in parentheses in units of the last digit): detection metrics against the mapped annotations, tile-level inventory and score agreement, and agreement of the section score over the 236 source views.}\\label{tab:external}
\\begin{tabular*}{\\tblwidth}{@{}lccccccccccccc@{}}
\\toprule
 & & \\multicolumn{3}{c}{AP$_{50}$} & \\multicolumn{3}{c}{Recall} & \\multicolumn{4}{c}{Tile level} & \\multicolumn{2}{c}{View level} \\\\
\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-12}\\cmidrule(lr){13-14}
Model & mAP$_{50}$ & line & block & pit & line & block & pit & MAE line & Score MAE & $\\rho$ & $\\kappa_w$ & $\\rho$ & $\\kappa_w$ \\\\
\\midrule
""" + '\n'.join(rows) + """
\\bottomrule
\\end{tabular*}
\\end{table*}
""")
# ---- Table: measurement-to-decision summary
rows = []
for pre, model, part in CFG:
    g = group(pre)
    if not g or part == 'Random': continue
    d = [x['det'] for x in g]; t = [x['tile'] for x in g]; s = [x['section'] for x in g]
    rows.append(f"{model} & {ms([x['map50'] for x in d])} & {ms([x['mae_N_line'] for x in t],'%.2f')} / {ms([x['mae_N_block'] for x in t],'%.2f')} / {ms([x['mae_N_pit'] for x in t],'%.2f')} & {ms([x['spearman'] for x in t])} & {ms([x['crit_recall'] for x in t])} & {ms([x['spearman'] for x in s])} & {ms([x['topk_overlap'] for x in s],'%.2f')} & {ms([x['adjacent_acc'] for x in s])} \\\\")
w('summary', r"""\begin{table*}[t]
\caption{Measurement-to-decision summary for the source-separated configurations (mean over three seeds, standard deviation in parentheses in units of the last digit): detection accuracy, tile-level inventory error, tile-level and section-level agreement of the priority score with the annotation-based reference, overlap of the nine highest-ranked sections and agreement of the section band within one band.}\label{tab:summary}
\begin{tabular*}{\tblwidth}{@{}lccccccc@{}}
\toprule
 & & Count MAE & \multicolumn{2}{c}{Tile level} & \multicolumn{3}{c}{Section level} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-8}
Model & mAP$_{50}$ & line / block / pit & score $\rho$ & Critical recall & score $\rho$ & top 10\% overlap & within one band \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table*}
""")
# ---- Table: uncertainty budget
if os.path.exists('results/uncertainty_budget.csv'):
    U = pd.read_csv('results/uncertainty_budget.csv', index_col=0)
    t, s_ = U.loc['tile'], U.loc['section']
    rows = [f"Repeatability (seed to seed) & A & {t.repeatability:.1f} & {s_.repeatability:.1f} \\\\",
            f"Operating point (confidence 0.15 to 0.25) & B & {t.operating_point:.1f} & {s_.operating_point:.1f} \\\\",
            f"Reference ambiguity (line/block label swap, $p$ = {t.p_swap:.2f}) & B & {t.reference_ambiguity:.1f} & {s_.reference_ambiguity:.1f} \\\\",
            f"\\midrule Combined, reproduction of the reference score & & {t.combined_reproduction:.1f} & {s_.combined_reproduction:.1f} \\\\",
            f"Observed dispersion after bias removal & & {t.residual:.1f} & {s_.residual:.1f} \\\\",
            f"Bias (detection minus reference) & & {t.bias:.1f} & {s_.bias:.1f} \\\\",
            f"\\midrule Proxy approximation (box to pixel-level extent) & B & {t.proxy:.1f} & {s_.proxy:.1f} \\\\",
            f"Combined, including proxy approximation & & {t.combined_with_proxy:.1f} & {s_.combined_with_proxy:.1f} \\\\",
            f"Expanded ($k$ = 2) & & {t.expanded_k2:.1f} & {s_.expanded_k2:.1f} \\\\"]
    w('uncertainty', r"""\begin{table}[t]
\caption{Uncertainty budget of the detection-based priority score (YOLO11s, source-separated test set) in score units (0--100): standard uncertainties of the tile score and of the section score $S$, their combination, the observed dispersion and bias, and the additional term for the box proxies relative to a pixel-level extent.}\label{tab:budget}
\begin{tabular*}{\tblwidth}{@{}llcc@{}}
\toprule
Component & Type & Tile & Section \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular*}
\end{table}
""")
print('tables written:', sorted(os.listdir('paper/tables')))
