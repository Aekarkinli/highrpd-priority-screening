"""Figures of the stretch-separated evaluation at the operating threshold selected on validation tiles.
Regenerates fig_error_propagation, fig_threshold, fig_frame_map, fig_examples and fig_external from the cached
predictions (results/runs/<run>/pred_*_conf005.json) and results/revision/<tag>/revision.json."""
import sys, json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp, matplotlib.patheffects as pe
from PIL import Image
from importlib import import_module
sys.path.insert(0, 'code'); from inventory import *
RA = import_module('18_revision_analyses')
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.linewidth': 0.6, 'figure.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'stix', 'font.serif': ['STIXGeneral', 'DejaVu Serif']})
COL = {0: '#0072B2', 1: '#D55E00', 2: '#009E73'}; NAME = {0: 'line', 1: 'block', 2: 'pit'}   # Okabe-Ito palette
BC = {'Low': '#dddddd', 'Moderate': '#fdd49e', 'High': '#fc8d59', 'Critical': '#b30000'}; BL = list(BC)
HALO = [pe.Stroke(linewidth=3.2, foreground='white'), pe.Normal()]
TAG = 'y11s_stretch'; RV = json.load(open(f'results/revision/{TAG}/revision.json')); RUNS = RV['runs']; RUN = RUNS[0]; T = RV['t_sel']
RV8 = json.load(open('results/revision/y8s_stretch/revision.json'))
GT = pd.read_csv('results/tiles_gt_inventory.csv'); D = GT[GT.split_stretch == 'test'].reset_index(drop=True)
stems = list(D.stem); src = D.src.values


def frame_tables(run):
    P = json.load(open(f'results/runs/{run}/pred_highrpd_conf005.json'))
    Fp = RA.features(P, stems, T); _, p = RA.score(Fp)
    return P, Fp, p, RA.group_table(Fp, p, stems, src)


Fg = RA.features({s: [b + (1.0,) for b in read_yolo_txt(f'data/HighRPD/labels/{s}.txt')] for s in stems}, stems); _, g = RA.score(Fg)
Gref = RA.group_table(Fg, g, stems, src)
TAB = {r: frame_tables(r) for r in RUNS}
P0, Fp0, p0, G0 = TAB[RUN]
Gmean = np.mean([TAB[r][3].loc[Gref.index].S_VMPS.values for r in RUNS], 0)
mt, mg = RV[f't{T:.2f}']['tile'], RV[f't{T:.2f}']['group']


def draw_box(ax, b, color, lw=1.8, ls='-', halo=True):
    c, x, y, w, h = b[:5]
    r = mp.Rectangle(((x - w / 2) * 640, (y - h / 2) * 640), w * 640, h * 640, fill=False, ec=color, lw=lw, ls=ls)
    if halo: r.set_path_effects(HALO)
    ax.add_patch(r)


# ---------- error propagation: tiles (one seed), bands (all seeds), frames (seed mean)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
hb = ax[0].hexbin(g, p0, gridsize=25, bins='log', cmap='Blues', mincnt=1, linewidths=0.2); ax[0].plot([0, 100], [0, 100], 'k-', lw=0.6)
for t in [25, 50, 75]: ax[0].axvline(t, color='k', lw=0.4, ls=':'); ax[0].axhline(t, color='k', lw=0.4, ls=':')
cb = fig.colorbar(hb, ax=ax[0], fraction=0.046, pad=0.02); cb.set_label('tiles (log)', fontsize=6); cb.ax.tick_params(labelsize=5.5)
ax[0].set_xlabel('Score from reference annotations'); ax[0].set_ylabel('Score from detections')
ax[0].set_title('(a) Tiles, one seed', fontsize=7.5)
cm = np.array(RV[f't{T:.2f}']['tile_cm']); cmn = cm / cm.sum(1, keepdims=True)
ax[1].imshow(cmn, cmap='Blues', vmin=0, vmax=1)
for i in range(4):
    for j in range(4): ax[1].text(j, i, f'{100*cmn[i,j]:.0f}%', ha='center', va='center', fontsize=6, color='white' if cmn[i, j] > 0.5 else 'black')
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(BL, fontsize=6); ax[1].set_yticks(range(4)); ax[1].set_yticklabels(BL, fontsize=6)
ax[1].set_xlabel('Band from detections'); ax[1].set_ylabel('Band from reference')
ax[1].set_title('(b) Tile bands, three seeds: $\\kappa_w$ = %.2f' % mt['qwk'][0], fontsize=7.5)
ax[2].scatter(Gref.S_VMPS, Gmean, s=11, color=COL[1], lw=0); ax[2].plot([0, 100], [0, 100], 'k-', lw=0.6)
for t in [25, 50, 75]: ax[2].axvline(t, color='k', lw=0.4, ls=':'); ax[2].axhline(t, color='k', lw=0.4, ls=':')
ax[2].set_xlabel('Frame score from reference'); ax[2].set_ylabel('Frame score from detections')
ax[2].set_title('(c) Frames (n = %d), seed mean' % len(Gref), fontsize=7.5)
ax[2].text(0.03, 0.97, '$\\rho$ = %.2f\nMAE = %.1f\nwithin one band %.1f%%' % (mg['spearman'][0], mg['mae'][0], 100 * mg['adjacent'][0]), transform=ax[2].transAxes, fontsize=6, va='top')
for a in [ax[0], ax[2]]: a.set_xlim(-2, 102); a.set_ylim(-2, 102); a.set_aspect('equal'); a.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig('figures/fig_error_propagation.pdf'); plt.savefig('figures/fig_error_propagation.png', dpi=200); plt.close()

# ---------- operating threshold: validation selection curve and test curves
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
for rv, lab, c in [(RV, 'YOLO11s', '#0072B2'), (RV8, 'YOLOv8s', '#D55E00')]:
    v = pd.DataFrame(rv['val_curve']); ax[0].errorbar(v.t, v.val_qwk, yerr=v.val_qwk_sd, fmt='o-', ms=3, lw=1.2, capsize=1.5, color=c, label=lab)
ax[0].set_ylabel('Tile $\\kappa_w$ on validation tiles'); ax[0].set_title('(a) Selection on validation tiles', fontsize=8); ax[0].legend(frameon=False, fontsize=6.5)
tc = pd.DataFrame(RV['test_curve'])
for key, lab, c, ls in [('tile_rho', "tile $\\rho$", '#0072B2', '--'), ('tile_qwk', 'tile $\\kappa_w$', '#D55E00', '--'),
                        ('grp_rho', "frame $\\rho$", '#0072B2', '-'), ('grp_qwk', 'frame $\\kappa_w$', '#D55E00', '-'), ('grp_crit', 'frame Critical recall', '#009E73', '-')]:
    ax[1].plot(tc.t, tc[key], marker='o', ms=2.5, lw=1.1, ls=ls, color=c, label=lab)
ax[1].set_ylabel('Agreement on test set (YOLO11s)'); ax[1].set_title('(b) Test set', fontsize=8); ax[1].legend(frameon=False, fontsize=6, loc='upper right')
for a in ax:
    a.set_xlabel('Confidence threshold'); a.set_ylim(0, 1.05); a.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5]); a.spines[['top', 'right']].set_visible(False)
    a.axvline(T, color='k', lw=0.6, ls='-'); a.axvline(0.25, color='k', lw=0.5, ls=':')
plt.tight_layout(); plt.savefig('figures/fig_threshold.pdf'); plt.savefig('figures/fig_threshold.png', dpi=200); plt.close()


# ---------- frame maps: reference against detections for two test frames
def frame_panel(ax, sub, col, npit, title):
    R0, R1, C0, C1 = sub.a.min(), sub.a.max(), sub.b.min(), sub.b.max()
    for r in sub.itertuples():
        v = getattr(r, col); bnd = band(pd.Series([v]))[0]
        ax.add_patch(mp.Rectangle((r.b - C0, R1 - r.a), 1, 1, fc=BC[bnd], ec='k', lw=0.3))
        ax.text(r.b - C0 + 0.5, R1 - r.a + 0.5, f'{v:.0f}', ha='center', va='center', fontsize=5)
        if getattr(r, npit) > 0: ax.plot(r.b - C0 + 0.85, R1 - r.a + 0.85, 'o', ms=2.2, color=COL[2])
    ax.set_xlim(0, C1 - C0 + 1); ax.set_ylim(0, R1 - R0 + 1); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=6.5)
    ax.text(0.02, -0.03, f'{(C1-C0+1)*640*0.0063:.0f} m x {(R1-R0+1)*640*0.0063:.0f} m', transform=ax.transAxes, fontsize=5.5, va='top')


LEG = [mp.Patch(fc=BC[b], ec='k', lw=0.3, label=b) for b in BC] + [plt.Line2D([], [], marker='o', color=COL[2], ls='', ms=3, label='pit present')]
cand = Gref[Gref.n_tiles >= 30].copy(); cand['pred'] = G0.loc[cand.index].S_VMPS; cand['d'] = (cand.S_VMPS - cand.pred).abs()
pick = [cand.sort_values('S_VMPS', ascending=False).index[0], cand.sort_values('d', ascending=False).index[0]]
TT = D.assign(s_gt=g, s_pred=p0, N_pit_pred=Fp0.N_pit.values)
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
for k, s in enumerate(pick):
    sub = TT[TT.src == s]; gr_, pr_ = Gref.loc[s], G0.loc[s]
    frame_panel(axs[k, 0], sub, 's_gt', 'N_pit', f'({"abcd"[2*k]}) frame {s}, reference annotations\nline {int(gr_.N_line)}, block {int(gr_.N_block)}, pit {int(gr_.N_pit)}; frame score {gr_.S_VMPS:.1f}')
    frame_panel(axs[k, 1], sub, 's_pred', 'N_pit_pred', f'({"abcd"[2*k+1]}) frame {s}, detections\nline {int(pr_.N_line)}, block {int(pr_.N_block)}, pit {int(pr_.N_pit)}; frame score {pr_.S_VMPS:.1f}')
fig.legend(handles=LEG, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.005), fontsize=7)
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig('figures/fig_frame_map.pdf'); plt.close()
print('frame map frames', pick)

# ---------- agreement and failure examples: reference boxes solid with white halo, detections dashed with black halo
TT['db'] = band(TT.s_pred).cat.codes - band(TT.s_gt).cat.codes; TT['n_pred_pit'] = Fp0.N_pit.values
ex = [(TT[(TT.db == 0) & (TT.s_gt > 60)].sample(1, random_state=1).stem.iloc[0], 'agreement'),
      (TT[(TT.db == 0) & (TT.n_pred_pit > 0) & (TT.N_pit > 0)].sample(1, random_state=4).stem.iloc[0], 'agreement, pit found'),
      (TT[(TT.db == 0) & (TT.s_gt.between(25, 50))].sample(1, random_state=5).stem.iloc[0], 'agreement'),
      (TT[(TT.db <= -2) & (TT.s_pred == 0)].sample(1, random_state=2).stem.iloc[0], 'missed tile'),
      (TT[(TT.db >= 2)].sample(1, random_state=3).stem.iloc[0], 'over-scored'),
      (TT[(TT.db <= -2) & (TT.s_pred > 0)].sample(1, random_state=8).stem.iloc[0], 'under-scored, partial detection')]
fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.4))
for k, (ax, (st, lab)) in enumerate(zip(axs.ravel(), ex)):
    ax.imshow(Image.open(f'data/HighRPD/images/{st}.jpg'))
    for b in read_yolo_txt(f'data/HighRPD/labels/{st}.txt'): draw_box(ax, b, COL[b[0]])
    for b in P0.get(st, []):
        if b[5] < T: continue
        c, x, y, w, h = b[:5]
        rr = mp.Rectangle(((x - w / 2) * 640, (y - h / 2) * 640), w * 640, h * 640, fill=False, ec=COL[int(c)], lw=1.6, ls=(0, (3, 2)))
        rr.set_path_effects([pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()]); ax.add_patch(rr)
    q = TT[TT.stem == st].iloc[0]; gb, pb = band(pd.Series([q.s_gt]))[0], band(pd.Series([q.s_pred]))[0]
    ax.set_title(f'({"abcdef"[k]}) {lab}\nref. {q.s_gt:.0f} ({gb}), det. {q.s_pred:.0f} ({pb})', fontsize=6.3); ax.axis('off')
h = [mp.Patch(fc='none', ec=COL[c], lw=1.8, label=NAME[c]) for c in COL] + [plt.Line2D([], [], color='0.3', lw=1.8, ls='-', label='reference box (white halo)'), plt.Line2D([], [], color='0.3', lw=1.6, ls=(0, (3, 2)), label='detected box (dashed, black halo)')]
fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.005), fontsize=6.5)
plt.tight_layout(rect=(0, 0.035, 1, 1), h_pad=2.0, w_pad=0.6); plt.savefig('figures/fig_examples.pdf'); plt.savefig('figures/fig_examples.png', dpi=200); plt.close()

# ---------- external data: bands and tile scores at native and matched resolution (one seed)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
for k, (name, idxf, lbdir, lab) in enumerate([('native', 'results/uavpdd_tiles_index.csv', 'data/uavpdd_tiles/labels', 'native, about 3 mm'),
                                             ('gsd_matched', 'results/uavpdd_tiles_gsd_index.csv', 'data/uavpdd_tiles_gsd/labels', 'resampled to 6.3 mm')]):
    U = pd.read_csv(idxf); us = list(U.stem)
    UG = RA.features({s: [b + (1.0,) for b in read_yolo_txt(f'{lbdir}/{s}.txt')] for s in us}, us); _, ug = RA.score(UG)
    UP = RA.features(json.load(open(f'results/runs/{RUN}/pred_uavpdd_{name}_conf005.json')), us, T); _, up = RA.score(UP)
    sg = band(pd.Series(ug)).value_counts(normalize=True).reindex(BL).values; sp = band(pd.Series(up)).value_counts(normalize=True).reindex(BL).values
    x = np.arange(4) + 5 * k
    ax[0].bar(x - 0.2, 100 * sg, 0.4, color='#0072B2', label='annotations' if k == 0 else None); ax[0].bar(x + 0.2, 100 * sp, 0.4, color='#D55E00', label='detections' if k == 0 else None)
    hb = ax[k + 1].hexbin(ug, up, gridsize=25, bins='log', cmap='Blues', mincnt=1, linewidths=0.2); ax[k + 1].plot([0, 100], [0, 100], 'k-', lw=0.6)
    e = RV['external'][name]
    ax[k + 1].set_title(f'({"bc"[k]}) {lab}: $\\rho$ = {e["tile_rho"][0]:.2f}', fontsize=7.5); ax[k + 1].set_aspect('equal')
    ax[k + 1].set_xlabel('Score from reference annotations'); ax[k + 1].set_ylabel('Score from detections')
ax[0].set_xticks(list(range(4)) + list(range(5, 9))); ax[0].set_xticklabels(['L', 'M', 'H', 'C'] * 2, fontsize=6)
ax[0].text(1.5, 97, 'native', ha='center', fontsize=6.5); ax[0].text(6.5, 97, 'resampled', ha='center', fontsize=6.5)
ax[0].set_ylabel('Share of tiles (%)'); ax[0].set_ylim(0, 100); ax[0].legend(frameon=False, fontsize=6, loc='center right'); ax[0].set_title('(a) Bands', fontsize=7.5)
for a in ax: a.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig('figures/fig_external.pdf'); plt.close()
print('done')
