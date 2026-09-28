"""Revised figures for the manuscript (colour-blind-safe class colours, haloed boxes, tighter section maps,
density scatter and annotated confusion matrix). Regenerates fig_samples, fig_section_map_gt, fig_error_propagation,
fig_section_map and fig_examples."""
import sys, json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mp, matplotlib.patheffects as pe
from PIL import Image
sys.path.insert(0, 'code'); from inventory import *
from ultralytics import YOLO
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.linewidth': 0.6, 'figure.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'stix', 'font.serif': ['STIXGeneral', 'DejaVu Serif']})
COL = {0: '#0072B2', 1: '#D55E00', 2: '#009E73'}; NAME = {0: 'line', 1: 'block', 2: 'pit'}   # Okabe-Ito palette
BC = {'Low': '#dddddd', 'Moderate': '#fdd49e', 'High': '#fc8d59', 'Critical': '#b30000'}; BL = list(BC)
HALO = [pe.Stroke(linewidth=3.2, foreground='white'), pe.Normal()]
RUN = 'y11s_source_s42'
df = pd.read_csv('results/tiles_gt_inventory.csv')


def draw_box(ax, b, color, lw=1.8, ls='-', halo=True):
    c, x, y, w, h = b[:5]
    r = mp.Rectangle(((x - w / 2) * 640, (y - h / 2) * 640), w * 640, h * 640, fill=False, ec=color, lw=lw, ls=ls)
    if halo: r.set_path_effects(HALO)
    ax.add_patch(r)


def tag(ax, text, xy, color):
    ax.text(xy[0] + 3, xy[1] + 3, text, fontsize=5.5, color='white', va='top', ha='left', bbox=dict(boxstyle='round,pad=0.15', fc=color, ec='none', alpha=0.9))


# ---------- Fig: sample tiles (2 x 3, annotation examples chosen to cover the classes and their combinations)
picks = [df[(df.N_line > 0) & (df.N_block == 0) & (df.N_pit == 0) & (df.N_line <= 3)].sample(1, random_state=3).stem.iloc[0],
         df[(df.N_line == 0) & (df.N_block > 0) & (df.N_pit == 0)].sample(1, random_state=4).stem.iloc[0],
         df[(df.N_pit >= 2) & (df.A_pit < 0.04) & (df.N_line == 0) & (df.N_block == 0)].sample(1, random_state=9).stem.iloc[0],
         df[(df.N_line >= 4) & (df.N_block == 0)].sample(1, random_state=7).stem.iloc[0],
         df[(df.C_multi == 1) & (df.N_pit == 0)].sample(1, random_state=10).stem.iloc[0],
         df[(df.C_multi == 1) & (df.N_pit > 0)].sample(1, random_state=6).stem.iloc[0]]
fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0))
for k, (ax, st) in enumerate(zip(axs.ravel(), picks)):
    ax.imshow(Image.open(f'data/HighRPD/images/{st}.jpg'))
    for b in read_yolo_txt(f'data/HighRPD/labels/{st}.txt'):
        draw_box(ax, b, COL[b[0]]); tag(ax, NAME[b[0]], ((b[1] - b[3] / 2) * 640, (b[2] - b[4] / 2) * 640), COL[b[0]])
    r = df[df.stem == st].iloc[0]
    ax.set_title(f'({"abcdef"[k]}) score {r.VMPS:.0f}, {r.priority}', fontsize=7.5); ax.axis('off')
h = [mp.Patch(fc='none', ec=COL[c], lw=1.8, label=NAME[c]) for c in COL]
fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig('figures/fig_samples.pdf'); plt.savefig('figures/fig_samples.png', dpi=200); plt.close()


# ---------- section map helper (tight crop around occupied tiles)
def section_panel(ax, sub, score_col, npit_col, title):
    R0, R1, C0, C1 = sub.a.min(), sub.a.max(), sub.b.min(), sub.b.max()
    for r in sub.itertuples():
        v = getattr(r, score_col); bnd = band(pd.Series([v]))[0]
        ax.add_patch(mp.Rectangle((r.b - C0, R1 - r.a), 1, 1, fc=BC[bnd], ec='k', lw=0.3))
        ax.text(r.b - C0 + 0.5, R1 - r.a + 0.5, f'{v:.0f}', ha='center', va='center', fontsize=5)
        if getattr(r, npit_col) > 0: ax.plot(r.b - C0 + 0.85, R1 - r.a + 0.85, 'o', ms=2.2, color=COL[2])
    ax.set_xlim(0, C1 - C0 + 1); ax.set_ylim(0, R1 - R0 + 1); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=6.5)
    ax.text(0.02, -0.03, f'{(C1-C0+1)*4:.0f} m x {(R1-R0+1)*4:.0f} m', transform=ax.transAxes, fontsize=5.5, va='top')


LEG = [mp.Patch(fc=BC[b], ec='k', lw=0.3, label=b) for b in BC] + [plt.Line2D([], [], marker='o', color=COL[2], ls='', ms=3, label='pit present')]

# ---------- Fig: section maps from annotations (two source groups)
sec = pd.read_csv('results/sections_gt_inventory.csv')
srcs = ['1x00157', '1x00132']
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
for k, (ax, s) in enumerate(zip(axs, srcs)):
    sub = df[df.src == s]; ss = sec[sec.src == s].iloc[0]
    section_panel(ax, sub, 'VMPS', 'N_pit', f'({"ab"[k]}) {s}: {int(ss.n_tiles)} tiles, line {int(ss.N_line)}, block {int(ss.N_block)}, pit {int(ss.N_pit)}; section score {ss.S_VMPS:.1f}')
fig.legend(handles=LEG, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01), fontsize=7)
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig('figures/fig_section_map_gt.pdf'); plt.close()

# ---------- Fig: error propagation (density scatter, confusion with percentages first, annotated section panel)
P = pd.read_csv(f'results/runs/{RUN}/tile_pred_inventory.csv'); S = json.load(open(f'results/runs/{RUN}/summary.json'))
SEC = pd.read_csv(f'results/runs/{RUN}/section_pred_inventory.csv', index_col=0)
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
hb = ax[0].hexbin(P.VMPS_gt, P.VMPS_pred, gridsize=25, bins='log', cmap='Blues', mincnt=1, linewidths=0.2)
ax[0].plot([0, 100], [0, 100], 'k-', lw=0.6)
for t in [25, 50, 75]: ax[0].axvline(t, color='k', lw=0.4, ls=':'); ax[0].axhline(t, color='k', lw=0.4, ls=':')
cb = fig.colorbar(hb, ax=ax[0], fraction=0.046, pad=0.02); cb.set_label('tiles (log)', fontsize=6); cb.ax.tick_params(labelsize=5.5)
ax[0].set_xlabel('Score from reference annotations'); ax[0].set_ylabel('Score from detections')
ax[0].set_title('(a) Tiles: $\\rho$ = %.2f, MAE = %.0f' % (S['tile']['spearman'], S['tile']['mae']), fontsize=7.5)
cm = np.array(S['tile']['cm']); cmn = cm / cm.sum(1, keepdims=True)
ax[1].imshow(cmn, cmap='Blues', vmin=0, vmax=1)
for i in range(4):
    for j in range(4): ax[1].text(j, i, f'{100*cmn[i,j]:.0f}%\n({cm[i,j]})', ha='center', va='center', fontsize=5.8, color='white' if cmn[i, j] > 0.5 else 'black')
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(BL, fontsize=6); ax[1].set_yticks(range(4)); ax[1].set_yticklabels(BL, fontsize=6)
ax[1].set_xlabel('Band from detections'); ax[1].set_ylabel('Band from reference')
ax[1].set_title('(b) Bands: $\\kappa_w$ = %.2f, exact %.0f%%' % (S['tile']['qwk'], 100 * S['tile']['band_acc']), fontsize=7.5)
ax[2].scatter(SEC.S_VMPS_gt, SEC.S_VMPS_pred, s=11, color=COL[1], lw=0); ax[2].plot([0, 100], [0, 100], 'k-', lw=0.6)
for t in [25, 50, 75]: ax[2].axvline(t, color='k', lw=0.4, ls=':'); ax[2].axhline(t, color='k', lw=0.4, ls=':')
ax[2].set_xlabel('Section score from reference'); ax[2].set_ylabel('Section score from detections')
ax[2].set_title('(c) Sections (n = %d)' % len(SEC), fontsize=7.5)
ax[2].text(0.03, 0.97, '$\\rho$ = %.2f\nMAE = %.1f\nwithin one band %.0f%%' % (S['section']['spearman'], S['section']['mae'], 100 * S['section']['adjacent_acc']), transform=ax[2].transAxes, fontsize=6, va='top')
for a in [ax[0], ax[2]]: a.set_xlim(-2, 102); a.set_ylim(-2, 102); a.set_aspect('equal'); a.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig('figures/fig_error_propagation.pdf'); plt.close()

# ---------- Fig: section maps, reference vs detections (two test groups), tight crop
GT = df.set_index('stem')
cand = SEC[(SEC.n_tiles_gt >= 30)].copy(); cand['d'] = (cand.S_VMPS_gt - cand.S_VMPS_pred).abs()
srcs = [cand.sort_values('S_VMPS_gt', ascending=False).index[0], cand.sort_values('d', ascending=False).index[0]]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4))
for k, s in enumerate(srcs):
    sub = P[P.src == s].copy(); g = GT.loc[sub.stem]; sub['a'] = g.a.values; sub['b'] = g.b.values; sub['N_pit_gt'] = g.N_pit.values
    sc = SEC.loc[s]
    section_panel(axs[k, 0], sub, 'VMPS_gt', 'N_pit_gt', f'({"abcd"[2*k]}) {s}, reference annotations\nline {int(sc.N_line_gt)}, block {int(sc.N_block_gt)}, pit {int(sc.N_pit_gt)}; section score {sc.S_VMPS_gt:.1f}')
    section_panel(axs[k, 1], sub, 'VMPS_pred', 'N_pit', f'({"abcd"[2*k+1]}) {s}, detections\nline {int(sc.N_line_pred)}, block {int(sc.N_block_pred)}, pit {int(sc.N_pit_pred)}; section score {sc.S_VMPS_pred:.1f}')
fig.legend(handles=LEG, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.005), fontsize=7)
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig('figures/fig_section_map.pdf'); plt.close()
print('section map groups', srcs)

# ---------- Fig: agreement and failure examples (2 x 3): reference boxes solid with white halo, detections dashed dark
model = YOLO(f'runs/{RUN}/weights/best.pt')
P['db'] = (band(P.VMPS_pred).cat.codes - band(P.VMPS_gt).cat.codes)
ex = [(P[(P.db == 0) & (P.VMPS_gt > 60)].sample(1, random_state=1).stem.iloc[0], 'agreement'),
      (P[(P.db == 0) & (P.N_pit > 0) & (P.VMPS_gt < 50)].sample(1, random_state=4).stem.iloc[0], 'agreement, pit found'),
      (P[(P.db == 0) & (P.VMPS_gt.between(25, 50))].sample(1, random_state=5).stem.iloc[0], 'agreement'),
      (P[(P.db <= -2) & (P.VMPS_pred == 0)].sample(1, random_state=2).stem.iloc[0], 'missed tile'),
      (P[(P.db >= 2)].sample(1, random_state=3).stem.iloc[0], 'over-scored, class confusion'),
      (P[(P.db <= -2) & (P.VMPS_pred > 0)].sample(1, random_state=8).stem.iloc[0], 'under-scored, partial detection')]
res = {r.path.replace('\\', '/').split('/')[-1][:-4]: r for r in model.predict([f'data/HighRPD/images/{s}.jpg' for s, _ in ex], conf=0.25, iou=0.7, verbose=False)}
fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.4))
for k, (ax, (st, lab)) in enumerate(zip(axs.ravel(), ex)):
    ax.imshow(Image.open(f'data/HighRPD/images/{st}.jpg'))
    for b in read_yolo_txt(f'data/HighRPD/labels/{st}.txt'): draw_box(ax, b, COL[b[0]], lw=1.8, ls='-')
    r = res[st]
    for c, (x, y, w, h) in zip(r.boxes.cls.cpu().numpy(), r.boxes.xywhn.cpu().numpy()):
        rr = mp.Rectangle(((x - w / 2) * 640, (y - h / 2) * 640), w * 640, h * 640, fill=False, ec=COL[int(c)], lw=1.6, ls=(0, (3, 2)))
        rr.set_path_effects([pe.Stroke(linewidth=3.0, foreground='black'), pe.Normal()]); ax.add_patch(rr)
    q = P[P.stem == st].iloc[0]; gb, pb = band(pd.Series([q.VMPS_gt]))[0], band(pd.Series([q.VMPS_pred]))[0]
    ax.set_title(f'({"abcdef"[k]}) {lab}\nref. {q.VMPS_gt:.0f} ({gb}), det. {q.VMPS_pred:.0f} ({pb})', fontsize=6.3); ax.axis('off')
h = [mp.Patch(fc='none', ec=COL[c], lw=1.8, label=NAME[c]) for c in COL] + [plt.Line2D([], [], color='0.3', lw=1.8, ls='-', label='reference box (white halo)'), plt.Line2D([], [], color='0.3', lw=1.6, ls=(0, (3, 2)), label='detected box (dashed, black halo)')]
fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.005), fontsize=6.5)
plt.tight_layout(rect=(0, 0.035, 1, 1), h_pad=2.0, w_pad=0.6); plt.savefig('figures/fig_examples.pdf'); plt.savefig('figures/fig_examples.png', dpi=200); plt.close()

# ---------- Fig: confidence-threshold sensitivity (YOLO11s, three seeds)
import glob
runs = [json.load(open(p)) for p in sorted(glob.glob('results/runs/y11s_source_s*/summary.json'))]
th = [r['conf'] for r in runs[0]['conf_sens']]
def stat(key):
    v = np.array([[c[key] for c in r['conf_sens']] for r in runs]); return v.mean(0), v.min(0), v.max(0)
fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.6))
for key, lab, c in [('spearman', "Spearman's $\\rho$", '#0072B2'), ('qwk', 'weighted $\\kappa_w$', '#D55E00'), ('crit_recall', 'Critical recall', '#009E73')]:
    m, lo, hi = stat(key); ax[0].plot(th, m, 'o-', ms=3, lw=1.2, color=c, label=lab); ax[0].fill_between(th, lo, hi, color=c, alpha=0.2)
ax[0].set_ylim(0, 1); ax[0].set_ylabel('Agreement with reference'); ax[0].legend(frameon=False, fontsize=6.5); ax[0].set_title('(a)', fontsize=8)
for key, lab, c in [('mae_N_line', 'line', '#0072B2'), ('mae_N_block', 'block', '#D55E00'), ('mae_N_pit', 'pit', '#009E73')]:
    m, lo, hi = stat(key); ax[1].plot(th, m, 'o-', ms=3, lw=1.2, color=c, label=lab); ax[1].fill_between(th, lo, hi, color=c, alpha=0.2)
ax[1].set_ylabel('Count MAE (boxes per tile)'); ax[1].legend(frameon=False, fontsize=6.5); ax[1].set_title('(b)', fontsize=8)
for a in ax: a.set_xlabel('Confidence threshold'); a.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5]); a.axvline(0.25, color='k', lw=0.5, ls=':'); a.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig('figures/fig_threshold.pdf'); plt.savefig('figures/fig_threshold.png', dpi=200); plt.close()

# ---------- Fig: external dataset (zero-shot)
E = pd.read_csv(f'results/runs/{RUN}/ext_tile_inventory.csv')
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
sg = band(E.VMPS_gt).value_counts(normalize=True).reindex(BL); sp = band(E.VMPS_pred).value_counts(normalize=True).reindex(BL)
x = np.arange(4); ax[0].bar(x - 0.18, 100 * sg, 0.36, color='#0072B2', label='annotations'); ax[0].bar(x + 0.18, 100 * sp, 0.36, color='#D55E00', label='detections')
sgv, spv = sg.values, sp.values
for i in range(4): ax[0].text(x[i] - 0.18, 100 * sgv[i] + 1, f'{100*sgv[i]:.0f}', ha='center', fontsize=6); ax[0].text(x[i] + 0.18, 100 * spv[i] + 1, f'{100*spv[i]:.0f}', ha='center', fontsize=6)
ax[0].set_xticks(x); ax[0].set_xticklabels(BL); ax[0].set_ylabel('Share of tiles (%)'); ax[0].set_ylim(0, 100); ax[0].legend(frameon=False, fontsize=6.5); ax[0].set_title('(a) UAV-PDD2023 screening bands', fontsize=8)
hb = ax[1].hexbin(E.VMPS_gt, E.VMPS_pred, gridsize=25, bins='log', cmap='Blues', mincnt=1, linewidths=0.2); ax[1].plot([0, 100], [0, 100], 'k-', lw=0.6)
cb = fig.colorbar(hb, ax=ax[1], fraction=0.046, pad=0.02); cb.set_label('tiles (log)', fontsize=6); cb.ax.tick_params(labelsize=5.5)
ax[1].set_xlabel('Score from reference annotations'); ax[1].set_ylabel('Score from detections'); ax[1].set_title('(b) Tiles: $\\rho$ = %.2f' % S['ext_tile']['spearman'], fontsize=8); ax[1].set_aspect('equal')
for a in ax: a.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig('figures/fig_external.pdf'); plt.close()
print('done')
