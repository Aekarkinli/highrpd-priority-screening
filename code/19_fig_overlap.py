"""Figure: ground overlap between consecutive HighRPD frames.
(a) two consecutive frames reassembled from their released tiles, the second registered onto the first;
(b) share of the tiles of a frame that fall on tiles of a later frame, by distance in the capture sequence;
(c) share of validation and test frames that overlap a training frame under the three partitions.
Output: figures/fig_overlap.pdf/.png"""
import json, sys, numpy as np, pandas as pd, cv2, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, 'code')
from importlib import import_module
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.linewidth': 0.6, 'figure.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'mathtext.fontset': 'stix', 'font.serif': ['STIXGeneral', 'DejaVu Serif']})
fo = import_module('16_frame_overlap')
O = pd.read_csv('results/frame_overlap_pairs.csv')
ok = O[(O.inliers >= 30) & ((O.scale - 1).abs() < 0.05) & (O.rot_deg.abs() < 5)]
pair = ok[(ok.idx_gap == 1) & (ok.series == 'DJI')].sort_values('inliers', ascending=False).iloc[len(ok) // 40]
ia, ma, da = fo.mosaic(pair.src_a); ib, mb, db = fo.mosaic(pair.src_b)
sift = cv2.SIFT_create(nfeatures=6000); ka, xa = sift.detectAndCompute(ia, ma); kb, xb = sift.detectAndCompute(ib, mb)
m = [x for x, y in cv2.BFMatcher().knnMatch(xa, xb, k=2) if x.distance < 0.75 * y.distance]
M, _ = cv2.estimateAffinePartial2D(np.float32([ka[x.queryIdx].pt for x in m]), np.float32([kb[x.trainIdx].pt for x in m]), method=cv2.RANSAC, ransacReprojThreshold=4.0)
Minv = cv2.invertAffineTransform(M)
# union canvas holding frame a and frame b warped into the coordinates of a
H, W = ib.shape; cb = np.float32([[0, 0], [W, 0], [W, H], [0, H]]) @ Minv[:, :2].T + Minv[:, 2]
x0, y0 = min(0, cb[:, 0].min()), min(0, cb[:, 1].min()); x1, y1 = max(ia.shape[1], cb[:, 0].max()), max(ia.shape[0], cb[:, 1].max())
T_ = np.float32([[1, 0, -x0], [0, 1, -y0]]); size = (int(x1 - x0), int(y1 - y0))
A2 = cv2.warpAffine(ia, T_, size); MA = cv2.warpAffine(ma, T_, size, flags=cv2.INTER_NEAREST)
Mb = np.vstack([Minv, [0, 0, 1]]); Tb = (np.vstack([T_, [0, 0, 1]]) @ Mb)[:2]
B2 = cv2.warpAffine(ib, Tb, size); MB = cv2.warpAffine(mb, Tb, size, flags=cv2.INTER_NEAREST)
rgb = np.full(size[::-1] + (3,), 255, np.uint8)
ga = A2[..., None].repeat(3, 2); rgb[MA > 0] = ga[MA > 0]
orange = np.array([213, 94, 0], float)
only_b = (MB > 0) & (MA == 0); both = (MB > 0) & (MA > 0)
rgb[only_b] = (0.55 * B2[only_b, None] + 0.45 * orange).astype(np.uint8)
rgb[both] = (0.55 * A2[both, None] + 0.45 * orange).astype(np.uint8)

fig = plt.figure(figsize=(7.2, 4.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.42, wspace=0.32)
ax = fig.add_subplot(gs[0, 0]); ax.imshow(rgb)
ax.set_title('(a) frame %s (grey), frame %s\nregistered onto it (orange)' % (pair.src_a.split('_')[-1], pair.src_b.split('_')[-1]), fontsize=7); ax.axis('off')
ax = fig.add_subplot(gs[0, 1:])
gaps = sorted(ok.idx_gap.unique()); data = [ok[ok.idx_gap == g].shared_tiles_a.values * 100 for g in gaps]
bp = ax.boxplot(data, positions=gaps, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color='black', lw=1))
for b in bp['boxes']: b.set(facecolor='#9ecae1', edgecolor='#3182bd', lw=0.6)
n_all = O.groupby('idx_gap').size(); n_ok = ok.groupby('idx_gap').size()
for g in gaps: ax.text(g, 103, f'{n_ok.get(g, 0)}', ha='center', fontsize=5.5, color='#555555')
ax.set_ylim(0, 110); ax.set_xlabel('Distance in capture sequence (frames)'); ax.set_ylabel('Tiles on a tile of the other frame (%)')
ax.set_title('(b) shared pavement between frames', fontsize=7)
ax = fig.add_subplot(gs[1, :])
T = pd.read_csv('results/table_splits.csv'); T = T[T.subset.isin(['val', 'test'])]
labs = ['Random', 'Frame-separated', 'Stretch-separated']; T['partition'] = T.partition.replace({'Source-separated': 'Frame-separated'})
AUD = json.load(open('results/revision/val_overlap_check.json'))       # exhaustive audit of the stretch-separated partition
sv = (T.partition == 'Stretch-separated') & (T.subset == 'val'); T.loc[sv, 'groups_overlapping_train'] = len(AUD['val_frames']) / T.loc[sv, 'frames']
x = np.arange(3); w = 0.36
for k, (sub, col) in enumerate([('val', '#9ecae1'), ('test', '#3182bd')]):
    v = [T[(T.partition == p) & (T.subset == sub)].groups_overlapping_train.iloc[0] * 100 for p in labs]
    ax.bar(x + (k - .5) * w, v, w, color=col, label='validation' if sub == 'val' else 'test')
    for xi, vi in zip(x + (k - .5) * w, v): ax.text(xi, vi + 2, f'{vi:.1f}', ha='center', fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylim(0, 115); ax.set_ylabel('Frames overlapping a\ntraining frame (%)')
ax.legend(frameon=False, fontsize=6.5, loc='upper right'); ax.set_title('(c) overlap with training frames by partition', fontsize=7)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
fig.savefig('figures/fig_overlap.pdf', bbox_inches='tight'); fig.savefig('figures/fig_overlap.png', bbox_inches='tight', dpi=200); print('written', pair.src_a, pair.src_b, pair.inliers)
