"""Frame-separated training subset matched in size and class composition to the stretch-separated training subset.
Whole frames are removed at random from the frame-separated training subset until it holds the same number of tiles as
the stretch-separated training subset; among 2000 draws within 15 tiles of the target, the one with class proportions
closest to the stretch-separated training subset is kept. Validation uses the frame-separated validation subset.
The subset keeps overlapping neighbours of the common contrast tiles (code/22_leakage_contrast.py) unless they are drawn.
Outputs: data/splits/sourcematched_train.txt, highrpd_sourcematched.yaml, results/revision/matched_split.json"""
import os, json, numpy as np, pandas as pd

G = pd.read_csv('results/tiles_gt_inventory.csv')
O = pd.read_csv('results/frame_overlap_pairs.csv')
O = O[(O.inliers >= 30) & ((O.scale - 1).abs() < 0.05) & (O.rot_deg.abs() < 5)]
src_tr = G[G.split_source == 'train'].groupby('src').agg(tiles=('stem', 'size'), n0=('n0', 'sum'), n1=('n1', 'sum'), n2=('n2', 'sum'))
st = G[G.split_stretch == 'train'][['n0', 'n1', 'n2']].sum(); target = int((G.split_stretch == 'train').sum())
p_st = (st / st.sum()).values
rng = np.random.default_rng(2026); best = None
for k in range(2000):
    order = rng.permutation(src_tr.index.values); keep = list(order); tiles = src_tr.tiles.sum()
    while tiles - src_tr.tiles[keep[-1]] >= target - 15:
        tiles -= src_tr.tiles[keep[-1]]; keep.pop()
    if abs(tiles - target) > 15: continue
    c = src_tr.loc[keep, ['n0', 'n1', 'n2']].sum().values; d = float(np.abs(c / c.sum() - p_st).sum())
    if best is None or d < best[0]: best = (d, sorted(keep), int(tiles), c.tolist())
d, keep, tiles, c = best
stems = G[G.src.isin(keep)].stem
with open('data/splits/sourcematched_train.txt', 'w') as f:
    f.write('\n'.join(os.path.abspath(f'data/HighRPD/images/{s}.jpg') for s in stems) + '\n')
with open('data/splits/highrpd_sourcematched.yaml', 'w') as f:
    f.write(f'names:\n  0: line\n  1: block\n  2: pit\npath: {os.path.abspath("data/splits")}\ntest: source_test.txt\ntrain: sourcematched_train.txt\nval: source_val.txt\n')
# common contrast frames that still overlap a frame of the matched training subset
C = G[(G.split_source == 'test') & (G.split_stretch.isin(['val', 'test']))].src.unique()
pairs = set(zip(O.src_a, O.src_b)) | set(zip(O.src_b, O.src_a)); K = set(keep)
ov = [c_ for c_ in C if any((c_, k) in pairs for k in K)]
ov_full = [c_ for c_ in C if any((c_, k) in pairs for k in src_tr.index)]
out = dict(target_tiles=target, tiles=tiles, frames=len(keep), frames_full=len(src_tr), class_counts=c,
           class_share=(np.array(c) / sum(c)).round(4).tolist(), stretch_class_share=p_st.round(4).tolist(), l1_distance=d,
           common_frames=len(C), common_frames_overlapping_matched_train=len(ov), common_frames_overlapping_full_train=len(ov_full))
json.dump(out, open('results/revision/matched_split.json', 'w'), indent=1); print(out)
