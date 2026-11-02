"""Ground overlap between source groups captured in sequence.
Each source group is reassembled from its tiles (row a, column b) into a frame mosaic at 1/4 scale. Consecutive groups
of the same capture series (index gap 1 to 6) are registered with SIFT features and a RANSAC similarity transform.
For every registered pair the script reports the number of inlier matches, the shift and the share of tiles of one group
whose centres fall on a tile of the other group (shared pavement in the released tiles).
Outputs: results/frame_overlap_pairs.csv"""
import re, os, numpy as np, pandas as pd, cv2

S = 4                     # downscale factor of the mosaic
TP = 640 // S             # tile size in mosaic pixels
T = pd.read_csv('results/tiles_index.csv') if os.path.exists('results/tiles_index.csv') else None
G = pd.read_csv('results/tiles_gt_inventory.csv')[['stem', 'src', 'a', 'b', 'split_source']]


def parse(s):
    m = re.match(r'(DJI|dji)_(\d{14})_(\d{4})', s)
    if m: return m.group(1), int(m.group(3))
    m = re.match(r'1x(\d+)', s); return '1x', int(m.group(1))


def mosaic(src):
    d = G[G.src == src]
    H, W = (d.a.max() + 1) * TP, (d.b.max() + 1) * TP
    img = np.zeros((H, W), np.uint8); mask = np.zeros((H, W), np.uint8)
    for st, a, b in zip(d.stem, d.a, d.b):
        im = cv2.imread(f'data/HighRPD/images/{st}.jpg', cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (TP, TP), interpolation=cv2.INTER_AREA)
        img[a * TP:(a + 1) * TP, b * TP:(b + 1) * TP] = im; mask[a * TP + 4:(a + 1) * TP - 4, b * TP + 4:(b + 1) * TP - 4] = 255
    return img, mask, d


sift = cv2.SIFT_create(nfeatures=6000)
cache = {}


def feats(src):
    if src not in cache:
        img, mask, d = mosaic(src); kp, des = sift.detectAndCompute(img, mask); cache[src] = (kp, des, d)
        if len(cache) > 12: cache.pop(next(iter(cache)))
    return cache[src]


def main():
    srcs = G.src.drop_duplicates().tolist()
    meta = pd.DataFrame([(s, *parse(s)) for s in srcs], columns=['src', 'series', 'idx']).sort_values(['series', 'idx'])
    split = G.groupby('src').split_source.first()
    bf = cv2.BFMatcher(); rows = []
    for ser, d in meta.groupby('series'):
        d = d.reset_index(drop=True)
        for i in range(len(d)):
            for j in range(i + 1, min(i + 7, len(d))):
                a, b = d.src[i], d.src[j]
                if d.idx[j] - d.idx[i] > 6: continue
                ka, da, ta = feats(a); kb, db, tb = feats(b)
                r = dict(src_a=a, src_b=b, series=ser, idx_gap=int(d.idx[j] - d.idx[i]), split_a=split[a], split_b=split[b], inliers=0)
                if da is not None and db is not None and len(ka) > 20 and len(kb) > 20:
                    m = [x for x, y in bf.knnMatch(da, db, k=2) if x.distance < 0.75 * y.distance]
                    if len(m) >= 12:
                        pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
                        M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4.0)
                        if M is not None:
                            n_in = int(inl.sum()); r.update(inliers=n_in, dx=float(M[0, 2]) * S, dy=float(M[1, 2]) * S,
                                                            scale=float(np.hypot(M[0, 0], M[1, 0])), rot_deg=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
                            # share of tiles of a whose centre maps onto an existing tile of b
                            ctr = np.float32([[(bb + .5) * TP, (aa + .5) * TP] for aa, bb in zip(ta.a, ta.b)])
                            mp = ctr @ M[:, :2].T + M[:, 2]
                            have = set(zip(tb.a, tb.b))
                            hit = [(int(y // TP), int(x // TP)) in have for x, y in mp]
                            r['shared_tiles_a'] = float(np.mean(hit)); r['n_tiles_a'] = len(ta)
                rows.append(r)
        print(ser, 'pairs', len(rows), flush=True)
    out = pd.DataFrame(rows); out.to_csv('results/frame_overlap_pairs.csv', index=False)
    ok = out[(out.inliers >= 30) & ((out.scale - 1).abs() < 0.05) & (out.rot_deg.abs() < 5)]
    print('pairs', len(out), 'registered (>=30 inliers, |scale-1|<0.05, |rotation|<5 deg)', len(ok))
    print(ok.groupby('idx_gap')[['inliers', 'shared_tiles_a']].median())
    cross = ok[(ok.split_a != ok.split_b) & ((ok.split_a == 'test') | (ok.split_b == 'test')) & ((ok.split_a == 'train') | (ok.split_b == 'train'))]
    print('registered train-test pairs', len(cross), 'median shared share', cross.shared_tiles_a.median())


if __name__ == '__main__':
    main()
