"""Exhaustive overlap audit of the stretch-separated partition.
Every pair of retained frames that lie in different subsets (train-val, train-test, val-test) is registered with the
procedure of code/16_frame_overlap.py (frame mosaics at 1/4 scale, SIFT, ratio test, RANSAC similarity transform),
irrespective of the distance in the capture sequence and without restricting the rotation, so that revisits of the same
pavement by another pass or in the opposite direction would also be found. A pair is flagged when at least 30 inliers
support a transform with a scale within 5 % of one; the acceptance rule of the main analysis (rotation below 5 degrees)
is reported as a subset. Inliers are also recorded for every pair so that the margin below the threshold is visible.
Outputs: results/overlap_audit_pairs.csv, results/revision/overlap_audit.json"""
import os, sys, json, time, numpy as np, pandas as pd, cv2
from importlib import import_module
sys.path.insert(0, 'code'); FO = import_module('16_frame_overlap')

G = FO.G.merge(pd.read_csv('results/tiles_gt_inventory.csv')[['stem', 'split_stretch']], on='stem')
lab = G.groupby('src').split_stretch.first(); lab = lab[lab != 'drop']
CACHE = 'data/sift_cache'; os.makedirs(CACHE, exist_ok=True)


def desc(src):
    p = f'{CACHE}/{src}.npz'
    if os.path.exists(p):
        z = np.load(p); return z['pt'], z['des']
    kp, des, _ = FO.feats(src); pt = np.float32([k.pt for k in kp])
    des = des if des is not None else np.zeros((0, 128), np.float32)
    np.savez(p, pt=pt, des=des); return pt, des


def main():
    srcs = list(lab.index); D = {}
    t0 = time.time()
    for i, s in enumerate(srcs):
        D[s] = desc(s)
        if i % 50 == 0: print('features', i, len(srcs), round(time.time() - t0), flush=True)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
    pairs = [(a, b) for i, a in enumerate(srcs) for b in srcs[i + 1:] if lab[a] != lab[b]]
    print('pairs', len(pairs), flush=True); rows = []
    for n, (a, b) in enumerate(pairs):
        (pa, da), (pb, db) = D[a], D[b]; r = dict(src_a=a, src_b=b, split_a=lab[a], split_b=lab[b], matches=0, inliers=0)
        if len(da) > 20 and len(db) > 20:
            m = [x for x, y in (p for p in flann.knnMatch(da, db, k=2) if len(p) == 2) if x.distance < 0.75 * y.distance]
            r['matches'] = len(m)
            if len(m) >= 12:
                A = pa[[x.queryIdx for x in m]]; B = pb[[x.trainIdx for x in m]]
                M, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=4.0)
                if M is not None:
                    r.update(inliers=int(inl.sum()), scale=float(np.hypot(M[0, 0], M[1, 0])), rot_deg=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
        rows.append(r)
        if n % 2000 == 0: print(n, len(pairs), round(time.time() - t0), flush=True)
    P = pd.DataFrame(rows); P.to_csv('results/overlap_audit_pairs.csv', index=False)
    flag = P[(P.inliers >= 30) & ((P.scale - 1).abs() < 0.05)]
    main_rule = flag[flag.rot_deg.abs() < 5]
    out = dict(frames=len(srcs), pairs=len(P), flagged_any_rotation=len(flag), flagged_main_rule=len(main_rule),
               max_inliers=int(P.inliers.max()), inliers_p99=float(P.inliers.quantile(.99)), inliers_p999=float(P.inliers.quantile(.999)),
               pairs_ge_12_inliers=int((P.inliers >= 12).sum()), by_subset=P.groupby(['split_a', 'split_b']).size().astype(int).to_dict().__repr__(),
               flagged=flag.to_dict('records'))
    json.dump(out, open('results/revision/overlap_audit.json', 'w'), indent=1, default=str)
    print({k: v for k, v in out.items() if k != 'flagged'}); print(flag)


if __name__ == '__main__':
    main()
