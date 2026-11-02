"""Overlap audit of the view-separated UAV-PDD2023 partition used for fine-tuning.
Every test image is registered against every training and validation image (images at 1/4 scale, SIFT, ratio test,
RANSAC similarity transform with any rotation). A pair is flagged when at least 30 inliers support a transform with a
scale between 0.8 and 1.25, which admits small changes of flight height between views.
Outputs: results/uavpdd_view_audit_pairs.csv, results/revision/uavpdd_view_audit.json"""
import os, json, time, numpy as np, pandas as pd, cv2

U = pd.read_csv('results/uavpdd_tiles_index.csv').groupby('image').agg(view=('view', 'first'), split=('split', 'first'))
sift = cv2.SIFT_create(nfeatures=4000)


def feats(img):
    im = cv2.imread(f'data/external/JPEGImages/{img}.jpg', cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    kp, des = sift.detectAndCompute(im, None)
    return np.float32([k.pt for k in kp]), (des if des is not None else np.zeros((0, 128), np.float32))


t0 = time.time(); D = {i: feats(i) for i in U.index}; print('features', round(time.time() - t0), flush=True)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))
test = list(U[U.split == 'test'].index); other = list(U[U.split != 'test'].index); rows = []
for n, a in enumerate(test):
    pa, da = D[a]
    for b in other:
        pb, db = D[b]; r = dict(img_a=a, img_b=b, view_a=U.view[a], view_b=U.view[b], split_b=U.split[b], inliers=0)
        if len(da) > 20 and len(db) > 20:
            m = [x for x, y in (p for p in flann.knnMatch(da, db, k=2) if len(p) == 2) if x.distance < 0.75 * y.distance]
            if len(m) >= 12:
                M, inl = cv2.estimateAffinePartial2D(pa[[x.queryIdx for x in m]], pb[[x.trainIdx for x in m]], method=cv2.RANSAC, ransacReprojThreshold=4.0)
                if M is not None:
                    r.update(inliers=int(inl.sum()), scale=float(np.hypot(M[0, 0], M[1, 0])), rot_deg=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
        rows.append(r)
    if n % 10 == 0: print(n, len(test), round(time.time() - t0), flush=True)
P = pd.DataFrame(rows); P.to_csv('results/uavpdd_view_audit_pairs.csv', index=False)
F = P[(P.inliers >= 30) & P.scale.between(0.8, 1.25)]
out = dict(test_images=len(test), other_images=len(other), pairs=len(P), flagged_pairs=len(F),
           flagged_test_views=int(F.view_a.nunique()), test_views=int(U[U.split == 'test'].view.nunique()),
           flagged_test_images=int(F.img_a.nunique()), max_inliers=int(P.inliers.max()), inliers_p999=float(P.inliers.quantile(.999)),
           flagged=F.sort_values('inliers', ascending=False).head(50).to_dict('records'))
json.dump(out, open('results/revision/uavpdd_view_audit.json', 'w'), indent=1, default=str)
print({k: v for k, v in out.items() if k != 'flagged'}); print(F.sort_values('inliers', ascending=False).head(20))
