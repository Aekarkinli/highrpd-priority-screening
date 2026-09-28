"""Prepare HighRPD for the pipeline.
1. Clip the few annotation boxes whose coordinates exceed the tile boundary (55 label files in the release) to [0, 1].
2. Build the tile index results/tiles_index.csv from the file names: the substring before '_640_' is the source
   identifier (source group), the two trailing integers are the row and column of the tile within the frame, and
   n, n0, n1, n2 are the numbers of boxes (all, line, block, pit) in the label file.
Run from the repository root after extracting HighRPD to data/HighRPD/{images,labels}."""
import os, re, glob, pandas as pd
IMG, LAB = 'data/HighRPD/images', 'data/HighRPD/labels'
os.makedirs('results', exist_ok=True)
clipped = []
for p in sorted(glob.glob(f'{LAB}/*.txt')):
    out, bad = [], False
    for line in open(p):
        q = line.split()
        if len(q) < 5: continue
        c = int(q[0]); x, y, w, h = map(float, q[1:5])
        x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        if min(x1, y1) < -1e-6 or max(x2, y2) > 1 + 1e-6:   # tolerance above the 6-decimal precision of the label files
            bad = True; x1, y1, x2, y2 = max(0, x1), max(0, y1), min(1, x2), min(1, y2)
        out.append(f'{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}')
    if bad:
        clipped.append(os.path.basename(p)); open(p, 'w').write('\n'.join(out) + '\n')
print(f'{len(clipped)} label files clipped to the tile boundary')
open('results/clipped_label_files.txt', 'w').write('\n'.join(clipped) + '\n')
pat = re.compile(r'^(.*)_640_(\d+)_(\d+)$'); rows = []
for f in sorted(os.listdir(IMG)):
    stem = os.path.splitext(f)[0]; m = pat.match(stem)
    if not m: raise ValueError(f'unexpected file name {f}')
    lp = f'{LAB}/{stem}.txt'
    cls = [int(l.split()[0]) for l in open(lp) if l.strip()] if os.path.exists(lp) else []
    rows.append(dict(stem=stem, src=m.group(1), a=int(m.group(2)), b=int(m.group(3)), n=len(cls), n0=cls.count(0), n1=cls.count(1), n2=cls.count(2)))
df = pd.DataFrame(rows); df.to_csv('results/tiles_index.csv', index=False)
print(f'{len(df)} tiles, {df.src.nunique()} source groups, boxes: line {df.n0.sum()}, block {df.n1.sum()}, pit {df.n2.sum()}')
