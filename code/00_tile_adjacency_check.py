"""Check that horizontally adjacent HighRPD tiles share an edge without overlap.
For pairs (row, col) and (row, col+1) of the same source group, the Pearson correlation between the last
pixel column of the left tile and the first column of the right tile is compared with the correlation of
strips of width w that would coincide if the tiles overlapped by w pixels."""
import numpy as np, pandas as pd
from PIL import Image
df = pd.read_csv('results/tiles_index.csv'); I = 'data/HighRPD/images'
load = lambda s: np.asarray(Image.open(f'{I}/{s}.jpg').convert('L'), float)
rows = []; n = 0
for src, sub in df.groupby('src'):
    if len(sub) < 10: continue
    keys = {(r.a, r.b): r.stem for r in sub.itertuples()}
    for (a, b), st in keys.items():
        if (a, b + 1) not in keys: continue
        A, B = load(st), load(keys[(a, b + 1)])
        rec = dict(src=src, row=a, col=b, edge=np.corrcoef(A[:, -1], B[:, 0])[0, 1])
        for w in [64, 128, 192, 256, 320]: rec[f'overlap_{w}'] = np.corrcoef(A[:, 640 - w:].ravel(), B[:, :w].ravel())[0, 1]
        rows.append(rec); n += 1
        if n >= 30: break
    if n >= 30: break
R = pd.DataFrame(rows); R.to_csv('results/tile_adjacency_check.csv', index=False)
print(R.drop(columns=['src', 'row', 'col']).median().round(3))
