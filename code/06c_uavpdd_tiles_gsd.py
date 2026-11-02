"""Resolution-matched tiles of UAV-PDD2023: each 2592x1944 'lr' quadrant (about 3 mm per pixel) is resampled by
3.0/6.3 to the ground sampling distance of HighRPD (about 6.3 mm per pixel, 1234x926 pixels) and cut into a 2x2 grid of
640x640 tiles, padded with black where the quadrant ends, as in the HighRPD tiles at frame borders.
Boxes are rescaled and clipped; clipped parts smaller than 4 pixels are discarded.
Outputs: data/uavpdd_tiles_gsd/{images,labels}, all.txt, uavpdd_tiles_gsd_all.yaml, results/uavpdd_tiles_gsd_index.csv"""
import os, sys, pandas as pd, yaml
from PIL import Image
sys.path.insert(0, 'code'); from inventory import read_yolo_txt
SRC = 'data/uavpdd_yolo'; OUT = 'data/uavpdd_tiles_gsd'; F = 3.0 / 6.3; S = 640
os.makedirs(f'{OUT}/images', exist_ok=True); os.makedirs(f'{OUT}/labels', exist_ok=True)
idx = pd.read_csv('results/uavpdd_index.csv'); rows = []
for r in idx.itertuples():
    im = Image.open(f'{SRC}/images/{r.stem}.jpg').convert('RGB'); W, H = im.size
    w2, h2 = round(W * F), round(H * F); im = im.resize((w2, h2), Image.LANCZOS)
    px = [(c, (x - w / 2) * w2, (y - h / 2) * h2, (x + w / 2) * w2, (y + h / 2) * h2) for c, x, y, w, h in read_yolo_txt(f'{SRC}/labels/{r.stem}.txt')]
    for i in range(2):
        for j in range(2):
            x0, y0 = j * S, i * S; stem = f'{r.stem}_g{i}{j}'
            if not os.path.exists(f'{OUT}/images/{stem}.jpg'):
                t = Image.new('RGB', (S, S)); t.paste(im.crop((x0, y0, min(x0 + S, w2), min(y0 + S, h2))), (0, 0)); t.save(f'{OUT}/images/{stem}.jpg', quality=95)
            lines = []
            for c, x1, y1, x2, y2 in px:
                a, b, cc, d = max(x1, x0), max(y1, y0), min(x2, x0 + S, w2), min(y2, y0 + S, h2)
                if cc - a >= 4 and d - b >= 4: lines.append(f'{c} {((a + cc) / 2 - x0) / S:.6f} {((b + d) / 2 - y0) / S:.6f} {(cc - a) / S:.6f} {(d - b) / S:.6f}')
            open(f'{OUT}/labels/{stem}.txt', 'w').write('\n'.join(lines) + ('\n' if lines else ''))
            valid = (min(x0 + S, w2) - x0) * (min(y0 + S, h2) - y0) / S ** 2
            rows.append(dict(stem=stem, image=r.stem, view=r.view, quadrant=r.quadrant, row=i, col=j, valid_fraction=valid, n=len(lines),
                             n0=sum(l[0] == '0' for l in lines), n1=sum(l[0] == '1' for l in lines), n2=sum(l[0] == '2' for l in lines)))
df = pd.DataFrame(rows); df.to_csv('results/uavpdd_tiles_gsd_index.csv', index=False)
print(len(df), 'tiles', df.view.nunique(), 'views | empty tiles', (df.n == 0).sum(), '| boxes', df[['n0', 'n1', 'n2']].sum().to_dict())
open(f'{OUT}/all.txt', 'w').write(''.join(os.path.abspath(f'{OUT}/images/{s}.jpg') + '\n' for s in df.stem))
yaml.safe_dump(dict(path=os.path.abspath(OUT), train='all.txt', val='all.txt', test='all.txt', names={0: 'line', 1: 'block', 2: 'pit'}), open(f'{OUT}/uavpdd_tiles_gsd_all.yaml', 'w'))
