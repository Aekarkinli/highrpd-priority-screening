"""Exact decomposition of the mean squared error of the detection-based score over the seeds of one configuration.
For errors e[s, i] of seed s on unit i, with b the mean error, e_i the seed-mean error of unit i:
    MSE = b^2 + B^2 + T^2,  B^2 = mean_i (e_i - b)^2,  T^2 = mean_{s,i} (e[s, i] - e_i)^2.
B is the spread of the seed-averaged error between units, T the variation between training runs around it.
The seed-averaged score has error e_i, so sqrt(b^2 + B^2) is the RMSE of the score averaged over the three runs.
Output: results/revision/decomposition_exact.json"""
import sys, json, numpy as np, pandas as pd
from importlib import import_module
sys.path.insert(0, 'code'); from inventory import *
RA = import_module('18_revision_analyses')
GT = pd.read_csv('results/tiles_gt_inventory.csv'); d = GT[GT.split_stretch == 'test'].reset_index(drop=True)
stems = list(d.stem); src = d.src.values
Fg = RA.features({s: [b + (1.0,) for b in read_yolo_txt(f'data/HighRPD/labels/{s}.txt')] for s in stems}, stems); _, g = RA.score(Fg)
Gref = RA.group_table(Fg, g, stems, src)


def parts(D, Rv, empty=None):
    E = D - Rv; m = E.shape[0]; b = E.mean(); ei = E.mean(0)
    B2 = np.mean((ei - b) ** 2); T2 = np.mean((E - ei) ** 2); mse = np.mean(E ** 2)
    x = dict(bias=float(b), rmse_single_run=float(np.sqrt(mse)), B=float(np.sqrt(B2)), T=float(np.sqrt(T2)),
             share_T=float(T2 / (B2 + T2)), share_B=float(B2 / (B2 + T2)), check=float(mse - b ** 2 - B2 - T2),
             rmse_seed_mean=float(np.sqrt(b ** 2 + B2)),
             B_minus_noise=float(np.sqrt(max(B2 - T2 / (m - 1), 0))))          # B^2 corrected for the run noise left in a 3-run mean
    if empty is not None: x['sq_error_share_empty_tiles'] = float(np.sum((E ** 2) * empty) / np.sum(E ** 2))
    return x


out = {}
for tag in ['y11s_stretch', 'y8s_stretch']:
    rv = json.load(open(f'results/revision/{tag}/revision.json')); T = rv['t_sel']; runs = rv['runs']
    P = {r: json.load(open(f'results/runs/{r}/pred_highrpd_conf005.json')) for r in runs}
    Dt, Dg, Em = [], [], []
    for r in runs:
        Fp = RA.features(P[r], stems, T); _, p = RA.score(Fp)
        Dt.append(p); Em.append((Fp.n_boxes == 0).values); Dg.append(RA.group_table(Fp, p, stems, src).loc[Gref.index].S_VMPS.values)
    out[tag] = dict(tile=parts(np.array(Dt), g, np.array(Em)), frame=parts(np.array(Dg), Gref.S_VMPS.values))
    print(tag, json.dumps(out[tag], indent=0))
json.dump(out, open('results/revision/decomposition_exact.json', 'w'), indent=1)
