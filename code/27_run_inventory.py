"""Inventory of all training runs: data, initial weights, seed, epoch limit, epochs run, best epoch, optimiser chosen by
the automatic selection, training time and purpose.
Output: results/run_inventory.csv"""
import os, re, json, yaml, pandas as pd

PURPOSE = [(r'^y(11|8)s_(random|source|stretch)_s\d+$', 'main comparison of partitions'),
           (r'^y11s_stretchpit_', 'pit oversampling (stretch-separated)'), (r'^y11s_pit_', 'pit oversampling (frame-separated, earlier analysis)'),
           (r'^y11s_stretch_e150_', '150-epoch schedule'), (r'^y11s_sourcematched_', 'size- and class-matched frame-separated control'),
           (r'^ft_uav_', 'fine-tuning on UAV-PDD2023')]
rows = []
for r in sorted(os.listdir('runs')):
    m = f'runs/{r}/meta.json'
    if not os.path.exists(m): continue
    meta = json.load(open(m)); a = yaml.safe_load(open(f'runs/{r}/args.yaml'))
    res = pd.read_csv(f'runs/{r}/results.csv'); res.columns = [c.strip() for c in res.columns]
    res = res.iloc[res.index[res.epoch == 1].max():].reset_index(drop=True)          # keep the last complete run after an aborted start
    fit = 0.1 * res['metrics/mAP50(B)'] + 0.9 * res['metrics/mAP50-95(B)']
    opt = ''
    if os.path.exists(f'runs/{r}.log'):
        txt = open(f'runs/{r}.log', encoding='utf8', errors='ignore').read(); o = re.search(r'optimizer:\S*\s+(\w+)\(lr=([\d.e-]+)', txt)
        if o: opt = f'{o.group(1)} lr={o.group(2)}'
    rows.append(dict(run=r, purpose=next((p for k, p in PURPOSE if re.search(k, r)), ''), model=os.path.basename(str(meta.get('model'))), init=str(a.get('model')),
                     data=os.path.basename(str(a.get('data'))), seed=a.get('seed'), epochs_limit=a.get('epochs'), patience=a.get('patience'), batch=a.get('batch'),
                     epochs_run=len(res), best_epoch=int(res.epoch.iloc[fit.idxmax()]), optimizer=opt, train_time_h=round(meta['train_time_s'] / 3600, 2) if meta.get('train_time_s') else None))
D = pd.DataFrame(rows); D.to_csv('results/run_inventory.csv', index=False); print(D.to_string()); print(len(D), 'runs')
