"""Create random (tile-level) and source-separated splits for HighRPD.
Outputs list files under data/splits and YAML files for Ultralytics."""
import pandas as pd, numpy as np, os, yaml
ROOT=os.path.abspath('data/HighRPD'); OUT='data/splits'; os.makedirs(OUT,exist_ok=True)
df=pd.read_csv('results/tiles_index.csv')
df['has_pit']=(df.n2>0).astype(int)
tot=df[['n0','n1','n2']].sum().values
def props(sub): return sub[['n0','n1','n2']].sum().values/ tot
# ---- random tile-level split 70/15/15
rng=np.random.default_rng(42); idx=rng.permutation(len(df)); n=len(df)
r_tr,r_va,r_te=idx[:int(.7*n)],idx[int(.7*n):int(.85*n)],idx[int(.85*n):]
df['split_random']='train'; df.loc[r_va,'split_random']='val'; df.loc[r_te,'split_random']='test'
# ---- source-separated split: choose seed giving class proportions closest to 70/15/15 by instances
srcs=df.src.unique(); best=None
for seed in range(200):
    rng=np.random.default_rng(seed); p=rng.permutation(srcs); m=len(p)
    s_tr,s_va,s_te=set(p[:int(.7*m)]),set(p[int(.7*m):int(.85*m)]),set(p[int(.85*m):])
    lab=df.src.map(lambda s:'train' if s in s_tr else ('val' if s in s_va else 'test'))
    dev=0
    for name,target in [('train',.7),('val',.15),('test',.15)]:
        sub=df[lab==name]; dev+=np.abs(props(sub)-target).sum()+abs(len(sub)/n-target)
    if best is None or dev<best[0]: best=(dev,seed,lab)
df['split_source']=best[2].values; print('source split seed',best[1],'dev',round(best[0],4))
for col in ['split_random','split_source']:
    print(col); print(df.groupby(col).agg(tiles=('stem','size'),sources=('src','nunique'),line=('n0','sum'),block=('n1','sum'),pit=('n2','sum'),pit_tiles=('has_pit','sum')))
df.to_csv('results/tiles_index.csv',index=False)
# partition composition table and Supplementary Table S1 (source groups with tile counts and subset assignment)
rows=[]
for col,name in [('split_random','Random'),('split_source','Source-separated')]:
    trs=set(df[df[col]=='train'].src)
    for part in ['train','val','test']:
        s=df[df[col]==part]; rows.append(dict(partition=name,subset=part,tiles=len(s),frames=s.src.nunique(),line=int(s.n0.sum()),block=int(s.n1.sum()),pit=int(s.n2.sum()),pit_tiles=int((s.n2>0).sum()),frames_shared_with_train=(s.src.isin(trs).mean() if part!='train' else float('nan'))))
pd.DataFrame(rows).to_csv('results/table_splits.csv',index=False)
g=df.groupby('src').agg(tiles=('stem','size'),line=('n0','sum'),block=('n1','sum'),pit=('n2','sum'),rows=('a','max'),cols=('b','max'),split_random_tiles_train=('split_random',lambda x:(x=='train').sum()),split_source=('split_source','first')).reset_index()
g['naming_pattern']=g.src.str.replace(r'\d','#',regex=True); g.to_csv('results/supplementary_table_S1_source_groups.csv',index=False)
for col,tag in [('split_random','random'),('split_source','source')]:
    for part in ['train','val','test']:
        stems=df.loc[df[col]==part,'stem']
        with open(f'{OUT}/{tag}_{part}.txt','w') as f:
            for s in stems: f.write(f'{ROOT}/images/{s}.jpg\n')
    # pit-oversampled training list (pit tiles repeated 3x) for source split
    if tag=='source':
        tr=df[df[col]=='train']
        with open(f'{OUT}/source_train_pitx3.txt','w') as f:
            for r in tr.itertuples():
                rep=3 if r.n2>0 else 1
                for _ in range(rep): f.write(f'{ROOT}/images/{r.stem}.jpg\n')
    names={0:'line',1:'block',2:'pit'}
    y=dict(path=os.path.abspath(OUT),train=f'{tag}_train.txt',val=f'{tag}_val.txt',test=f'{tag}_test.txt',names=names)
    yaml.safe_dump(y,open(f'{OUT}/highrpd_{tag}.yaml','w'))
    if tag=='source':
        y['train']='source_train_pitx3.txt'; yaml.safe_dump(y,open(f'{OUT}/highrpd_source_pitx3.yaml','w'))
print('done')
