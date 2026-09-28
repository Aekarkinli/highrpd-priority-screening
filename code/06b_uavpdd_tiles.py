"""Tile UAV-PDD2023 'lr' images (2592x1944) into 4x3 non-overlapping 640x640 tiles (central 2560x1920 crop)."""
import os, numpy as np, pandas as pd, yaml, sys
from PIL import Image
sys.path.insert(0,'code'); from inventory import read_yolo_txt
SRC='data/uavpdd_yolo'; OUT='data/uavpdd_tiles'; os.makedirs(f'{OUT}/images',exist_ok=True); os.makedirs(f'{OUT}/labels',exist_ok=True)
idx=pd.read_csv('results/uavpdd_index.csv'); S=640; X0,Y0=16,32; rows=[]
for r in idx.itertuples():
    im=Image.open(f'{SRC}/images/{r.stem}.jpg'); W,H=im.size
    boxes=read_yolo_txt(f'{SRC}/labels/{r.stem}.txt')
    px=[(c,(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H) for c,x,y,w,h in boxes]
    for i in range(3):
        for j in range(4):
            x0,y0=X0+j*S,Y0+i*S; stem=f'{r.stem}_{i:02d}_{j:02d}'
            if not os.path.exists(f'{OUT}/images/{stem}.jpg'): im.crop((x0,y0,x0+S,y0+S)).save(f'{OUT}/images/{stem}.jpg',quality=95)
            lines=[]
            for c,x1,y1,x2,y2 in px:
                a,b,cc,d=max(x1,x0),max(y1,y0),min(x2,x0+S),min(y2,y0+S)
                if cc-a>=4 and d-b>=4: lines.append(f'{c} {((a+cc)/2-x0)/S:.6f} {((b+d)/2-y0)/S:.6f} {(cc-a)/S:.6f} {(d-b)/S:.6f}')
            open(f'{OUT}/labels/{stem}.txt','w').write('\n'.join(lines)+('\n' if lines else ''))
            rows.append(dict(stem=stem,image=r.stem,view=r.view,quadrant=r.quadrant,row=i,col=j,n=len(lines),n0=sum(l[0]=='0' for l in lines),n1=sum(l[0]=='1' for l in lines),n2=sum(l[0]=='2' for l in lines)))
df=pd.DataFrame(rows)
# view-separated split for fine-tuning (70/15/15 by view id)
views=sorted(df.view.unique()); rng=np.random.default_rng(0); p=rng.permutation(views); m=len(p)
sp={v:'train' for v in p[:int(.7*m)]}; sp.update({v:'val' for v in p[int(.7*m):int(.85*m)]}); sp.update({v:'test' for v in p[int(.85*m):]})
df['split']=df.view.map(sp); df.to_csv('results/uavpdd_tiles_index.csv',index=False)
print(df.groupby('split').agg(tiles=('stem','size'),views=('view','nunique'),line=('n0','sum'),block=('n1','sum'),pit=('n2','sum'),empty=('n',lambda x:(x==0).sum())))
for part in ['train','val','test','all']:
    sub=df if part=='all' else df[df.split==part]
    open(f'{OUT}/{part}.txt','w').write(''.join(os.path.abspath(f'{OUT}/images/{s}.jpg')+'\n' for s in sub.stem))
yaml.safe_dump(dict(path=os.path.abspath(OUT),train='train.txt',val='val.txt',test='test.txt',names={0:'line',1:'block',2:'pit'}),open(f'{OUT}/uavpdd_tiles.yaml','w'))
yaml.safe_dump(dict(path=os.path.abspath(OUT),train='all.txt',val='all.txt',test='all.txt',names={0:'line',1:'block',2:'pit'}),open(f'{OUT}/uavpdd_tiles_all.yaml','w'))
