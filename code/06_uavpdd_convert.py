"""Convert UAV-PDD2023 (PASCAL VOC) to YOLO format with HighRPD class mapping.
Only the 'lr' variant of each view is kept (lr, tb and r180 are flipped copies of the same view)."""
import glob, os, xml.etree.ElementTree as ET, pandas as pd, numpy as np, yaml
SRC='data/external'; OUT='data/uavpdd_yolo'; os.makedirs(f'{OUT}/images',exist_ok=True); os.makedirs(f'{OUT}/labels',exist_ok=True)
MAP={'Longitudinal crack':0,'Transverse crack':0,'Oblique crack':0,'Alligator crack':1,'Pothole':2,'Repair':None}
rows=[]
for xml in sorted(glob.glob(f'{SRC}/Annotations/lr_*.xml')):
    stem=os.path.splitext(os.path.basename(xml))[0]; r=ET.parse(xml).getroot()
    W=int(r.find('size/width').text); H=int(r.find('size/height').text)
    lines=[]; cnt=dict(LC=0,TC=0,OC=0,AC=0,PH=0,RP=0)
    for o in r.findall('object'):
        n=o.find('name').text; cnt[{'Longitudinal crack':'LC','Transverse crack':'TC','Oblique crack':'OC','Alligator crack':'AC','Pothole':'PH','Repair':'RP'}[n]]+=1
        c=MAP[n]
        if c is None: continue
        b=o.find('bndbox'); x1,y1,x2,y2=[float(b.find(k).text) for k in ['xmin','ymin','xmax','ymax']]
        x1,y1,x2,y2=max(0,x1),max(0,y1),min(W,x2),min(H,y2)
        lines.append(f'{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}')
    open(f'{OUT}/labels/{stem}.txt','w').write('\n'.join(lines)+('\n' if lines else ''))
    src=f'{SRC}/JPEGImages/{stem}.jpg'; dst=f'{OUT}/images/{stem}.jpg'
    if not os.path.exists(dst): os.link(src,dst) if os.stat(src).st_dev==os.stat(OUT).st_dev else __import__('shutil').copy(src,dst)
    view='_'.join(stem.split('_')[1:2]); quad='_'.join(stem.split('_')[2:])
    rows.append(dict(stem=stem,view=view,quadrant=quad,W=W,H=H,**cnt,n_mapped=len(lines)))
df=pd.DataFrame(rows); df.to_csv('results/uavpdd_index.csv',index=False)
print(len(df),'images;',df.view.nunique(),'source views'); print(df[['LC','TC','OC','AC','PH','RP']].sum())
with open(f'{OUT}/all.txt','w') as f:
    for s in df.stem: f.write(os.path.abspath(f'{OUT}/images/{s}.jpg')+'\n')
yaml.safe_dump(dict(path=os.path.abspath(OUT),train='all.txt',val='all.txt',test='all.txt',names={0:'line',1:'block',2:'pit'}),open(f'{OUT}/uavpdd.yaml','w'))
