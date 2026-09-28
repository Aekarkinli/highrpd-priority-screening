"""Pixel-level reference for bounding-box proxies using SAM box prompts (pseudo-reference, not ground truth)."""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0,'code'); from inventory import read_yolo_txt
from ultralytics import SAM
from skimage.morphology import skeletonize
from scipy.ndimage import rotate
from PIL import Image
rng=np.random.default_rng(7)
df=pd.read_csv('results/tiles_gt_inventory.csv')
dom=df[['N_line','N_block','N_pit']].idxmax(axis=1)
sel=pd.concat([df[(dom=='N_line')&(df.N_line<=6)].sample(60,random_state=1),
               df[(dom=='N_block')&(df.N_block<=6)].sample(45,random_state=2),
               df[(df.N_pit>0)&(df.n<=8)].sample(45,random_state=3)]).drop_duplicates('stem')
sel[['stem']].to_csv('results/sam_subset.csv',index=False); print('tiles',len(sel))
model=SAM('sam_b.pt'); S=640
inst=[]; masks={}
def main():
    for r in sel.itertuples():
        img=np.asarray(Image.open(f'data/HighRPD/images/{r.stem}.jpg').convert('RGB'))
        boxes=read_yolo_txt(f'data/HighRPD/labels/{r.stem}.txt')
        px=[[ (x-w/2)*S,(y-h/2)*S,(x+w/2)*S,(y+h/2)*S ] for (_,x,y,w,h) in boxes]
        res=model(img,bboxes=px,verbose=False)[0]
        m=res.masks.data.cpu().numpy().astype(bool) if res.masks is not None else np.zeros((len(px),S,S),bool)
        for j,(b,bx) in enumerate(zip(boxes,px)):
            mk=m[j].copy(); x1,y1,x2,y2=[int(round(v)) for v in bx]
            crop=np.zeros_like(mk); crop[max(0,y1):y2,max(0,x1):x2]=True; mk&=crop   # restrict mask to its prompt box
            area=mk.sum()/(S*S); w,h=b[3],b[4]
            d=dict(stem=r.stem,inst=j,cls=b[0],w=w,h=h,bbox_area=w*h,bbox_diag=np.hypot(w,h)/np.sqrt(2),mask_area=area,
                   mask_fill=area/(w*h) if w*h>0 else np.nan)
            if b[0]==0 and mk.sum()>0:
                sk=skeletonize(mk); d['skel_len']=sk.sum()/(S*np.sqrt(2))
                # orientation sensitivity: rotate the mask, recompute axis-aligned bbox
                ys,xs=np.where(mk); sub=mk[ys.min():ys.max()+1,xs.min():xs.max()+1]
                pad=int(np.hypot(*sub.shape))+2; can=np.zeros((2*pad,2*pad),bool); can[pad-sub.shape[0]//2:pad-sub.shape[0]//2+sub.shape[0],pad-sub.shape[1]//2:pad-sub.shape[1]//2+sub.shape[1]]=sub
                areas=[];diags=[]
                for th in range(0,91,10):
                    rm=rotate(can,th,order=0,reshape=False)>0; yy,xx=np.where(rm)
                    ww,hh=(xx.max()-xx.min()+1)/S,(yy.max()-yy.min()+1)/S; areas.append(ww*hh); diags.append(np.hypot(ww,hh)/np.sqrt(2))
                d['rot_area_cv']=np.std(areas)/np.mean(areas); d['rot_diag_cv']=np.std(diags)/np.mean(diags)
                d['rot_area_maxmin']=max(areas)/min(areas); d['rot_diag_maxmin']=max(diags)/min(diags)
            inst.append(d)
        masks[r.stem]=np.packbits(m,axis=None); masks[r.stem+'_shape']=np.array(m.shape)
    I=pd.DataFrame(inst); I.to_csv('results/sam_instances.csv',index=False); np.savez_compressed('results/sam_masks.npz',**masks)
    from scipy.stats import spearmanr
    out=[]
    for c,name in [(0,'line'),(1,'block'),(2,'pit')]:
        s=I[(I.cls==c)&(I.mask_area>0)]
        row=dict(cls=name,n=len(s),rho_area=spearmanr(s.bbox_area,s.mask_area).correlation,rho_diag=spearmanr(s.bbox_diag,s.mask_area).correlation,fill_median=s.mask_fill.median(),fill_iqr=(s.mask_fill.quantile(.75)-s.mask_fill.quantile(.25)))
        if c==0: row.update(rho_area_skel=spearmanr(s.bbox_area,s.skel_len).correlation,rho_diag_skel=spearmanr(s.bbox_diag,s.skel_len).correlation,rot_area_cv=s.rot_area_cv.median(),rot_diag_cv=s.rot_diag_cv.median(),rot_area_maxmin=s.rot_area_maxmin.median(),rot_diag_maxmin=s.rot_diag_maxmin.median())
        out.append(row)
    # tile level
    T=I.groupby(['stem','cls']).agg(bbox_area=('bbox_area','sum'),bbox_diag=('bbox_diag','sum'),mask_area=('mask_area','sum'),skel=('skel_len','sum')).reset_index()
    for c,name in [(0,'line'),(1,'block'),(2,'pit')]:
        s=T[T.cls==c]; out.append(dict(cls=name+'_tile',n=len(s),rho_area=spearmanr(s.bbox_area,s.mask_area).correlation,rho_diag=spearmanr(s.bbox_diag,s.mask_area).correlation,rho_area_skel=spearmanr(s.bbox_area,s.skel).correlation if c==0 else np.nan,rho_diag_skel=spearmanr(s.bbox_diag,s.skel).correlation if c==0 else np.nan))
    O=pd.DataFrame(out); O.to_csv('results/sam_proxy_summary.csv',index=False); print(O.round(3).to_string())
if __name__=='__main__': main()
