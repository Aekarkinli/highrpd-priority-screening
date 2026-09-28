"""Distress inventory proxies and Visual Maintenance Priority Score (VMPS).
Boxes: list of (cls, xc, yc, w, h) normalized to tile size (square tile)."""
import numpy as np, pandas as pd, json, os
from shapely.geometry import box as sbox
from shapely.ops import unary_union
WEIGHTS=dict(L_line=1.0,A_block=1.5,A_pit=2.5,N_pit=0.5,C_multi=0.5)
BANDS=[(0,25,'Low'),(25,50,'Moderate'),(50,75,'High'),(75,100.01,'Critical')]
def union_area(bs):
    if not bs: return 0.0
    polys=[sbox(x-w/2,y-h/2,x+w/2,y+h/2) for (_,x,y,w,h) in bs]
    return float(unary_union(polys).area)
def tile_features(boxes):
    line=[b for b in boxes if b[0]==0]; blk=[b for b in boxes if b[0]==1]; pit=[b for b in boxes if b[0]==2]
    f=dict(N_line=len(line),N_block=len(blk),N_pit=len(pit),
        L_line=sum(np.hypot(b[3],b[4]) for b in line)/np.sqrt(2),   # chord length proxy, tile-diagonal units
        A_line_bbox=sum(b[3]*b[4] for b in line),                     # naive bbox area (for comparison only)
        A_block=union_area(blk),A_pit=union_area(pit))
    f['C_multi']=int(sum(1 for k in ['N_line','N_block','N_pit'] if f[k]>0)>=2)
    return f
def fit_reference(feat_df,q=95):
    """Reference scale: percentile q of each component over the training tiles (tiles with the class present)."""
    ref={}
    for k,col in [('L_line','L_line'),('A_block','A_block'),('A_pit','A_pit'),('N_pit','N_pit')]:
        v=feat_df[col][feat_df[col]>0]; ref[k]=float(np.percentile(v,q)) if len(v) else 1.0
    return ref
def vmps_raw(feat_df,ref,weights=WEIGHTS):
    """Weighted sum of clipped, reference-scaled components (unbounded above)."""
    c={k:np.minimum(feat_df[k]/ref[k],1.0) for k in ['L_line','A_block','A_pit','N_pit']}; c['C_multi']=feat_df['C_multi'].astype(float)
    return sum(weights[k]*c[k] for k in weights)
def vmps(feat_df,ref,weights=WEIGHTS):
    """0-100 score: raw score divided by the training-set P95 of the raw score (ref['P95_raw']), clipped at 100."""
    return (100*np.minimum(vmps_raw(feat_df,ref,weights)/ref['P95_raw'],1.0)).round(4)
def band(v):
    """Screening bands with left-closed intervals: Low [0,25), Moderate [25,50), High [50,75), Critical [75,100]."""
    return pd.cut(v,bins=[0,25,50,75,100.01],right=False,labels=['Low','Moderate','High','Critical'])
def section_aggregate(df,score='VMPS',group='src'):
    g=df.groupby(group)
    out=g.agg(n_tiles=('stem','size'),N_line=('N_line','sum'),N_block=('N_block','sum'),N_pit=('N_pit','sum'),
              mean_VMPS=(score,'mean'),p90_VMPS=(score,lambda x: np.percentile(x,90)),
              max_VMPS=(score,'max'),sum_L_line=('L_line','sum'),sum_A_block=('A_block','sum'),sum_A_pit=('A_pit','sum'))
    out['critical_ratio']=g[score].apply(lambda x:(x>=75).mean()); out['pit_tile_ratio']=g['N_pit'].apply(lambda x:(x>0).mean())
    out['S_VMPS']=0.4*out.mean_VMPS+0.4*out.p90_VMPS+0.2*100*out.critical_ratio
    return out
def read_yolo_txt(path,conf=False):
    boxes=[]
    if os.path.exists(path):
        for line in open(path):
            p=line.split()
            if len(p)>=5:
                b=(int(p[0]),*map(float,p[1:5]))
                if conf and len(p)>=6: b=b+(float(p[5]),)
                boxes.append(b)
    return boxes
