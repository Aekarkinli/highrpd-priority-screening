import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.patches as mp
from PIL import Image; from scipy.ndimage import rotate
sys.path.insert(0,'code'); from inventory import *
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.linewidth': 0.6, 'figure.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'stix', 'font.serif': ['STIXGeneral', 'DejaVu Serif']})
COL={0:'#4477aa',1:'#ee6677',2:'#228833'}; NAME={0:'line',1:'block',2:'pit'}; BC={'Low':'#dddddd','Moderate':'#fdd49e','High':'#fc8d59','Critical':'#b30000'}
df=pd.read_csv('results/tiles_gt_inventory.csv')
# --- Figure: sample tiles with annotations
picks=[df[(df.N_line>0)&(df.N_block==0)&(df.N_pit==0)&(df.N_line<=3)].sample(1,random_state=3).stem.iloc[0],
       df[(df.N_line==0)&(df.N_block>0)&(df.N_pit==0)].sample(1,random_state=4).stem.iloc[0],
       df[(df.N_pit>0)&(df.N_line==0)&(df.N_block==0)].sample(1,random_state=5).stem.iloc[0],
       df[(df.C_multi==1)&(df.N_pit>0)].sample(1,random_state=6).stem.iloc[0],
       df[(df.N_line>=4)&(df.N_block==0)].sample(1,random_state=7).stem.iloc[0],
       df[(df.N_block>=3)].sample(1,random_state=8).stem.iloc[0],
       df[(df.N_pit>=2)].sample(1,random_state=9).stem.iloc[0],
       df[(df.C_multi==1)&(df.N_pit==0)].sample(1,random_state=10).stem.iloc[0]]
fig,axs=plt.subplots(2,4,figsize=(7.2,3.8))
for ax,st in zip(axs.ravel(),picks):
    ax.imshow(Image.open(f'data/HighRPD/images/{st}.jpg')); 
    for c,x,y,w,h in read_yolo_txt(f'data/HighRPD/labels/{st}.txt'): ax.add_patch(mp.Rectangle(((x-w/2)*640,(y-h/2)*640),w*640,h*640,fill=False,ec=COL[c],lw=0.9))
    r=df[df.stem==st].iloc[0]; ax.set_title(f'VMPS {r.VMPS:.0f} ({r.priority})',fontsize=7); ax.axis('off')
h=[mp.Patch(fc='none',ec=COL[c],label=NAME[c]) for c in COL]; fig.legend(handles=h,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.01))
plt.tight_layout(rect=(0,0.04,1,1)); plt.savefig('figures/fig_samples.png',dpi=250); plt.savefig('figures/fig_samples.pdf'); plt.close()
# --- Figure: VMPS distribution and components
fig,ax=plt.subplots(1,3,figsize=(7.2,2.0))
ax[0].hist(df.VMPS,bins=50,color='grey'); 
for t in [25,50,75]: ax[0].axvline(t,color='k',lw=0.5,ls='--')
ax[0].set_xlabel('VMPS (reference annotations)'); ax[0].set_ylabel('Tiles'); ax[0].set_title('(a) Tile-level score distribution')
sh=df.priority.value_counts(normalize=True).reindex(['Low','Moderate','High','Critical']); ax[1].bar(range(4),sh.values*100,color=[BC[b] for b in sh.index],edgecolor='k',lw=0.4); ax[1].set_xticks(range(4)); ax[1].set_xticklabels(sh.index,fontsize=6.5); ax[1].set_ylabel('Share of tiles (%)'); ax[1].set_title('(b) Screening bands')
import json; RF=json.load(open('results/reference_scale_p95.json')); comp=pd.DataFrame({'line':df.L_line/RF['L_line'],'block':df.A_block/RF['A_block'],'pit area':df.A_pit/RF['A_pit'],'pit count':df.N_pit/RF['N_pit']}).clip(upper=1)
ax[2].boxplot([comp[c][comp[c]>0] for c in comp.columns],tick_labels=['line','block','pit area','pit count'],widths=0.5,flierprops=dict(ms=1.5,alpha=0.3)); ax[2].tick_params(axis='x',labelsize=6.5); ax[2].set_ylabel('Scaled component'); ax[2].set_title('(c) Scaled components')
for a in ax: a.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('figures/fig_vmps_dist.pdf'); plt.close()
# --- Figure: section-level maps (two source images)
sec=pd.read_csv('results/sections_gt_inventory.csv')
cand=sec[(sec.n_tiles>=40)&(sec.N_pit>=4)].sort_values('S_VMPS',ascending=False)
srcs=[cand.index[0],cand.index[len(cand)//2]] if len(cand)>1 else list(cand.index[:2])
srcs=[sec.loc[i,'src'] for i in srcs]
fig,axs=plt.subplots(1,2,figsize=(7.2,2.9))
for ax,s in zip(axs,srcs):
    sub=df[df.src==s]; R,C=sub.a.max()+1,sub.b.max()+1
    for r in sub.itertuples():
        ax.add_patch(mp.Rectangle((r.b,R-1-r.a),1,1,fc=BC[r.priority],ec='k',lw=0.3)); ax.text(r.b+0.5,R-1-r.a+0.5,f'{r.VMPS:.0f}',ha='center',va='center',fontsize=4.5)
        if r.N_pit>0: ax.plot(r.b+0.85,R-1-r.a+0.85,'o',ms=2,color=COL[2])
    ax.set_xlim(0,C); ax.set_ylim(0,R); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ss=sec[sec.src==s].iloc[0]; ax.set_title(f'{s}: {int(ss.n_tiles)} tiles, line {int(ss.N_line)}, block {int(ss.N_block)}, pit {int(ss.N_pit)}; section score {ss.S_VMPS:.1f}',fontsize=6.5)
h=[mp.Patch(fc=BC[b],ec='k',lw=0.3,label=b) for b in BC]+[plt.Line2D([],[],marker='o',color=COL[2],ls='',ms=3,label='pit present')]
fig.legend(handles=h,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(0.5,-0.01),fontsize=7)
plt.tight_layout(rect=(0,0.06,1,1)); plt.savefig('figures/fig_section_map_gt.pdf'); plt.close()
print('section map sources',srcs)
# --- Figure: SAM proxy analysis (line instances)
I=pd.read_csv('results/sam_instances.csv'); M=np.load('results/sam_masks.npz')
L=I[(I.cls==0)&(I.mask_area>0)]
angles=range(0,91,5); A=[];D=[]
for r in L.itertuples():
    shp=M[r.stem+'_shape']; m=np.unpackbits(M[r.stem])[:np.prod(shp)].reshape(shp)[r.inst].astype(bool)
    ys,xs=np.where(m); sub=m[ys.min():ys.max()+1,xs.min():xs.max()+1]; pad=int(np.hypot(*sub.shape))+2
    can=np.zeros((2*pad,2*pad),bool); can[pad-sub.shape[0]//2:pad-sub.shape[0]//2+sub.shape[0],pad-sub.shape[1]//2:pad-sub.shape[1]//2+sub.shape[1]]=sub
    a=[];d=[]
    for th in angles:
        rm=rotate(can,th,order=0,reshape=False)>0; yy,xx=np.where(rm); ww,hh=xx.max()-xx.min()+1,yy.max()-yy.min()+1; a.append(ww*hh); d.append(np.hypot(ww,hh))
    A.append(np.array(a)/a[0]); D.append(np.array(d)/d[0])
A=np.array(A); D=np.array(D)
fig,ax=plt.subplots(1,3,figsize=(7.2,2.2))
ex=L.sort_values('bbox_diag',ascending=False).iloc[[5,40]]
for k,(r,a) in enumerate(zip(ex.itertuples(),ax[:2])):
    shp=M[r.stem+'_shape']; m=np.unpackbits(M[r.stem])[:np.prod(shp)].reshape(shp)[r.inst].astype(bool)
    img=np.asarray(Image.open(f'data/HighRPD/images/{r.stem}.jpg')); a.imshow(img); a.imshow(np.ma.masked_where(~m,m),cmap='autumn',alpha=0.7,vmin=0,vmax=1)
    b=read_yolo_txt(f'data/HighRPD/labels/{r.stem}.txt')[r.inst]; a.add_patch(mp.Rectangle(((b[1]-b[3]/2)*640,(b[2]-b[4]/2)*640),b[3]*640,b[4]*640,fill=False,ec=COL[0],lw=0.9)); a.axis('off')
    a.set_title(f'({"ab"[k]}) box area {r.bbox_area:.3f}, diagonal {r.bbox_diag:.2f}, mask fill {r.mask_fill:.2f}',fontsize=6.5)
th=np.array(list(angles))
for arr,c,l in [(A,'#ee6677','bounding-box area'),(D,'#4477aa','bounding-box diagonal')]:
    med=np.median(arr,0); ax[2].plot(th,med,color=c,label=l); ax[2].fill_between(th,np.percentile(arr,25,0),np.percentile(arr,75,0),color=c,alpha=0.2)
ax[2].set_xlabel('Rotation of crack mask (degrees)'); ax[2].set_ylabel('Proxy relative to 0 degrees'); ax[2].legend(frameon=False,fontsize=6.5); ax[2].set_title('(c) Orientation sensitivity (n=%d line masks)'%len(L),fontsize=6.5); ax[2].spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('figures/fig_proxy_orientation.pdf'); plt.savefig('figures/fig_proxy_orientation.png',dpi=200); plt.close()
np.save('results/sam_rotation_curves.npy',np.stack([np.median(A,0),np.median(D,0)]))
print('median max/min area',np.median(A.max(1)/A.min(1)),'diag',np.median(D.max(1)/D.min(1)))
