"""Dataset characterization tables and figures (GT annotations)."""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'code'); from inventory import read_yolo_txt
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.linewidth': 0.6, 'figure.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'mathtext.fontset': 'stix', 'font.serif': ['STIXGeneral', 'DejaVu Serif']})
df=pd.read_csv('results/tiles_gt_inventory.csv')
rows=[]
for r in df.itertuples():
    for (c,x,y,w,h) in read_yolo_txt(f'data/HighRPD/labels/{r.stem}.txt'): rows.append((r.stem,r.split_source,c,w,h))
B=pd.DataFrame(rows,columns=['stem','split','cls','w','h']); B['area']=B.w*B.h; B['ar']=B.w/B.h; B['diag']=np.hypot(B.w,B.h)/np.sqrt(2)
B['long']=B[['w','h']].max(axis=1); B['short']=B[['w','h']].min(axis=1)
B.to_csv('results/boxes_gt.csv',index=False)
names={0:'line',1:'block',2:'pit'}
T=B.groupby('cls').agg(n=('w','size'),area_med=('area','median'),area_p95=('area',lambda x:x.quantile(.95)),long_med=('long','median'),short_med=('short','median'),ar_med=('ar','median'),full_extent=('long',lambda x:(x>=0.999).mean())).rename(index=names)
T['tiles_with']=[ (df.N_line>0).sum(),(df.N_block>0).sum(),(df.N_pit>0).sum()]
T['share_instances']=T.n/T.n.sum()
T.round(4).to_csv('results/table_class_stats.csv'); print(T.round(4))
comb=df.assign(L=df.N_line>0,Bk=df.N_block>0,P=df.N_pit>0).groupby(['L','Bk','P']).size(); print(comb)
print('multi-class tiles',df.C_multi.mean().round(3),'; tiles per source median',df.groupby('src').size().median())
# line orientation classes: horizontal (w>2h), vertical (h>2w), other
L=B[B.cls==0]; ori=np.where(L.w>2*L.h,'transverse-like',np.where(L.h>2*L.w,'longitudinal-like','oblique/irregular')); print(pd.Series(ori).value_counts(normalize=True).round(3))
# Figure: class statistics
fig,ax=plt.subplots(1,4,figsize=(7.2,1.9))
ax[0].bar(['line','block','pit'],[T.n['line'],T.n['block'],T.n['pit']],color=['#4477aa','#ee6677','#228833']); ax[0].set_ylabel('Instances'); ax[0].set_title('(a) Instances per class')
for c,col in zip([0,1,2],['#4477aa','#ee6677','#228833']): ax[1].hist(np.sqrt(B[B.cls==c].area),bins=40,range=(0,1),histtype='step',color=col,label=names[c],density=True)
ax[1].set_xlabel('sqrt(box area / tile area)'); ax[1].set_ylabel('Density'); ax[1].legend(frameon=False); ax[1].set_title('(b) Box size')
ax[2].hist(np.log10(L.ar),bins=40,color='#4477aa'); ax[2].set_xlabel('log10(width / height), line'); ax[2].set_title('(c) Line box aspect ratio')
ax[3].hist(df.groupby('src').size(),bins=30,color='grey'); ax[3].set_xlabel('Tiles per source group'); ax[3].set_ylabel('Source images'); ax[3].set_title('(d) Tiles per source')
for a in ax: a.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig('figures/fig_dataset_stats.pdf'); plt.savefig('figures/fig_dataset_stats.png')
