import sys; sys.path.insert(0,'code')
from inventory import *
df=pd.read_csv('results/tiles_index.csv')
feats=[]
for r in df.itertuples():
    f=tile_features(read_yolo_txt(f'data/HighRPD/labels/{r.stem}.txt')); f['stem']=r.stem; feats.append(f)
F=pd.DataFrame(feats); df=df.merge(F,on='stem')
tr=df[df.split_source=='train']; ref=fit_reference(tr); ref['P95_raw']=float(np.percentile(vmps_raw(tr,ref),95)); json.dump(ref,open('results/reference_scale_p95.json','w'),indent=1)
print('reference (P95 of train, class present):',ref)
df['VMPS_raw']=vmps_raw(df,ref); df['VMPS']=vmps(df,ref); df['priority']=band(df.VMPS)
df.to_csv('results/tiles_gt_inventory.csv',index=False)
print(df.groupby('split_source').priority.value_counts(normalize=True).unstack().round(3))
print(df.VMPS.describe().round(2))
print('by dominant content:'); 
for k in ['N_line','N_block','N_pit']: print(k,'only:',df[(df[['N_line','N_block','N_pit']]>0).sum(1).eq(1)&(df[k]>0)].VMPS.describe()[['mean','50%','max']].round(1).to_dict())
sec=section_aggregate(df); sec.to_csv('results/sections_gt_inventory.csv'); print(sec.describe().round(2).T)
# sensitivity of reference percentile and weights
from scipy.stats import spearmanr, kendalltau
qrows=[]
for q in [90,95,99]:
    rq=fit_reference(tr,q); rq['P95_raw']=float(np.percentile(vmps_raw(tr,rq),95)); s=vmps(df,rq); sr=vmps_raw(df,rq)
    qrows.append(dict(component_percentile=q,spearman=spearmanr(sr,df.VMPS_raw).correlation,band_agreement=(band(s)==df.priority).mean(),**{b:(band(s)==b).mean() for b in ['Low','Moderate','High','Critical']}))
for q in [90,99]:
    rq=dict(ref); rq['P95_raw']=float(np.percentile(vmps_raw(tr,ref),q)); s=vmps(df,rq)
    qrows.append(dict(component_percentile=95,score_percentile=q,spearman=1.0,band_agreement=(band(s)==df.priority).mean(),**{b:(band(s)==b).mean() for b in ['Low','Moderate','High','Critical']}))
pd.DataFrame(qrows).round(4).to_csv('results/vmps_reference_sensitivity.csv',index=False); print(pd.DataFrame(qrows).round(3))
scen={'Base':WEIGHTS,'Pit-light':dict(L_line=1,A_block=1.5,A_pit=2.0,N_pit=0.3,C_multi=0.5),'Pit-heavy':dict(L_line=1,A_block=1.5,A_pit=3.0,N_pit=0.8,C_multi=0.5),'Crack-heavy':dict(L_line=1.3,A_block=1.8,A_pit=2.5,N_pit=0.5,C_multi=0.5),'Equal':dict(L_line=1,A_block=1,A_pit=1,N_pit=1,C_multi=1)}
rows=[]
top=set(df.nlargest(int(0.05*len(df)),'VMPS_raw').stem)
for n,w in scen.items():
    rw=dict(ref); rw['P95_raw']=float(np.percentile(vmps_raw(tr,ref,w),95)); s=vmps(df,rw,w); sr=vmps_raw(df,rw,w); t=set(df.assign(s=sr).nlargest(int(0.05*len(df)),'s').stem)
    rows.append(dict(scenario=n,spearman=spearmanr(sr,df.VMPS_raw).correlation,kendall=kendalltau(sr,df.VMPS_raw).correlation,band_agreement=(band(s)==df.priority).mean(),top5pct_overlap=len(top&t)/len(top),critical_share=(band(s)=='Critical').mean()))
pd.DataFrame(rows).round(4).to_csv('results/vmps_weight_sensitivity.csv',index=False); print(pd.DataFrame(rows).round(4))
