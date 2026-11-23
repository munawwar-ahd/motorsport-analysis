import pandas as pd, numpy as np, time, json
from scipy.optimize import minimize

t0=time.time()
r=pd.read_csv('races.csv'); l=pd.read_csv('lap_times.csv'); p=pd.read_csv('pit_stops.csv'); res=pd.read_csv('results.csv')
rids=r[(r.year>=2011)].raceId
p=p[p.raceId.isin(rids)].copy()
l=l[l.raceId.isin(rids)].copy()
p['lap']=p.lap.astype(int); p['ms']=pd.to_numeric(p.milliseconds, errors='coerce')
l['lap']=l.lap.astype(int); l['ms']=l.milliseconds.astype(int); l['position']=l.position.astype(int)
print('loaded',time.time()-t0)

# neutral/pitloss/wear per race for a sample
rows=[]
for rid in rids[:20]:
    x=l[l.raceId==rid]
    total=x.lap.max()
    pits=p[p.raceId==rid]
    io=set()
    for _,s in pits.iterrows():
        io.add((s.driverId,s.lap)); io.add((s.driverId,s.lap+1))
    meds={}
    for lap,g in x[x.lap>=2].groupby('lap'):
        g=g[[ (d,ll) not in io for d,ll in zip(g.driverId,g.lap)]]
        if len(g)>=5: meds[lap]=g.ms.median()
    vals=np.array(list(meds.values()))
    p15=np.percentile(vals,15); thr=p15*1.08
    neut=[lap for lap,v in meds.items() if v>thr]
    print(rid, r.loc[r.raceId==rid,'name'].values[0], 'total',total,'p15',round(p15),'thr',round(thr),'neut',len(neut),neut[:10], 'medmax',max(vals))
print('done',time.time()-t0)
