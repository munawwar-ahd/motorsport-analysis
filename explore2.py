import pandas as pd, numpy as np, time, json
from scipy.optimize import minimize
from collections import defaultdict

t0=time.time()
r=pd.read_csv('races.csv'); l=pd.read_csv('lap_times.csv'); p=pd.read_csv('pit_stops.csv'); res=pd.read_csv('results.csv'); cir=pd.read_csv('circuits.csv')
lrids=set(l.raceId.unique()); prids=set(p.raceId.unique())
rids=[int(x) for x in r[(r.year>=2011)&(r.raceId.isin(lrids))&(r.raceId.isin(prids))].raceId]
print('races',len(rids))
p=p[p.raceId.isin(rids)].copy(); l=l[l.raceId.isin(rids)].copy()
p['lap']=p.lap.astype(int); p['ms']=pd.to_numeric(p.milliseconds, errors='coerce')
l['lap']=l.lap.astype(int); l['ms']=l.milliseconds.astype(int); l['position']=l.position.astype(int)

races={}
for rid in rids:
    rr=r[r.raceId==rid].iloc[0]
    x=l[l.raceId==rid]
    total=int(x.lap.max())
    pits=p[p.raceId==rid].copy()
    io=set()
    for _,s in pits.iterrows():
        io.add((int(s.driverId),int(s.lap))); io.add((int(s.driverId),int(s.lap+1)))
    meds={}
    for lap,g in x[x.lap>=2].groupby('lap'):
        g2=g[[ (int(d),int(ll)) not in io for d,ll in zip(g.driverId,g.lap)]]
        if len(g2)>=5: meds[int(lap)]=float(g2.ms.median())
    vals=np.array(list(meds.values()))
    p15=float(np.percentile(vals,15)); thr=p15*1.08
    neut=sorted([lap for lap,v in meds.items() if v>thr])
    nset=set(neut)
    # pit losses
    losses=[]
    for _,s in pits.iterrows():
        d=int(s.driverId); sl=int(s.lap)
        if sl<2 or sl>=total: continue
        if sl in nset or sl+1 in nset: continue
        dg=x[x.driverId==d].set_index('lap')
        if sl not in dg.index or sl+1 not in dg.index: continue
        before=[]
        for ll in range(sl-1, max(0,sl-40), -1):
            if ll<2 or ll in nset or (d,ll) in io: continue
            before.append(float(dg.loc[ll,'ms']))
            if len(before)==3: break
        if len(before)<3: continue
        loss=(float(dg.loc[sl,'ms'])+float(dg.loc[sl+1,'ms']))-2*np.median(before)
        loss/=1000
        if 5<loss<45:
            losses.append((d,sl,loss,float(s.ms)/1000 if pd.notna(s.ms) else np.nan))
    # tyre wear
    drivers=sorted(x.driverId.unique())
    Xs=[]; ys=[]
    for d in drivers:
        dg=x[x.driverId==d].sort_values('lap')
        pitlaps=sorted(pits[pits.driverId==d].lap.tolist())
        starts=[1]+[pl+1 for pl in pitlaps]
        for _,row in dg.iterrows():
            lap=int(row.lap)
            if lap<2 or lap in nset or (d,lap) in io: continue
            stint=np.searchsorted(starts, lap, side='right')-1
            age=lap-starts[stint]+1
            Xs.append((age,lap)); ys.append(float(row.ms))
    Xs=np.array(Xs,float); ys=np.array(ys,float)
    n=len(ys)
    if n>10:
        dm=np.zeros(n)
        # demean by driver not available; do per driver below
        # rebuild with driver ids
    # rebuild with driver labels
    rows=[]
    for d in drivers:
        dg=x[x.driverId==d].sort_values('lap')
        pitlaps=sorted(pits[pits.driverId==d].lap.tolist())
        starts=[1]+[pl+1 for pl in pitlaps]
        for _,row in dg.iterrows():
            lap=int(row.lap)
            if lap<2 or lap in nset or (d,lap) in io: continue
            stint=np.searchsorted(starts, lap, side='right')-1
            age=lap-starts[stint]+1
            rows.append((d,age,lap,float(row.ms)))
    df=pd.DataFrame(rows,columns=['d','age','lap','ms'])
    if len(df)>10:
        dm=df.copy()
        for c in ['age','lap','ms']:
            dm[c]=df[c]-df.groupby('d')[c].transform('mean')
        X=np.column_stack([dm.age.values,dm.lap.values]); y=dm.ms.values
        beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
        resid=y-X@beta
        sst=float((y**2).sum()); sse=float((resid**2).sum())
        r2=1-sse/sst if sst>0 else 0
        nd=df.d.nunique(); dof=max(1,len(df)-nd-2)
        sd=float(np.sqrt(sse/dof))
        slope=float(beta[0]/1000); fuel=float(beta[1]/1000)
    else:
        slope=fuel=r2=sd=0; n=len(df)
    races[rid]={'year':int(rr.year),'total':total,'neut':neut,'neutfrac':len(neut)/max(1,total-1),
                'losses':losses,'slope':slope,'fuel':fuel,'r2':r2,'sd':sd,'n':len(df)}

print('races computed',time.time()-t0)
slopes=np.array([v['slope'] for v in races.values()])
print('slopes pct',np.percentile(slopes,[0,5,10,25,50,75,90,95,100]))
print('slope<0.005',(slopes<0.005).sum(),'neutfrac>0.2',sum(v['neutfrac']>0.2 for v in races.values()),'n<100',sum(v['n']<100 for v in races.values()))
print('r2 pct',np.percentile([v['r2'] for v in races.values()],[0,25,50,75,100]))
print('sd pct',np.percentile([v['sd'] for v in races.values()],[0,25,50,75,100]))
allloss=[z[2] for v in races.values() for z in v['losses']]
print('losses n',len(allloss),'median',np.median(allloss),'p25',np.percentile(allloss,25),'p75',np.percentile(allloss,75))
# circuit pool
cstats=defaultdict(list)
for rid,v in races.items():
    cid=int(r.loc[r.raceId==rid,'circuitId'].iloc[0])
    for z in v['losses']: cstats[cid].append(z[2])
for cid,vals in sorted(cstats.items(), key=lambda kv:-len(kv[1])):
    a=np.array(vals)
    if len(a)>=15:
        print(cid, cir.loc[cir.circuitId==cid,'name'].values[0][:28],len(a),round(np.median(a),2),round(np.percentile(a,25),2),round(np.percentile(a,75),2))
    else:
        print(cid, cir.loc[cir.circuitId==cid,'name'].values[0][:28],len(a),'INSUFF')
print('global pit',np.median(allloss),len(allloss))
print('elapsed',time.time()-t0)
