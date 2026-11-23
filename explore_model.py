import pandas as pd, numpy as np, time, json
from scipy.optimize import minimize
from scipy.special import logsumexp

t0=time.time()
r=pd.read_csv('races.csv'); l=pd.read_csv('lap_times.csv'); p=pd.read_csv('pit_stops.csv'); res=pd.read_csv('results.csv')
lrids=set(l.raceId.unique()); prids=set(p.raceId.unique())
rids=[int(x) for x in r[(r.year>=2011)&(r.raceId.isin(lrids))&(r.raceId.isin(prids))].raceId]
print('races',len(rids))
p=p[p.raceId.isin(rids)].copy(); l=l[l.raceId.isin(rids)].copy(); res=res[res.raceId.isin(rids)].copy()
p['lap']=p.lap.astype(int); p['ms']=pd.to_numeric(p.milliseconds, errors='coerce')
l['lap']=l.lap.astype(int); l['ms']=l.milliseconds.astype(int); l['position']=l.position.astype(int)

rows=[]; groups=[]; race_meta={}
for rid in rids:
    rr=r[r.raceId==rid].iloc[0]
    x=l[l.raceId==rid]
    total=int(x.lap.max())
    pits=p[p.raceId==rid]
    io=set()
    for _,s in pits.iterrows():
        io.add((int(s.driverId),int(s.lap))); io.add((int(s.driverId),int(s.lap+1)))
    meds={}
    for lap,g in x[x.lap>=2].groupby('lap'):
        g2=g[[ (int(d),int(ll)) not in io for d,ll in zip(g.driverId,g.lap)]]
        if len(g2)>=5: meds[int(lap)]=float(g2.ms.median())
    vals=np.array(list(meds.values()))
    if len(vals):
        thr=float(np.percentile(vals,15))*1.08
        nset=sorted([lap for lap,v in meds.items() if v>thr])
    else:
        nset=[]
    nset=set(nset)
    drivers=sorted(int(d) for d in x.driverId.unique())
    # per driver arrays by lap
    lapms={}; pos={}; pits_by={}
    for d in drivers:
        dg=x[x.driverId==d].sort_values('lap')
        lapms[d]={int(a):int(b) for a,b in zip(dg.lap,dg.ms)}
        pos[d]={int(a):int(b) for a,b in zip(dg.lap,dg.position)}
        pits_by[d]=[(int(a),float(b)) for a,b in zip(pits[pits.driverId==d].lap,pits[pits.driverId==d].ms)]
    # cumulative times
    cum={}
    for d in drivers:
        c={0:0.0}; acc=0.0
        for lap in range(1,total+1):
            if lap not in lapms[d]: break
            acc+=lapms[d][lap]
            c[lap]=acc
        cum[d]=c
    # clean check
    def clean(d,lap):
        return lap>=2 and lap not in nset and (d,lap) not in io and lap in lapms[d]
    def recent_pace(d,lap):
        vals=[]
        for ll in range(lap,max(0,lap-10),-1):
            if clean(d,ll): vals.append(lapms[d][ll])
            if len(vals)==5: break
        if len(vals)<2:
            vals=[]
            for ll in range(lap,0,-1):
                if ll in lapms[d]: vals.append(lapms[d][ll])
                if len(vals)==5: break
        return float(np.median(vals)) if vals else np.nan
    winrow=res[(res.raceId==rid)&(res.positionOrder==1)]
    winner=int(winrow.driverId.iloc[0]) if len(winrow) else None
    if winner is None or winner not in drivers:
        print('NO WINNER',rid); continue
    for lap in range(1,total+1):
        running=[d for d in drivers if lap in lapms[d]]
        if len(running)<2: continue
        leader=min(running,key=lambda d:pos[d][lap])
        pace_lead=recent_pace(leader,lap)
        stops={d:sum(1 for sl,_ in pits_by[d] if sl<lap) for d in running}
        med_stops=float(np.median(list(stops.values())))
        laps_left=total-lap
        gapl={d:(cum[d].get(lap,0)-cum[leader].get(lap,0)) for d in running}
        for d in running:
            pace=recent_pace(d,lap)
            if np.isnan(pace) or np.isnan(pace_lead): pv=0.0
            else: pv=float(np.clip((pace-pace_lead)/1000,-3,5))
            f1=np.log(pos[d][lap])
            f2=f1*laps_left/total
            f3=min(gapl[d]/1000,300)/np.sqrt(laps_left+1)
            f4=pv*np.sqrt(laps_left+1)/10
            f5=float(np.clip(stops[d]-med_stops,-2,2))
            rows.append((rid,lap,d,pos[d][lap],gapl[d]/1000,pace/1000 if not np.isnan(pace) else 0,stops[d],f1,f2,f3,f4,f5))
        groups.append((rid,lap,len(running),sum(1 for d in running if d==winner)))
    race_meta[rid]={'year':int(rr.year),'total':total,'winner':winner}
print('rows',len(rows),'groups',len(groups),'time',time.time()-t0)
df=pd.DataFrame(rows,columns=['rid','lap','d','pos','gap','pace','stops','f1','f2','f3','f4','f5'])
df['year']=df.rid.map(lambda z:race_meta[z]['year'])
df['winner']=df.d.map(lambda z:0)
# mark winner
wmap={rid:race_meta[rid]['winner'] for rid in race_meta}
df['y']=(df.d==df.rid.map(wmap)).astype(int)
df['total']=df.rid.map(lambda z:race_meta[z]['total'])
print(df.groupby('year').size().to_string())
print('winner rows',df.y.sum(),'groups',df.groupby(['rid','lap']).ngroups)
print('f desc');print(df[['f1','f2','f3','f4','f5','gap','pos','stops']].describe().to_string())

feat=['f1','f2','f3','f4','f5']
train=df[df.year<=2022].copy(); test=df[df.year>=2023].copy()
mean=train[feat].mean().values; std=train[feat].std(ddof=0).values.copy(); std[std==0]=1
def make_X(d,cols,mean,std):
    return ((d[cols].values-mean)/std)
Xtr=make_X(train,feat,mean,std); Xte=make_X(test,feat,mean,std)
ytr=train.y.values; yte=test.y.values

def group_index(d):
    key=d.rid.astype(str)+'_'+d.lap.astype(str)
    codes,uniques=pd.factorize(key,sort=False)
    # codes may not be contiguous in order? factorize does
    order=np.argsort(codes,kind='stable'); codes=codes[order]
    return codes, order, len(uniques)
ctr,otr,Gtr=group_index(train); cte,ote,Gte=group_index(test)
Xtr=Xtr[otr]; ytr=ytr[otr]
Xte=Xte[ote]; yte=yte[ote]

def fit(X,codes,G,y,lam=1e-3):
    starts=np.flatnonzero(np.r_[True,codes[1:]!=codes[:-1]])
    counts=np.diff(np.r_[starts,len(codes)])
    win_idx=np.flatnonzero(y==1)
    # map winner row to group: assume one per group
    win_group=codes[win_idx]
    win_pos=win_idx
    # group starts lookup
    gstart=starts[win_group]
    # winner local index
    win_local=win_pos-gstart
    def fg(beta):
        s=X@beta
        # per-group max and sumexp
        gmax=np.maximum.reduceat(s,starts)
        ex=np.exp(s-gmax[codes])
        gsum=np.add.reduceat(ex,starts)
        logz=gmax+np.log(gsum)
        ll=np.sum(s[win_pos]-logz[win_group])
        loss=-ll/G+0.5*lam*np.dot(beta,beta)
        # probs
        p=ex/gsum[codes]
        grad=(X*p[:,None]).sum(axis=0)
        grad-=np.sum(X[win_pos],axis=0)
        grad=grad/G+lam*beta
        return loss,grad
    res=minimize(lambda b:fg(b),np.zeros(X.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-12,'gtol':1e-9})
    return res

print('fitting model')
res=fit(Xtr,ctr,Gtr,ytr)
beta=res.x
print('success',res.success,res.message,'loss',res.fun,'beta',beta)

# baselines
res_pos=fit(Xtr[:,[0]],ctr,Gtr,ytr)
print('pos beta',res_pos.x,res_pos.fun)

def metrics(X,codes,G,y,beta,lam=1e-3):
    s=X@beta
    starts=np.flatnonzero(np.r_[True,codes[1:]!=codes[:-1]])
    gmax=np.maximum.reduceat(s,starts)
    ex=np.exp(s-gmax[codes]); gsum=np.add.reduceat(ex,starts)
    p=ex/gsum[codes]
    win_idx=np.flatnonzero(y==1); wg=codes[win_idx]
    ll=-np.mean(np.log(p[win_idx]))
    # multiclass brier
    ss=np.add.reduceat((p-y)**2,starts); brier=np.mean(ss)
    acc=np.mean([np.argmax(p[starts[g]:starts[g]+np.diff(np.r_[starts,len(p)])[g]])==(win_idx[list(wg).index(g)]-starts[g]) for g in range(G)]) if False else None
    # accuracy easier: winner's p is max in group
    correct=0
    for g in range(G):
        a=starts[g]; b=starts[g+1] if g+1<G else len(p)
        wi=win_idx[np.where(wg==g)[0][0]]
        if np.argmax(p[a:b])==wi-a: correct+=1
    return ll,brier,correct/G,p

for name,bb in [('model',beta),('pos',res_pos.x)]:
    # for pos use only first col standardized
    if name=='pos':
        Xt=Xte[:,[0]]
    else: Xt=Xte
    ll,br,acc,pp=metrics(Xt,cte,Gte,yte,bb)
    print(name,'test logloss',ll,'brier',br,'acc',acc)
# leader & uniform
p_leader=np.zeros(len(yte)); leader_mask=(test.pos.values==1)
eps=1e-6
# group sizes
starts=np.flatnonzero(np.r_[True,cte[1:]!=cte[:-1]])
sizes=np.diff(np.r_[starts,len(cte)])
p_leader=np.where(leader_mask,1-eps,eps/(np.repeat(sizes,sizes)-1))
p_uniform=1/np.repeat(sizes,sizes)
for name,pp in [('leader',p_leader),('uniform',p_uniform)]:
    win_idx=np.flatnonzero(yte==1)
    ll=-np.mean(np.log(np.clip(pp[win_idx],1e-12,None)))
    ss=np.add.reduceat((pp-yte)**2,starts); br=np.mean(ss)
    correct=0
    for g in range(Gte):
        a=starts[g]; b=starts[g+1] if g+1<Gte else len(pp)
        wi=win_idx[np.where(cte[win_idx]==g)[0][0]] if False else None
    print(name,'logloss',ll,'brier',br)
print('leader %',leader_mask.mean(),'winner leader groups',np.mean([ (test[(test.rid==rid)&(test.lap==lap)].pos.min()==1) for rid,lap in []]) if False else '')
# leader accuracy over groups
acc_leader=np.mean([ (test[(test.rid==rid)&(test.lap==lap)].loc[test[(test.rid==rid)&(test.lap==lap)].y==1,'pos'].iloc[0]==1) for rid,lap in test[['rid','lap']].drop_duplicates().values])
print('leader acc',acc_leader)
print('elapsed',time.time()-t0)
