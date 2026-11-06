# -*- coding: utf-8 -*-
from data import (tt)
import pyomo.environ as pyo
import time

def FS_Check(gid,dt,data,s):
    pl = []
    for l in s.x.l.unique():
        d = s.x.loc[s.x['l'] == l]
        rl = []
        a = d.loc[d['a0']==0]
        rl.append(a.a0.iloc[0])
        rl.append(a.a3.iloc[0])
        cond = True
        start_time = time.time()
        while cond:
            current_time = time.time()
            if rl[-1] == 0:
                break
            if current_time - start_time > 60:
                return ['infeasible',[]]
            b = d.loc[(d['a0']==rl[-1]) & (d['a3']!=rl[-1])]
            rl.append(b.a3.iloc[0])
        pl.append(rl)        

    pk = []
    for k in data.k.index:
        ok = data.k.oid.loc[k]
        dk = data.k.did.loc[k]
        a = s.f.ae.loc[(s.f['a0'] == ok) & (s.f['k'] == k)].iloc[0]
        b = s.f.ae.loc[(s.f['a3'] == dk) & (s.f['k'] == k)].iloc[0]
        l1 =  s.x.l.loc[s.x['ae'] == a].iloc[0]
        l2 =  s.x.l.loc[s.x['ae'] == b].iloc[0]
        pk.append([l1,l2])

    uhkll = []
    for h in data.n.loc[data.n['Type'] == 'H'].index:
        for k in data.k.index:
            a  = s.f[(s.f['a0']!=h)&(s.f['a3']==h)&(s.f['k']==k)]
            if len(a) > 0:
                b  = s.f[(s.f['a0']==h)&(s.f['a3']!=h)&(s.f['k']==k)]
                l1 =  s.x.l.loc[s.x['ae'] == a.ae.iloc[0]].iloc[0]
                l2 =  s.x.l.loc[s.x['ae'] == b.ae.iloc[0]].iloc[0]
                if l1 != l2:
                    uhkll.append([h,k,l1,l2])


    model = pyo.ConcreteModel('FS_Check')
    model.nl_index = pyo.Set(initialize=[(n,l) for l in range(len(pl)) for n in pl[l]])
    model.enl = pyo.Var(model.nl_index, domain = pyo.NonNegativeReals,bounds=(0,480+dt))
    model.dnl = pyo.Var(model.nl_index, domain = pyo.NonNegativeReals,bounds=(0,480+dt))

    def obj_rule(m):
        return sum(m.enl[0,l]-m.dnl[0,l] for l in range(len(pl)))
    model.obj = pyo.Objective(rule = obj_rule, sense = pyo.minimize)

    @model.Constraint(uhkll)
    def hub_sync(m,h,k,l1,l2):
        return m.enl[h,l1] <= m.dnl[h,l2]

    lpl_index = [] 
    for l in range(len(pl)):
        lpl = len(pl[l])
        for i in range(1,lpl):
            n = pl[l][i-1]
            m = pl[l][i]
            lpl_index.append([l,n,m])        
        
    @model.Constraint(lpl_index)
    def nl_times(m,l,i,j):
        return m.dnl[i,l] + tt(i, j, data.tt_m,dt) == m.enl[j,l]

    @model.Constraint(model.nl_index)
    def de_times(m,n,l):
        if n == 0:
            return m.enl[n,l] >= m.dnl[n,l]
        else:
            return m.enl[n,l] <= m.dnl[n,l]

    @model.Constraint(data.k.index)
    def k_p(m,k):
        ok = data.k.oid[k]
        okt = data.k.l[k]
        l = pk[k][0]
        return m.dnl[ok,l] >= okt

    @model.Constraint(data.k.index)
    def k_d(m,k):
        dk = data.k.did[k]
        dkt = data.k.u[k]
        l = pk[k][1]
        return m.enl[dk,l] <= dkt

    @model.Constraint(range(len(pl)))
    def l_time(m,l):
        return m.enl[0,l] - m.dnl[0,l] <= 480+dt

    opt = pyo.SolverFactory('gurobi')
    result = opt.solve(model, tee=True)
    return result.solver.Termination_condition, uhkll