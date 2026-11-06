# -*- coding: utf-8 -*-
from data import (DataContainer,data_read,tt,cc)
import numpy as np
np.float_ = np.float64
import pyomo.environ as pyo
import networkx as nx
import pandas as pd
import csv
import os

def ae_exists(i,ti,tti,j,tj,ttj,tt_m,dt):
    if tt(i,j,tt_m,dt) >= ttj - ti:
        return False
    elif i != j and tj - tti >=  tt(i,j,tt_m,dt):
        return False
    elif i == j == 0:
        return False
    return True

def aek_exists(i,ti,tti,j,tj,ttj,k,kdata,tt_m,dt):
    okn = kdata.oid[k]
    okt = kdata.l[k]
    dkn = kdata.did[k]
    dkt = kdata.u[k]
    if i == dkn or j == okn:
        return False
    elif okt + tt(okn,i,tt_m,dt) + tt(i,j,tt_m,dt) + tt(j,dkn,tt_m,dt) > dkt:
        return False
    elif okt + tt(okn,i,tt_m,dt) >= min(tti,ttj-tt(i,j,tt_m,dt)):
        return False
    elif max(okt + tt(okn,i,tt_m,dt)+tt(i,j,tt_m,dt),ti+tt(i,j,tt_m,dt),
              tj) + tt(j,dkn,tt_m,dt) > dkt:
        return False
    return True

def ofndp_int(ii,gid,data,div,rs_path):
    ch = 100
    cl = 500
    cr = 2
    tl = 3  
    th = 2
    dt = 15
    wl = 480
    wlq = 100
    
    TH = data.k.u.max()+1
    interval =  int(np.ceil(TH/div))
    it = [(t,t+interval) for t in range(0,TH,interval)]
    n_i = list(data.n.index)
    n0_i = n_i.copy()
    n0_i.append(0)
    nt0_i = [(n,t[0],t[1]) for n in n0_i for t in it]
    nt_i = [n for n in nt0_i if n[0]!=0]
    ae0_i = [(*n,*m) for n in nt0_i for m in nt0_i if n!=m and ae_exists(*n, *m, data.tt_m,dt)]
    ae_i = [a for a in ae0_i if a[0] != 0 and a[3] != 0]
    d_ae0_i = [a for a in ae0_i if a[0] != a[3]]
    d_ae_i = [a for a in ae_i if a[0] != a[3]]
    h_ae0_i = [a for a in ae0_i if a[0] == a[3]]
    h_ae_i = [a for a in ae_i if a[0] == a[3]]
    h_i = list(data.n[data.n['Type']=='H'].index)
    k_i=list(data.k.index)
    l_i = list(range(5))
    ae_ik = [[a for a in ae_i if aek_exists(*a, k,data.k,data.tt_m,dt)] for k in k_i]
    
    model = pyo.ConcreteModel(name="te_interval")
    model.xal = pyo.Var(ae0_i,l_i, within=pyo.Binary,initialize=0)
    model.yh = pyo.Var(h_i, within=pyo.Binary,initialize=0)
    model.fak_index = pyo.Set(initialize=[(a,k) for k in k_i for a in ae_ik[k]])
    model.fak = pyo.Var(model.fak_index, within=pyo.Binary,initialize=0)
    model.cut = pyo.ConstraintList()
    model.zn = pyo.Var(n_i, within=pyo.Binary,initialize=0)

    def obj_rule(m):
        return sum(ch*m.yh[h] for h in h_i)+ \
            sum(cl*m.xal[a,l] for l in l_i for a in d_ae0_i if a[0]==0)+ \
            sum(cr*cc(a[0],a[3],data.d_m)*m.xal[a,l] for l in l_i for \
                a in d_ae_i)
                
    model.obj = pyo.Objective(rule = obj_rule, sense = pyo.minimize)

    @model.Constraint(h_i,l_i) # 28
    def hub(m,h,l):
        return sum(m.xal[a,l] for a in d_ae0_i if a[0]==h) <= m.yh[h]
    
    @model.Constraint(k_i,h_ae_i) # 29
    def link2(m,k,i,ti1,ti2,j,tj1,tj2):
        if (i,ti1,ti2,j,tj1,tj2) not in ae_ik[k]:
            return pyo.Constraint.Skip
        okn = data.k.oid[k]
        dkn = data.k.did[k]
        if i in h_i or i == okn or i == dkn:
            return  pyo.Constraint.Skip
        else:
            return m.fak[i,ti1,ti2,j,tj1,tj2,k] <= sum(m.xal[i,ti1,ti2,j,tj1,tj2,l] for l in l_i)
    
    @model.Constraint(nt_i) # 30
    def l_visit(m,n,t,t2):
        val=len([a for a in ae0_i if (a[3],a[4],a[5]) == (n,t,t2)])
        if n in h_i or val < 1:
            return pyo.Constraint.Skip
        else:
            return sum(m.xal[a,l] for l in l_i for a in ae0_i if (a[3],a[4],a[5]) == (n,t,t2)) <= 1
    
    @model.Constraint(k_i) # 31
    def origin(m,k):
        okn = data.k.oid[k]
        okt = data.k.l[k]
        return sum(m.fak[a,k] for a in ae_ik[k] if a[0]==okn and (okt >= a[1] and okt < a[2])) == 1
    
    @model.Constraint(k_i) # 32
    def destin(m,k):
        dkn = data.k.did[k]
        dkt = data.k.u[k]
        return sum(m.fak[a,k] for a in ae_ik[k] if a[3]==dkn and (dkt >= a[4] and dkt < a[5])) == 1
    
    # Function to check if the tuple exists in the first or second part of any element in the list
    def check_tuple_in_list(tuples_list, check_tuple):
        for tup in tuples_list:
            if tup[:3] == check_tuple or tup[3:] == check_tuple:
                return False
        return True
    
    @model.Constraint(k_i,nt_i) # 33
    def k_flow_cons(m,k,n,t,t2):
        okn = data.k.oid[k]
        okt = data.k.l[k]
        dkn = data.k.did[k]
        dkt = data.k.u[k]
        if (n == okn and t<=okt<t2) or (n == dkn and t<=dkt<t2):
            return pyo.Constraint.Skip
        elif check_tuple_in_list(ae_ik[k], (n,t,t2)):
            return pyo.Constraint.Skip
        else:
            return sum(m.fak[a,k] for a in ae_ik[k] if (a[3],a[4],a[5])==(n,t,t2)) - \
                sum(m.fak[a,k] for a in ae_ik[k] if (a[0],a[1],a[2])==(n,t,t2)) == 0
    
    @model.Constraint(l_i,nt_i) # 34
    def l_flow_cons(m,l,n,t,t2):
        return sum(m.xal[a,l] for a in ae0_i if (a[3],a[4],a[5]) == (n,t,t2)) - \
            sum(m.xal[a,l] for a in ae0_i if (a[0],a[1],a[2]) == (n,t,t2)) == 0
    
    @model.Constraint(k_i,d_ae_i) # 35
    def link(m,k,i,ti,ti2,j,tj,tj2):
        if (i,ti,ti2,j,tj,tj2) not in ae_ik[k]:
            return pyo.Constraint.Skip
        return m.fak[i,ti,ti2,j,tj,tj2,k] <= sum(m.xal[i,ti,ti2,j,tj,tj2,l] for l in l_i)
            
    @model.Constraint(l_i) # 36
    def lorry_use(m,l):
        return sum(m.xal[a,l] for a in d_ae0_i if a[0]==0) <= 1
    
    def check_in_list(tuples_list, check_tuple):
        for tup in tuples_list:
            if tup == check_tuple:
                return False
        return True
    
    @model.Constraint(d_ae_i) # 37
    def l_cap(m,i,ti,ti2,j,tj,tj2):
        if check_in_list(ae_ik,(i,ti,ti2,j,tj,tj2)):
            return pyo.Constraint. Skip
        else:
            return sum(data.k.nk[k]*m.fak[i,ti,ti2,j,tj,tj2,k] for k in k_i if (i,ti,ti2,j,tj,tj2) in ae_ik[k]) <= wlq
    
    @model.Constraint(l_i) # 38
    def l_maxt(m,l):
        return sum(tt(a[0],a[3],data.tt_m)*m.xal[a,l] for a in ae_i) <= wl
    
    def _add_cut(c,l):
        c_i = range(len(c)-1)
        return model.cut.add(sum(model.xal[c[i],c[i+1],l] for i in c_i)+ model.xal[c[-1],c[0],l]
                          <= len(c)-1)
    
    @model.Constraint(l_i,ae_i)
    def valid_ineq(m,l,i,ti,ti2,j,tj,tj2):
        return m.xal[i,ti,ti2,j,tj,tj2,l] <= sum(m.xal[a,l] for a in d_ae0_i if a[0]==0)
    
    @model.Constraint(l_i)
    def Symmetry(m,l):
        if l == 0:
            return pyo.Constraint.Skip
        else:
            return sum(model.xal[a,l] for a in d_ae0_i if a[0]==0) <= sum(model.xal[a,l-1] for a in d_ae0_i if a[0]==0)
    
    model.c1 = pyo.Constraint(expr = sum(model.xal[a,0] for a in d_ae0_i if a[0]==0) >= 1)
    
    # @model.Constraint(it,l_i)
    # def _ineq1(m,t,t1,l):
    #     return sum(tt(a[0],a[3],data.tt_m)*m.xal[a,l] for a in ae_i if a[1]==a[4]== t and a[2]==a[5]==t1) <= interval
    
    # @model.Constraint(k_i)
    # def _ineq2(m,k):
    #     okt = data.k.l[k]
    #     dkt = data.k.u[k]
    #     return okt + sum(tt(a[0],a[3],data.tt_m)*m.fak[a,k] for a in ae_ik[k]) <= dkt
        
    
    opt = pyo.SolverFactory('gurobi_persistent', options={'TimeLimit' : tl*3600,
                                                'Threads' : th,'SoftMemLimit' : 10})
    opt.set_instance(model)
    opt.set_gurobi_param('PreCrush', 1)
    opt.set_gurobi_param('LazyConstraints', 1)
    
    from gurobipy import GRB
    def my_callback(cb_m, cb_opt, cb_where):
        if cb_where == GRB.Callback.MIPSOL:
            cb_opt.cbGetSolution(vars = [model.xal[a,l] for a in ae_i for l in l_i])
            for l in l_i:
                G = nx.DiGraph()
                G.add_edges_from([((a[0],a[1],a[2]),(a[3],a[4],a[5])) for a in ae_i if pyo.value(model.xal[a,l]>0)])
                Cycles=list(nx.simple_cycles(G))
                if len(Cycles) >= 1:
                    # print('adding cut')
                    for ll in l_i:
                        for c in Cycles:
                            cb_opt.cbLazy(_add_cut(c,ll))
                            
    opt.set_callback(my_callback)
    result = opt.solve(model, tee=True,load_solutions=False)
    if result.solution.Gap < np.inf:
        model.solutions.load_from(result)
        l_used = int(sum(pyo.value(model.xal[a,l])for l in l_i for a in d_ae0_i if a[0]==0))    
        h_used = int(sum(pyo.value(model.yh[h]) for h in h_i))
        of1 = ch*int(sum(pyo.value(model.yh[h]) for h in h_i))
        of2 = int(sum(cl*pyo.value(model.xal[a,l])for l in l_i for a in d_ae0_i if a[0]==0))
        of3 = sum(cr*cc(a[0],a[3],data.d_m)*pyo.value(model.xal[a,l]) for l in l_i for a in d_ae_i)
        of = of1 + of2 + of3
        sol_ae = [(a,l) for l in l_i for a in ae0_i if pyo.value(model.xal[a,l]>0)]
        sol_ae_pd = pd.DataFrame(sol_ae, columns=['ae','l'])
        sol_ae_pd.to_csv(f'{rs_path}/int_cbl_{gid}_l.csv')
        sol_aek = [(a,k) for k in k_i for a in ae_ik[k] if pyo.value(model.fak[a,k]>0)]
        sol_aek_pd = pd.DataFrame(sol_aek, columns=['ae','k'])
        sol_aek_pd.to_csv(f'{rs_path}/int_cbl_{gid}_k.csv')
        solved = 1
    else:
        l_used = np.nan
        h_used = np.nan
        of1 = np.nan
        of2 = np.nan
        of3 = np.nan
        of = np.nan
        solved = 0
    lb = result.problem.Lower_bound
    ub = result.problem.Upper_bound
    gap = abs(result.solution.Gap/ub)*100
    cpu_t = result.solver.Wallclock_time
    status = result.solver.Termination_condition
    var = result.problem.Number_of_variables
    cons = result.problem.Number_of_constraints
    q = sum(data.k.nk)
    if not os.path.isfile(f'{rs_path}/summary.csv'):
        try:
          with open(f'{rs_path}/summary.csv', 'w') as f:
              write = csv.writer(f)
              write.writerow(['job_id','gid','k','n','total_q','TH','int','model','variables','constraints','solved',
                              'status','time','ch','cl','cr','ofv','ub','lb','gap_%','l_used',
                              'hubs_used',
                              'h','l','r'])
        except OSError as error:
          print(error)
    with open(f'{rs_path}/summary.csv', 'a') as f:
        write = csv.writer(f)
        write.writerow([ii,gid,len(k_i),len(data.n),
                        q, data.k.u.max(), len(it),
                        'int',var,cons,solved,status,cpu_t,
                        of1,of2,of3,of,ub,lb,gap,l_used,h_used,
                        ch,cl,cr])
    return result        

# folder_path = 'cbl/cbl_inst'
# rs_path = 'cbl/sol_end'
# import os
# if not os.path.isdir(f'{rs_path}'):
#     try:
#       os.makedirs(f'{rs_path}')
#     except OSError as error:
#       print(error)
# ii = 12
# gid = 12
# data = DataContainer(data_read(gid,folder_path))
# ofndp_int(data, 1)
# from FCheck import FS_Check
# from Data import (sol_read,SolContainer)
# s = SolContainer(sol_read(gid, rs_path, 'int'))
# fsc = FS_Check(gid, 15, data, s)

# from RGCH import rgch

# h_sol = rgch(100, data, 15) 

# n_dk = data.k.copy()
# ind = int(len(n_dk))
# for i in fsc[1]:
#     k = int(i[1])
#     h = int(i[0])
#     o = int(n_dk.loc[k].oid)
#     d = int(n_dk.loc[k].did)
#     nk = int(n_dk.loc[k].nk)
#     l = int(n_dk.loc[k].l)
#     u = int(n_dk.loc[k].u)
#     u2 = int(l + (u-l)/2)
#     l2 = int(u - (u-l)/2)
#     n_dk.loc[ind] = [h,d,nk,0,0,l2,u]
#     n_dk.loc[k].did = h
#     n_dk.loc[k].u = u2
#     ind += 1
    
# data.k = n_dk

# from RGCH import rgch
# h_sol = rgch(50, data, 15) 
