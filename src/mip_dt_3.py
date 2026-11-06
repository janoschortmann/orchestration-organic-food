# -*- coding: utf-8 -*-
from data import (tt,cc)
import pyomo.environ as pyo
import numpy as np
np.float_ = np.float64

def ae_exists(i,ti,j,tj,data,d_list,o_list,dt = 1):
    if i == 0 and j not in d_list and ti ==0:
        return True
    elif j == 0 and i not in o_list and tj > 0:
        return True
    elif i == j and ti + dt == tj and i != 0:
        return True
    elif tj == ti + tt(i, j, data.tt_m, dt) and i != 0 and j != 0:
        return True
    return False

def aek_exists(i,ti,j,tj,k,kdata,tt_m,dt=1):
    okn = kdata.oid[k]
    okt = kdata.l[k]
    dkn = kdata.did[k]
    dkt = kdata.u[k]
    if ti < okt or (ti == okt and i != okn):
        return False
    elif tj > dkt or (tj == dkt and j != dkn):
        return False
    elif i != j and j == okn:
        return False
    elif i != j and i == dkn:
        return False
    elif tj + tt(j,dkn,tt_m,dt) > dkt and j != dkn:
        return False
    return True

def check_tuple_in_list(tuples_list, check_tuple):
    for tup in tuples_list:
        if tup[:2] == check_tuple or tup[2:] == check_tuple:
            return False
    return True

def check_in_list(tuples_list, check_tuple):
    for tup in tuples_list:
        if tup == check_tuple:
            return False
    return True

def a_in_solution(i,j,h_i,solution_arcs):
    if i == 0 or j == 0:
        return True
    if i in h_i or j in h_i:
        return True
    if (i,j) in solution_arcs:
        return True
    return False

def mip_e_solution(data,dt,l,solution_arcs):
    ch = 100
    cl = 500
    cr = 2
    tl = 3  
    th = 2
    wtl = 480
    ql = 100

    h_i = list(data.n[data.n['Type']=='H'].index)
    k_i = list(data.k.index)
    l_i = list(range(l))

    d_list = list(data.n[data.n['Type']=='D'].index)
    d_list.append(0)
    o_list = list(data.n[data.n['Type']=='O'].index)
    o_list.append(0)

    TH = data.k.u.max()
    it = [t for t in range(0,TH+dt,dt)]
    n_i = list(data.n.index)
    n0_i = n_i.copy()
    n0_i.append(0)

    nt0_i = [(0,t) for t in [0,TH+dt]]
    nt0_i.extend([(n,t) for n in n_i for t in it])
    nt_i = [n for n in nt0_i if n[0]!=0]
    if len(solution_arcs) == 0:
        ae0_i = [(*n,*m) for n in nt0_i for m in nt0_i if n!=m and 
                 ae_exists(*n, *m, data,d_list,o_list,dt)]
    else:
        ae0_i = [(*n,*m) for n in nt0_i for m in nt0_i if n!=m and 
                 ae_exists(*n, *m, data,d_list,o_list,dt) and
                 a_in_solution(n[0], m[0], h_i, solution_arcs)]
    
    ae_i = [a for a in ae0_i if a[0] != 0 and a[2] != 0]
    d_ae0_i = [a for a in ae0_i if a[0] != a[2]]
    d_ae_i = [a for a in ae_i if a[0] != a[2]]
    
    ae_ik = [[a for a in ae_i if aek_exists(*a,k,data.k,data.tt_m,dt)] for k in k_i]

    model = pyo.ConcreteModel(name='mip_e')
    model.xal = pyo.Var(ae0_i,l_i, within=pyo.Binary,initialize=0)
    model.yh = pyo.Var(h_i, within=pyo.Binary,initialize=0)
    model.fak_index = pyo.Set(initialize=[(a,k,l) for k in k_i for a in ae_ik[k] for l in l_i])
    model.fakl = pyo.Var(model.fak_index, within=pyo.Binary,initialize=0)

    def obj_rule(m):
        return sum(ch*m.yh[h] for h in h_i)+ \
            sum(cl*m.xal[a,l] for l in l_i for a in d_ae0_i if a[0]==0)+ \
                sum(cr*cc(a[0],a[2],data.d_m)*m.xal[a,l] for l in l_i for a in d_ae_i)

    model.obj = pyo.Objective(rule = obj_rule, sense = pyo.minimize)

    @model.Constraint(h_i,l_i)
    def hub(m,h,l):
        return sum(m.xal[a,l] for a in d_ae0_i if a[0]==h) <= m.yh[h]

    @model.Constraint(ae_i,k_i,l_i)
    def link(m,i,ti,j,tj,k,l):
        okn = data.k.oid[k]
        dkn = data.k.did[k]
        if (i,ti,j,tj) not in ae_ik[k]:
            return pyo.Constraint.Skip
        if i in h_i and i == j:
            return pyo.Constraint.Skip
        elif i == okn == j:
            return pyo.Constraint.Skip
        elif j == dkn == i:
            return pyo.Constraint.Skip
        else:
            return m.fakl[(i,ti,j,tj),k,l] <= m.xal[(i,ti,j,tj),l]

    @model.Constraint(k_i,nt_i)
    def k_flow_cons2(m,k,n,t):
        okn = data.k.oid[k]
        okt = data.k.l[k]
        dkn = data.k.did[k]
        dkt = data.k.u[k]
        if check_tuple_in_list(ae_ik[k], (n,t)):
            return pyo.Constraint.Skip
        elif (n == okn and t==okt):
            return sum(m.fakl[a,k,l] for a in ae_ik[k] for l in l_i if a[0]==okn and a[1]==okt) == 1
        elif (n == dkn and t==dkt):
            return sum(m.fakl[a,k,l] for a in ae_ik[k] for l in l_i if a[2]==dkn and a[3]==dkt) == 1 
        else:
            return sum(m.fakl[a,k,l] for a in ae_ik[k] for l in l_i if (a[2],a[3])==(n,t)) - \
                sum(m.fakl[a,k,l] for a in ae_ik[k] for l in l_i if (a[0],a[1])==(n,t)) == 0

    @model.Constraint(k_i,nt_i,l_i) # 33
    def k_flow_cons(m,k,n,t,l):
        okn = data.k.oid[k]
        okt = data.k.l[k]
        dkn = data.k.did[k]
        dkt = data.k.u[k]
        if check_tuple_in_list(ae_ik[k], (n,t)):
            return pyo.Constraint.Skip    
        if (n == okn and t==okt) or (n == dkn and t==dkt) or n in h_i:
            return pyo.Constraint.Skip
        else:
            return sum(m.fakl[a,k,l] for a in ae_ik[k] if (a[2],a[3])==(n,t)) - \
                sum(m.fakl[a,k,l] for a in ae_ik[k] if (a[0],a[1])==(n,t)) == 0    
    
    @model.Constraint(l_i,nt_i)
    def l_flow_cons(m,l,n,t):
        return sum(m.xal[a,l] for a in ae0_i if (a[2],a[3]) == (n,t)) - \
            sum(m.xal[a,l] for a in ae0_i if (a[0],a[1]) == (n,t)) == 0

    @model.Constraint(l_i)
    def lorry_use(m,l):
        return sum(m.xal[a,l] for a in d_ae0_i if a[0]==0) <= 1

    @model.Constraint(l_i)
    def l_maxt(m,l):
        return sum(tt(a[0],a[2],data.tt_m,dt)*m.xal[a,l] for a in ae_i) <= wtl

    @model.Constraint(d_ae_i)
    def l_cap(m,i,ti,j,tj):
        if check_in_list(ae_ik,(i,ti,j,tj)):
            return pyo.Constraint. Skip
        return sum(data.k.nk[k]*m.fak[i,ti,j,tj,k] for k in k_i if (i,ti,j,tj) in ae_ik[k]) <= ql

    @model.Constraint(l_i)
    def Symmetry(m,l):
        if l == 0:
            return pyo.Constraint.Skip
        else:
            return sum(model.xal[a,l] for a in d_ae0_i if a[0]==0) <= sum(model.xal[a,l-1] for a in d_ae0_i if a[0]==0)

    model.c1 = pyo.Constraint(expr = sum(model.xal[a,0] for a in d_ae0_i if a[0]==0) >= 1)
        
    opt = pyo.SolverFactory('gurobi_persistent', options={'TimeLimit' : tl*3600,
                                                'Threads' : th,'SoftMemLimit' : 10})
    opt.set_instance(model)
    result = opt.solve(model, tee=True,load_solutions=False)
    if result.solution.Gap < np.inf:
        model.solutions.load_from(result)
        l_used = int(sum(pyo.value(model.xal[a,l])for l in l_i for a in d_ae0_i if a[0]==0))    
        h_used = int(sum(pyo.value(model.yh[h]) for h in h_i))
        of1 = ch*int(sum(pyo.value(model.yh[h]) for h in h_i))
        of2 = int(sum(cl*pyo.value(model.xal[a,l])for l in l_i for a in d_ae0_i if a[0]==0))
        of3 = sum(cr*cc(a[0],a[2],data.d_m)*pyo.value(model.xal[a,l]) for l in l_i for a in d_ae_i)
        of = of1 + of2 + of3
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
    return [dt,len(k_i),len(data.n),
                    q, data.k.u.max(), len(it),var,cons,solved,status,cpu_t,
                    of1,of2,of3,of,ub,lb,gap,l_used,h_used,
                    ch,cl,cr]