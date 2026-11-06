# -*- coding: utf-8 -*-
from data import (DataContainer,data_read,SolContainer,lb_solution_read,tt,cc)
from FCheck import FS_Check
import pyomo.environ as pyo
import numpy as np
np.float_ = np.float64
import pandas as pd
import sys


ii = sys.argv[1]
gid = sys.argv[2]
intervals = sys.argv[3]
# ii = 5
# gid = 12 
data_folder = 'cbl_inst'
folder_path = 'solution_int_lb'
rs_path = 'solution_int_ub'
intervals = 1

ch = 100
cl = 500
cr = 2
tl = 3  
th = 2
dt = 15
wtl = 480
ql = 100
l = 8

data = DataContainer(data_read(gid,data_folder))
lb_solution = SolContainer(lb_solution_read(gid, folder_path, intervals))
feasible = FS_Check(gid, dt, data, lb_solution)

def ae_exists(i,ti,j,tj,tt_m,dt = 1):
    if i == 0 and j !=0 and ti ==0:
        return True
    elif j == 0 and i !=0 and tj > 0:
        return True
    elif i == j and ti + dt == tj and i != 0:
        return True
    elif tj == ti + tt(i, j, data.tt_m, dt) and i != 0 and j != 0:
        return True
    return False

def a_in_solution(i,j,data,solution_arcs):
    h_i = list(data.n[data.n['Type']=='H'].index)
    d_i = list(data.n[data.n['Type']=='D'].index)
    if i == 0 and j not in d_i or j == 0:
        return True
    if i in h_i or j in h_i:
        return True
    if (i,j) in solution_arcs:
        return True
    return False

solution_arcs = [(a.a0,a.a3) for i, a in lb_solution.x.iterrows()]
# for i,d in data.k.iterrows():
#     solution_arcs.append((d.oid,d.did))


TH = data.k.u.max()
it = [t for t in range(0,TH+dt,dt)]
n_i = list(data.n.index)
n0_i = n_i.copy()
n0_i.append(0)
nt0_i = [(0,t) for t in [it[0],TH+dt]]
nt0_i.extend([(n,t) for n in n_i for t in it])
nt_i = [n for n in nt0_i if n[0]!=0]
h_i = list(data.n[data.n['Type']=='H'].index)
k_i = list(data.k.index)
l_i = list(range(l))
ae0_i_1 = [(*n,*m) for n in nt0_i for m in nt0_i if n!=m and ae_exists(*n, *m, data.tt_m,dt)]
ae0_i = [a for a in ae0_i_1 if a_in_solution(a[0],a[2],data,solution_arcs)]
ae_i = [a for a in ae0_i if a[0] != 0 and a[2] != 0]
d_ae0_i = [a for a in ae0_i if a[0] != a[2]]
d_ae_i = [a for a in ae_i if a[0] != a[2]]
h_ae0_i = [a for a in ae0_i if a[0] == a[2]]
h_ae_i = [a for a in ae_i if a[0] == a[2]]

model = pyo.ConcreteModel(name="te_interval")
model.xal = pyo.Var(ae0_i,l_i, within=pyo.Binary,initialize=0)
model.yh = pyo.Var(h_i, within=pyo.Binary,initialize=0)
model.fakl = pyo.Var(ae_i,k_i,l_i, within=pyo.Binary,initialize=0)

def obj_rule(m):
    return sum(ch*m.yh[h] for h in h_i)+ \
        sum(cl*m.xal[a,l] for l in l_i for a in d_ae0_i if a[0]==0)+ \
        sum(cr*cc(a[0],a[2],data.d_m)*m.xal[a,l] for l in l_i for \
            a in d_ae_i)
            
model.obj = pyo.Objective(rule = obj_rule, sense = pyo.minimize)

@model.Constraint(h_i,l_i)
def hub(m,h,l):
    return sum(m.xal[a,l] for a in d_ae0_i if a[0]==h) <= m.yh[h]

@model.Constraint(ae_i,k_i,l_i)
def link(m,i,ti,j,tj,k,l):
    okn = data.k.oid[k]
    dkn = data.k.did[k]
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
    if (n == okn and t==okt):
        return sum(m.fakl[a,k,l] for a in ae_i for l in l_i if a[0]==okn and t == okt) == 1
    elif (n == dkn and t==dkt):
        return sum(m.fakl[a,k,l] for a in ae_i for l in l_i if a[2]==dkn and t == dkt) == 1 
    else:
        return sum(m.fakl[a,k,l] for a in ae_i for l in l_i if (a[2],a[3])==(n,t)) - \
            sum(m.fakl[a,k,l] for a in ae_i for l in l_i if (a[0],a[1])==(n,t)) == 0

@model.Constraint(k_i,nt_i,l_i) # 33
def k_flow_cons(m,k,n,t,l):
    okn = data.k.oid[k]
    okt = data.k.l[k]
    dkn = data.k.did[k]
    dkt = data.k.u[k]
    if (n == okn and t==okt) or (n == dkn and t==dkt) or n in h_i:
        return pyo.Constraint.Skip
    else:
        return sum(m.fakl[a,k,l] for a in ae_i if (a[2],a[3])==(n,t)) - \
            sum(m.fakl[a,k,l] for a in ae_i if (a[0],a[1])==(n,t)) == 0

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
import os
if not os.path.isdir(f'{rs_path}'):
    try:
      os.makedirs(f'{rs_path}')
    except OSError as error:
      print(error)

if result.solution.Gap < np.inf:
    model.solutions.load_from(result)
    l_used = int(sum(pyo.value(model.xal[a,l])for l in l_i for a in d_ae0_i if a[0]==0))    
    h_used = int(sum(pyo.value(model.yh[h]) for h in h_i))
    of1 = ch*int(sum(pyo.value(model.yh[h]) for h in h_i))
    of2 = int(sum(cl*pyo.value(model.xal[a,l])for l in l_i for a in d_ae0_i if a[0]==0))
    of3 = sum(cr*cc(a[0],a[2],data.d_m)*pyo.value(model.xal[a,l]) for l in l_i for a in d_ae_i)
    of = of1 + of2 + of3
    sol_ae = [(a,l) for l in l_i for a in ae0_i if pyo.value(model.xal[a,l]>0)]
    sol_ae_pd = pd.DataFrame(sol_ae, columns=['ae','l'])
    sol_ae_pd.to_csv(f'{rs_path}/{intervals}_int_cbl_{gid}_l.csv')
    sol_aek = [(a,k,l) for k in k_i for a in ae_i for l in l_i if pyo.value(model.fakl[a,k,l]>0)]
    sol_aek_pd = pd.DataFrame(sol_aek, columns=['ae','k','l'])
    sol_aek_pd.to_csv(f'{rs_path}/{intervals}_int_cbl_{gid}_k.csv')
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

import csv

if not os.path.isfile(f'{rs_path}/summary.csv'):
    try:
      with open(f'{rs_path}/summary.csv', 'w') as f:
          write = csv.writer(f)
          write.writerow(['job_id','gid','k','n','total_q','TH','int','model','variables','constraints','solved',
                          'status','time','ch','cl','cr','ofv','ub','lb','gap_%','l_used',
                          'hubs_used',
                          'h','l','r','F'])
    except OSError as error:
      print(error)
with open(f'{rs_path}/summary.csv', 'a') as f:
    write = csv.writer(f)
    write.writerow([ii,gid,len(k_i),len(data.n),
                    q, data.k.u.max(), len(it),
                    'int',var,cons,solved,status,cpu_t,
                    of1,of2,of3,of,ub,lb,gap,l_used,h_used,
                    ch,cl,cr,feasible[0]])

import time
time.sleep(60)