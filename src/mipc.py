# -*- coding: utf-8 -*-
import os
import pandas as pd
from scipy.spatial.distance import euclidean
import math
import numpy as np
np.float_ = np.float64
import networkx as nx
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
import sys
import csv
import time
import pyomo.environ as pyo

def DataRead(ind = 0, path = 'instances', lw = 480):
    inst = os.listdir(f'{path}/')
    f = f'{path}/{inst[ind]}'
    df = pd.read_csv(f,nrows=1)
    o_d = pd.read_csv(f,header=2, nrows= df.k[0])
    d_d = pd.read_csv(f,header=3+df.k[0],nrows= df.k[0],usecols=[0,1,2,3,4])
    h_data = pd.read_csv(f,header=4+2*df.k[0],nrows=7,usecols=[0,1,2,3])
    h_data.pid = h_data.pid.astype(int)
    h_data.set_index('pid',drop = False, inplace = True)
    n_data = pd.concat([o_d,d_d])
    n_data.drop(['q'],axis=1,inplace=True)
    n_data.set_index('id',drop=True,inplace=True)
    n_data.loc[0] = [60,60,0,0,lw]
    for i in h_data.index:
            n_data.loc[i]=(h_data.x[i],h_data.y[i],h_data.pid[i],0,lw)
    n_data.sort_index(inplace=True)
    n_data.pid = n_data.pid.astype(int)
    values = {"l": 15, "u": lw}
    n_data.fillna(value=values,inplace = True)
    
    return o_d, d_d, n_data, h_data, f, inst

def cc(n,m,n_data):
    if n == 0 or m == 0:
        return 0
    nn = n_data.loc[n]
    mm = n_data.loc[m]
    return euclidean([nn.y_la, nn.x_lt], [mm.y_la, mm.x_lt])

def tt(n,m,n_data,dt=1):
    if n == 0 or m == 0:
        return 0
    nn = n_data.loc[n]
    mm = n_data.loc[m]
    return math.ceil(euclidean([nn.x_lt, nn.y_la, ], [mm.x_lt, mm.y_la]) / dt)*dt
    

ii = int(sys.argv[1])
path = (sys.argv[2])
# rs_path = sys.argv[3]

# ii= 0
# path='ins_s'
rs_path = 'rs_mipc'

tl=3
th=2
wl=480
M = 10000

o_d = DataRead(ii,path)[0]
d_d = DataRead(ii,path)[1]
n_data = DataRead(ii,path)[2]
h_data = DataRead(ii,path)[3]
file = DataRead(ii,path)[4]
insss = DataRead(ii,path)[5]
gid = file.split(sep='/')[1].split(sep='_')[0]
ins_name = file.split(sep='/')[1].split(sep='.')[0]
h_i = list(h_data.index)
initial_k = len(o_d)

n0_i = list(n_data.index)
n_i =[n for n in n0_i if n!=0]
a0_i = [(i,j) for i in n0_i for j in n0_i if i != j]
a_i = [(i,j) for i in n_i for j in n_i if i != j]
n0_i_nh = [n for n in n0_i if n not in h_i]
n_i_nh = [n for n in n_i if n not in h_i]
a0_i_nh = [a for a in a0_i if a[0] not in h_i and a[1] not in h_i]
a_i_nh = [a for a in a_i if a[0] not in h_i and a[1] not in h_i]

k_i=list(o_d.index)
l_i = list(range(int(math.ceil((len(k_i)+1)/2))))

ch = 500
cl = 1000
cr = 1.5

model = pyo.ConcreteModel(name="mipc")
model.yh = pyo.Var(h_i, within=pyo.Binary)
model.xal = pyo.Var(a0_i,l_i,within=pyo.Binary)
model.falk = pyo.Var(a_i,l_i,k_i,within=pyo.Binary)
model.uhkll = pyo.Var(h_i,k_i,l_i,l_i,within=pyo.Binary)
model.enl = pyo.Var(n0_i,l_i,within=pyo.NonNegativeReals)
model.dnl = pyo.Var(n0_i,l_i,within=pyo.NonNegativeReals)
model.cut = pyo.ConstraintList()

def obj_rule(m):
    return sum(ch*m.yh[h] for h in h_i) + \
        sum(cl*m.xal[a,l] for l in l_i for a in a0_i if a[0]==0)+ \
        sum(cr*cc(a[0],a[1],n_data)*m.xal[a,l] for l in l_i for a in a_i)
model.obj = pyo.Objective(rule = obj_rule, sense = pyo.minimize)

@model.Constraint(h_i,k_i,l_i,l_i) # 2
def hubconstraint(m,h,k,l,lp):
    return m.uhkll[h,k,l,lp]<=m.yh[h]

@model.Constraint(n0_i,l_i) # 3
def lorriesconservation(m,n,l):
    return sum(m.xal[a,l] for a in a0_i if a[1]==n) - \
        sum(m.xal[a,l] for a in a0_i if a[0]==n) == 0

@model.Constraint(h_i,k_i,l_i,l_i) # 5
def hubkflowcoservation(m,h,k,l,lp):
    return sum(m.falk[a,l,k] for a in a_i if a[1]==h) + \
    sum(m.falk[a,lp,k] for a in a_i if a[0]==h) <= m.uhkll[h,k,l,lp] +1
        
@model.Constraint(n_i,k_i) # 6 7 8
def odflow(m,n,k):
    if n==d_d.id[k]:
        return sum(m.falk[a,l,k] for l in l_i for a in a_i if a[1]==n) == 1
    elif n==o_d.id[k]:
        return sum(m.falk[a,l,k] for l in l_i for a in a_i if a[0]==n) == 1
    elif n not in h_i:
        return sum(m.falk[a,l,k] for l in l_i for a in a_i if a[1]==n) - \
                sum(m.falk[a,l,k] for l in l_i for a in a_i if a[0]==n) == 0
    else:
        return pyo.Constraint.Skip

@model.Constraint(a_i,l_i,k_i) # 12
def linking(m,i,j,l,k):
    return m.falk[i,j,l,k] <= m.xal[i,j,l]

@model.Constraint(l_i) # 11
def ldepotd(m,l):
    return sum(m.xal[a,l] for a in a0_i if a[0]==0) <=1

@model.Constraint(n_i_nh) # 10
def nonhubvisits(m,n):
    return sum(m.xal[a,l] for l in l_i for a in a0_i if a[1]==n) <= 1

@model.Constraint(h_i,l_i)
def lhubvisit(m,h,l):
    return sum(m.xal[a,l] for a in a0_i if a[1]==h) <= m.yh[h]

@model.Constraint(h_i,k_i,l_i,l_i) # 14
def hubsynchro(m,h,k,l,lp):
    return m.enl[h,l] - m.dnl[h,lp] <= M*(1-m.uhkll[h,k,l,lp]) 

@model.Constraint(l_i,a0_i) # 15
def nohubvisit(m,l,i,j):
    return m.dnl[i,l] +tt(i,j,n_data)- m.enl[j,l] <= M*(1-m.xal[i,j,l])

@model.Constraint(l_i,n_i) # 20
def nonhubad(m,l,n):
    return m.enl[n,l] <= m.dnl[n,l]

@model.Constraint(l_i)
def depodepart_t(m,l): # 21
    return m.dnl[0,l]<=m.enl[0,l]

@model.Constraint(n_i_nh,l_i,k_i) # 22 23
def visittime(m,n,l,k):
    if n==o_d.id[k]:
        return m.dnl[n,l] >= o_d.l[k]
    elif n==d_d.id[k]:
        return m.enl[n,l] <= d_d.u[k]
    else:
        return pyo.Constraint.Skip

@model.Constraint(l_i) # 24
def l_maxt(m,l):
    return sum(tt(a[0], a[1], n_data)*m.xal[a,l] for a in a_i) <= wl

@model.Constraint(a_i_nh,l_i) # 25
def l_cap(m,i,j,l):
    return sum(o_d.q[k]*m.falk[i,j,l,k] for k in k_i) <= 100

def _add_cut(c,l):
    c_i = range(len(c)-1)
    return model.cut.add(sum(model.xal[c[i],c[i+1],l] for i in c_i)+ model.xal[c[-1],c[0],l]
                     <= len(c)-1)

@model.Constraint(l_i)
def Symmetry(m,l):
    if l == 0:
        return pyo.Constraint.Skip
    else:
        return sum(model.xal[a,l] for a in a0_i if a[0]==0) <= sum(model.xal[a,l-1] for a in a0_i if a[0]==0)

model.c1 = pyo.Constraint(expr = sum(model.xal[a,0] for a in a0_i if a[0]==0) >= 1)

@model.Constraint(l_i,a_i)
def valid_ineq(m,l,i,j):
    return m.xal[i,j,l] <= sum(m.xal[a,l] for a in a0_i if a[0]==0)

opt = pyo.SolverFactory('gurobi_persistent', options={'TimeLimit' : tl*3600,
                                                      'Threads' : th,'SoftMemLimit' : 10})
opt.set_instance(model)
opt.set_gurobi_param('PreCrush', 1)
opt.set_gurobi_param('LazyConstraints', 1)

from gurobipy import GRB

def my_callback(cb_m, cb_opt, cb_where):
    if cb_where == GRB.Callback.MIPSOL:
        cb_opt.cbGetSolution(vars = [model.xal[a,l] for a in a0_i for l in l_i])
        for l in l_i:
            G = nx.DiGraph()
            G.add_edges_from([a for a in a0_i if pyo.value(model.xal[a,l]>0)])
            Cycles=list(nx.simple_cycles(G))
            if len(Cycles) > 1:
                print('adding cut')
                for c in Cycles:
                    cb_opt.cbLazy(_add_cut(c,l))

opt.set_callback(my_callback)
result = opt.solve(model, tee=True, load_solutions=False)

if not os.path.isdir(f'{rs_path}/files'):
    try:
      os.makedirs(f'{rs_path}/files')
    except OSError as error:
      print(error)


if len(result.solution) > 0:
    model.solutions.load_from(result)
    l_used = int(sum(pyo.value(model.xal[a,l])for l in l_i for a in a0_i if a[0]==0))    
    h_used = int(sum(pyo.value(model.yh[h]) for h in h_i))
    of1 = ch*int(sum(pyo.value(model.yh[h]) for h in h_i))
    of2 = int(sum(cl*pyo.value(model.xal[a,l])for l in l_i for a in a0_i if a[0]==0))
    of3 = sum(cr*cc(a[0],a[1],n_data)*pyo.value(model.xal[a,l]) for l in l_i for a in a_i)
    of = of1 + of2 + of3
    ttd = sum(cc(a[0],a[1],n_data)*pyo.value(model.xal[a,l]) for l in l_i for a in a_i)
    gap = result.solution.Gap  
    sol_ae = [(a,l) for l in l_i for a in a0_i if pyo.value(model.xal[a,l]>0)]
    sol_ae_pd = pd.DataFrame(sol_ae, columns=['ae','l'])
    sol_ae_pd.to_csv(f'{rs_path}/files/mipc_{gid}_l.xlsx')
    sol_aek = [(a,k) for k in k_i for a in a_i if pyo.value(model.fak[a,k]>0)]
    sol_aek_pd = pd.DataFrame(sol_aek, columns=['ae','k'])
    sol_aek_pd.to_csv(f'{rs_path}/files/mipc_{gid}_k.xlsx')
    solved = 1
else:
    l_used = np.nan
    h_used = np.nan
    of1 = np.nan
    of2 = np.nan
    of3 = np.nan
    of = of1 + of2 + of3
    ttd = np.nan
    gap = np.nan
    ipk_k = np.nan
    solved = 0

lb = result.problem.Lower_bound
ub = result.problem.Upper_bound
cpu_t = result.solver.Wallclock_time
status = result.solver.Termination_condition
var = result.problem.Number_of_variables
cons = result.problem.Number_of_constraints
q = sum(o_d.q)

if not os.path.isfile(f'{rs_path}/summary_mipc.csv'):
    try:
      with open(f'{rs_path}/summary_mipc.csv', 'w') as f:
          write = csv.writer(f)
          write.writerow(['job_id','gid','inst','k','n','total_q','TH','int','model','variables','constraints','solved',
                          'status','time','ch','cl','cr','ofv','ub','lb','gap_%','lorrys',
                          'hubs_used','total_d'])
    except OSError as error:
      print(error)
with open(f'{rs_path}/summary_mipe_int.csv', 'a') as f:
    write = csv.writer(f)
    write.writerow([ii,gid,ins_name,len(k_i),len(o_d.pid.unique())+len(d_d.pid.unique()),
                    q, d_d.u.max(), 1,
                    'mipc',var,cons,solved,status,cpu_t,
                    of1,of2,of3,of,ub,lb,gap,l_used,h_used,ttd])
time.sleep(60)