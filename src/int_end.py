# -*- coding: utf-8 -*-
from data import (DataContainer,data_read,sol_read,SolContainer)
from FCheck import FS_Check
from ofndp_mips import ofndp_int
import numpy as np
np.float_ = np.float64
from RGCH import rgch
import csv
import sys
from mip_dt_3 import mip_e_solution
 
# folder_path = 'cbl/cbl_inst'
# rs_path = 'cbl/solution_int_end'
folder_path = '../cbl_inst'
rs_path = 'solution_int_end'
import os
if not os.path.isdir(f'{rs_path}'):
    try:
      os.makedirs(f'{rs_path}')
    except OSError as error:
      print(error)

# ii = sys.argv[1]
# gid = sys.argv[2]
ii = 2
gid = 2
data = DataContainer(data_read(gid,folder_path))
initial_solution =  rgch(50, data, 15)[0]
mip_int = ofndp_int(ii,gid,data, 2, rs_path)
if mip_int.problem.upper_bound < np.inf:
    s = SolContainer(sol_read(gid, rs_path, 'int'))
    solution_arcs = [(a.a0, a.a3) for i, a in s.x.iterrows()]
    mip_feasibility = FS_Check(gid, 15, data, s)
    if mip_feasibility[0] == 'infeasible' and len(mip_feasibility[1]) > 0:
        n_dk = data.k.copy()
        ind = int(len(n_dk))
        for i in mip_feasibility[1]:
            k = int(i[1])
            h = int(i[0])
            o = int(n_dk.loc[k].oid)
            d = int(n_dk.loc[k].did)
            nk = int(n_dk.loc[k].nk)
            l = int(n_dk.loc[k].l)
            u = int(n_dk.loc[k].u)
            u2 = int(l + (u-l)/2)
            l2 = int(u - (u-l)/2)
            n_dk.loc[ind] = [h,d,nk,0,0,l2,u]
            n_dk.loc[k].did = h
            n_dk.loc[k].u = u2
            ind += 1
        ndata = DataContainer((n_dk,data.n,data.tt_m,data.d_m))
        heuristic_solution = rgch(50, ndata, 15)[0] + 100 * len(set([i[0] for i in mip_feasibility[1]]))
        feasible = 0
        hs_2 = mip_e_solution(data, 60, 8, solution_arcs)
    elif mip_feasibility[0] == 'infeasible' and len(mip_feasibility[1]) == 0:
        heuristic_solution = np.inf
        hs_2 = mip_e_solution(data, 60, 8, solution_arcs)
        feasible = 0
    else:
        heuristic_solution = mip_int.problem.upper_bound
        feasible = 1
else:
    feasible = 0
    heuristic_solution = np.inf
    hs_2 = mip_e_solution(data, 60, 8, [])


if not os.path.isfile(f'{rs_path}/h_summary.csv'):
    try:
      with open(f'{rs_path}/h_summary.csv', 'w') as f:
          write = csv.writer(f)
          write.writerow(['job_id','gid','k','n','feasible','initial_solution',
                          'heuristic_solution','lb'])
    except OSError as error:
      print(error)
with open(f'{rs_path}/h_summary.csv', 'a') as f:
    write = csv.writer(f)
    write.writerow([ii,gid,len(data.k),len(data.n),
                    feasible,initial_solution,heuristic_solution,
                    mip_int.problem.lower_bound])