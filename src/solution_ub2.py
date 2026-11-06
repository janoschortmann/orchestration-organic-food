# -*- coding: utf-8 -*-
from data import (DataContainer,data_read,SolContainer,lb_solution_read,tt,cc,sol_read)
from FCheck import FS_Check
from mip_dt_3 import mip_e_solution
import numpy as np
np.float_ = np.float64
import sys


ii = sys.argv[1]
gid = sys.argv[2]
dt = int(sys.argv[3])

data_folder = 'cbl_inst'
folder_path = 'solution_int_end'
rs_path = 'solution_int_ub_2'
l = 8


data = DataContainer(data_read(gid,data_folder))
lb_solution = SolContainer(sol_read(gid, folder_path, 'int'))
feasible = FS_Check(gid, 15, data, lb_solution)
solution_arcs = [(a.a0,a.a3) for i, a in lb_solution.x.iterrows()]
ub_solution = mip_e_solution(data, dt, l, solution_arcs)

import csv
import os
if not os.path.isdir(f'{rs_path}'):
    try:
      os.makedirs(f'{rs_path}')
    except OSError as error:
      print(error)
if not os.path.isfile(f'{rs_path}/summary.csv'):
    try:
      with open(f'{rs_path}/summary.csv', 'w') as f:
          write = csv.writer(f)
          write.writerow(['job_id','gid','F','dt','k','n','q', 'TH',
                          'it','var','cons','solved','status','cpu_t',
                          'of1','of2','of3','of','ub','lb','gap','l_used','h_used',
                          'ch','cl','cr'])
    except OSError as error:
      print(error)
with open(f'{rs_path}/summary.csv', 'a') as f:
    write = csv.writer(f)
    write.writerow([ii,gid,feasible[0],*ub_solution])

import time
time.sleep(60)

