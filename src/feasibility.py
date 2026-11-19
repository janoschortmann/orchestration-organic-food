"""
LP that tries to undo the interval projection (article 2, Appendix B.4, (B.4)-(B.10)).

Given the routes (ordered visits) and the legs of every commodity taken from the
interval relaxation, look for arrival times e and departure times b such that
- b[l, i] + tau(n_i, n_i+1) = e[l, i+1]                   (B.6)
- e[l, i] <= b[l, i], and every time in [0, H]             (B.7)
- the pickup at the origin is not before the release time  (B.10)
- the delivery is not after the due time                   (B.9)
- at a transfer, the first vehicle arrives at the hub
  before the second one leaves it                          (B.5)
If the LP is feasible, the relaxed routes are a feasible solution with the same cost.
"""
import pyomo.environ as pyo
from pyomo.contrib.appsi.base import TerminationCondition
from pyomo.contrib.appsi.solvers import Highs


def schedule(inst, walks, legs):
    """Return {l: [{'node', 'arrive', 'depart'}]} or None if no schedule exists."""
    horizon = float(inst.k['u'].max())
    idx = [(l, i) for l, visits in walks.items() for i in range(len(visits))]
    m = pyo.ConcreteModel(name='feasibility_lp')
    m.e = pyo.Var(idx, bounds=(0, horizon))
    m.b = pyo.Var(idx, bounds=(0, horizon))
    m.c = pyo.ConstraintList()
    for l, visits in walks.items():
        for i, v in enumerate(visits):
            m.c.add(m.e[l, i] <= m.b[l, i])
            if i > 0:
                m.c.add(m.b[l, i - 1] + inst.tau(visits[i - 1]['node'], v['node']) == m.e[l, i])
    for k, chain in legs.items():
        row = inst.k.loc[k]
        for n, leg in enumerate(chain):
            l = leg['vehicle']
            if n == 0:
                m.c.add(m.b[l, leg['pickup']] >= float(row.l))
            else:
                prev = chain[n - 1]
                m.c.add(m.e[prev['vehicle'], prev['drop']] <= m.b[l, leg['pickup']])
            if n == len(chain) - 1:
                m.c.add(m.e[l, leg['drop']] <= float(row.u))
    # (B.4): earliest schedule (any feasible point would do)
    m.obj = pyo.Objective(expr=sum(m.b[i] for i in idx), sense=pyo.minimize)
    opt = Highs()
    opt.config.load_solution = False
    res = opt.solve(m)
    if res.termination_condition != TerminationCondition.optimal:
        return None
    res.solution_loader.load_vars()
    return {l: [{'node': v['node'], 'arrive': pyo.value(m.e[l, i]), 'depart': pyo.value(m.b[l, i])}
                for i, v in enumerate(visits)] for l, visits in walks.items()}
