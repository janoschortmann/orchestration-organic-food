"""
Reduced upper bound solution (RUBS, article 2, Section 4.3.3, "Valid solution 2").

The TE model (23)-(33) on a coarse grid of step `delta` (4 x 15 = 60 min in the article),
restricted to the physical arcs used by the relaxed routes plus every arc to or from a
hub. Rounding is conservative, so every solution of this model is feasible in continuous
time: travel times are rounded up to a multiple of delta (at least delta), release times
are rounded up and due times rounded down.

The model is built by `interval_relaxation.build_model` on a time-expanded network
(each "interval" is one grid point), so both stages share the same constraints. Travel
arcs move forward in time, so no sub-tour cuts are needed.
"""
import math
import time

import pyomo.environ as pyo

from extract import commodity_paths, legs_from_paths, vehicle_walks
from interval_relaxation import DEPOT, IntervalNetwork, build_model


def build_te_network(inst, delta, allowed=None):
    """Time-expanded network on the grid {0, delta, ...}; `allowed` = set of physical arcs (m, n)."""
    horizon = float(inst.k['u'].max())
    n_points = int(math.floor(horizon / delta)) + 1
    T = range(n_points)
    hubs = set(inst.hubs)
    step = lambda m, n: 0 if m == n else max(1, math.ceil(inst.tau(m, n) / delta - 1e-9))
    net = IntervalNetwork(delta, n_points, [], [], [], [], [])
    phys = inst.nodes
    net.nodes = [(n, t) for n in phys for t in T]
    for m in phys:
        for n in phys:
            if m == n or (allowed is not None and (m, n) not in allowed and m not in hubs and n not in hubs):
                continue
            for t in T:
                if t + step(m, n) < n_points:
                    net.travel_arcs.append(((m, t), (n, t + step(m, n))))
    net.wait_arcs = [((n, t), (n, t + 1)) for n in phys for t in T if t + 1 < n_points]
    net.depot_out = [(DEPOT, (n, t)) for n in phys if n not in inst.destinations for t in T]
    net.depot_in = [((n, t), DEPOT) for n in phys if n not in inst.origins for t in T]

    for k, row in inst.k.iterrows():
        o, d = int(row.oid), int(row.did)
        rel = math.ceil(float(row.l) / delta - 1e-9)
        due = math.floor(float(row.u) / delta + 1e-9)
        net.k_source[k], net.k_sink[k] = (o, rel), (d, due)
        early = lambda n: rel + step(o, n)
        late = lambda n: due - step(n, d)
        net.k_travel[k] = [a for a in net.travel_arcs
                           if a[1][0] != o and a[0][0] != d
                           and a[0][1] >= early(a[0][0]) and a[1][1] <= late(a[1][0])]
        free, on_board = [], []
        for a in net.wait_arcs:
            n, t = a[0]
            if t >= early(n) and t + 1 <= late(n):
                (free if n in (o, d) or n in hubs else on_board).append(a)
        net.k_wait_free[k], net.k_wait_vehicle[k] = free, on_board
    return net


def solve_rubs(inst, params, allowed, delta=60, solver='gurobi', log=False):
    """Return (result dict, solution in the common format or None)."""
    t0 = time.time()
    net = build_te_network(inst, delta, allowed)
    for k in inst.commodities:
        if net.k_sink[k][1] < net.k_source[k][1]:
            return {'status': 'window_too_short_for_grid', 'time': time.time() - t0}, None
    params = _copy(params, valid_inequalities=False, aggregate_commodities=False)
    try:
        m = build_model(inst, net, params)
    except ValueError as err:
        return {'status': f'not_built: {err}', 'time': time.time() - t0}, None
    if solver == 'gurobi':
        opt = pyo.SolverFactory('gurobi_persistent')
        opt.set_instance(m)
        opt.set_gurobi_param('TimeLimit', max(1.0, params.time_limit))
        opt.set_gurobi_param('Threads', params.threads)
        opt.set_gurobi_param('SoftMemLimit', params.mem_gb)
        res = opt.solve(m, tee=log, load_solutions=False)
        grb = opt._solver_model
        out = {'status': str(res.solver.termination_condition), 'bound': grb.ObjBound if grb.SolCount else None,
               'has_solution': grb.SolCount > 0}
        if out['has_solution']:
            opt.load_vars()
    else:
        from pyomo.contrib.appsi.solvers import Highs
        opt = Highs()
        opt.config.time_limit = max(1.0, params.time_limit)
        opt.config.load_solution = False
        res = opt.solve(m)
        out = {'status': str(res.termination_condition), 'has_solution': res.best_feasible_objective is not None}
        if out['has_solution']:
            res.solution_loader.load_vars()
    out['time'] = time.time() - t0
    out['n_variables'] = m.nvariables()
    if not out['has_solution']:
        return out, None
    out['model_value'] = pyo.value(m.obj)
    walks, arc_visits = vehicle_walks(m)
    legs = legs_from_paths(commodity_paths(m, inst), arc_visits)
    routes = {l: [{'node': v['node'],
                   'arrive': v['te_nodes'][0][1] * delta,
                   'depart': v['te_nodes'][-1][1] * delta} for v in visits]
              for l, visits in walks.items()}
    return out, {'routes': routes, 'legs': legs}


def _copy(params, **changes):
    from dataclasses import replace
    return replace(params, **changes)
