"""
Relax-Solve-and-Fix (article 2, Algorithm 1) with a total time budget.

Stages
0. RGCH on the original commodities: a first upper bound, used to size the fleet of the
   relaxation (an optimal solution uses at most (UB - routing lower bound) / vehicle cost
   vehicles, so the relaxation stays a relaxation).
1. Interval relaxation -> lower bound and relaxed routes.                (lines 2-4)
2. Feasibility LP on the relaxed routes; if feasible, UB = their cost.  (lines 5-8)
3. RGCH on the commodities split at the hubs of the relaxed solution.   (lines 9, 11)
4. RUBS: TE model on a coarse grid restricted to the relaxed arcs.      (lines 10, 12)
UB = best verified solution (lines 13-14). Every candidate is checked by solution.check.

Time: stage 0 and 3 get at most `heuristic_time` each; the relaxation gets `lb_share` of
what remains; RUBS gets the rest. Time left after RUBS goes back to the relaxation: its
Gurobi search resumes where it stopped (5. below), which can raise the LB and, if it finds a
better relaxed solution, gives one more candidate through the feasibility LP.
"""
import math
import time
from dataclasses import asdict, dataclass, replace

from extract import commodity_paths, hub_sequences, legs_from_paths, vehicle_walks
from feasibility import schedule
from interval_relaxation import RelaxationParams, resume, solve as solve_relaxation
from rgch import requests_from_instance, rgch
from rubs import solve_rubs
from solution import check, summary


@dataclass
class PipelineParams:
    total_time: float = 10800
    lb_share: float = 0.7
    heuristic_time: float = 120
    rgch_iterations: int = 50
    rubs_delta: float = 60
    seed: int = 0


def routing_lower_bound(inst):
    """Every origin of a commodity is left by an arc and every destination is entered by one
    (routes start at origins or hubs and end at destinations or hubs, never the other way)."""
    nodes = inst.nodes
    out_o = sum(min(inst.c(o, n) for n in nodes if n != o) for o in set(inst.k['oid']))
    in_d = sum(min(inst.c(n, d) for n in nodes if n != d) for d in set(inst.k['did']))
    return max(out_o, in_d)


def run(inst, rp: RelaxationParams, pp: PipelineParams, solver='gurobi', log=True, fleet_from_ub=True):
    t_start = time.time()
    left = lambda: pp.total_time - (time.time() - t_start)
    kw = dict(working_time=rp.working_time, capacity=rp.capacity,
              hub_cost=rp.hub_cost, vehicle_cost=rp.vehicle_cost)
    evaluate = lambda sol: check(inst, sol, **kw)
    record = {'stages': {}, 'candidates': {}}
    best = {'cost': math.inf, 'source': None, 'solution': None}

    def offer(name, sol):
        cost, errors = evaluate(sol)
        record['candidates'][name] = {'cost': cost, 'feasible': not errors, 'errors': errors[:5]}
        if not errors and cost < best['cost']:
            best.update(cost=cost, source=name, solution=sol)

    # 0. RGCH on the original commodities
    t0 = time.time()
    sol0, cost0, it0 = rgch(inst, requests_from_instance(inst), rp.working_time, rp.capacity,
                            pp.rgch_iterations, min(pp.heuristic_time, 0.05 * pp.total_time),
                            pp.seed, evaluate)
    offer('rgch_original', sol0)
    record['stages']['rgch_original'] = {'time': time.time() - t0, 'iterations': it0, 'cost': cost0}

    # fleet of the relaxation
    if fleet_from_ub:
        fleet = int((best['cost'] - routing_lower_bound(inst)) // rp.vehicle_cost)
        rp = replace(rp, n_vehicles=max(1, fleet))
    record['fleet'] = rp.n_vehicles

    # 1. interval relaxation
    rp = replace(rp, time_limit=max(1.0, pp.lb_share * (left() - pp.heuristic_time)))
    lb_res, m = solve_relaxation(inst, rp, solver=solver, log=log)
    lb_res.pop('solution', None)
    record['stages']['relaxation'] = lb_res
    lb = lb_res['lb']
    if 'infeasible' in lb_res['status'].lower():
        raise RuntimeError(f'The relaxation with {rp.n_vehicles} vehicles is infeasible; '
                           'the fleet is too small (use a larger --vehicles).')

    if not lb_res['has_solution'] and left() > 10:
        # no relaxed routes yet: the UB stages need them, so give the relaxation the rest of the time
        more = resume(m, left() - 5)
        if more is not None:
            record['stages']['relaxation_resumed'] = more
            lb = max(lb, more['lb'])
            lb_res = more

    if lb_res['has_solution']:
        walks, arc_visits = vehicle_walks(m)
        paths = commodity_paths(m, inst)
        legs = legs_from_paths(paths, arc_visits)

        # 2. feasibility LP
        t0 = time.time()
        routes = schedule(inst, walks, legs) if legs is not None else None
        record['stages']['feasibility'] = {'time': time.time() - t0,
                                           'checked': legs is not None, 'feasible': routes is not None}
        if routes is not None:
            offer('relaxation_routes', {'routes': routes, 'legs': legs})

        optimal = best['cost'] <= lb + 1e-6
        if not optimal:
            # 3. RGCH on split commodities
            t0 = time.time()
            seqs = hub_sequences(paths, set(inst.hubs))
            reqs = requests_from_instance(inst, seqs)
            sol3, cost3, it3 = rgch(inst, reqs, rp.working_time, rp.capacity, pp.rgch_iterations,
                                    min(pp.heuristic_time, max(1.0, left())), pp.seed + 1, evaluate)
            offer('rgch_split', sol3)
            record['stages']['rgch_split'] = {'time': time.time() - t0, 'iterations': it3, 'cost': cost3,
                                              'split_commodities': sum(1 for s in seqs.values() if s)}

            # 4. RUBS
            if left() > 5:
                allowed = set()
                for l, visits in walks.items():
                    for i in range(1, len(visits)):
                        allowed.add((visits[i - 1]['node'], visits[i]['node']))
                fleet_rubs = max(len(walks), len(best['solution']['routes']))
                rp_rubs = replace(rp, n_vehicles=fleet_rubs, time_limit=left() - 2)
                rubs_res, sol4 = solve_rubs(inst, rp_rubs, allowed, pp.rubs_delta, solver, log=False)
                record['stages']['rubs'] = rubs_res
                if sol4 is not None and sol4['legs'] is not None:
                    offer('rubs', sol4)

            # 5. time left: resume the relaxation
            if best['cost'] > lb + 1e-6 and left() > 10:
                t0 = time.time()
                before = lb_res.get('relaxation_value')
                more = resume(m, left() - 5)
                if more is not None:
                    lb = max(lb, more['lb'])
                    stage = {'time': time.time() - t0, 'lb': more['lb'], 'status': more['status'],
                             'relaxation_value': more.get('relaxation_value')}
                    if more['has_solution'] and (before is None or more['relaxation_value'] < before - 1e-6):
                        walks, arc_visits = vehicle_walks(m)
                        legs = legs_from_paths(commodity_paths(m, inst), arc_visits)
                        routes = schedule(inst, walks, legs) if legs is not None else None
                        stage['feasible'] = routes is not None
                        if routes is not None:
                            offer('relaxation_routes_resumed', {'routes': routes, 'legs': legs})
                    record['stages']['relaxation_resumed'] = stage

    ub = best['cost']
    record.update({
        'lb': lb, 'ub': ub, 'ub_source': best['source'],
        'gap': (ub - lb) / ub if ub < math.inf and lb is not None else None,
        'optimal': ub < math.inf and lb is not None and ub - lb <= 1e-6 * max(1, ub),
        'total_time': time.time() - t_start,
        'pipeline_params': asdict(pp),
        'relaxation_params': asdict(rp),
        'solution': best['solution'],
        'solution_summary': summary(inst, best['solution'], **kw) if best['solution'] else None,
    })
    return record
