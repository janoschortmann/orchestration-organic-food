"""
Interval-based relaxation of the CRO-FDP (article 2, Section 4.3.1 and Appendix B).

The horizon [0, H] is split into `n_intervals` intervals of equal length. All moments
of a physical node that fall in the same interval are glued into one node (n, j).
Solving the resulting MIP gives a lower bound (LB) on the continuous-time problem.

Differences with the published code (all needed for the LB to be valid):
- Travel times are exact (no rounding up to delta), so the LB bounds the
  continuous-time problem solved by the LBBD of article 3.
- A commodity may wait at its origin, at its destination and at hubs without a
  vehicle (constraint 31 of the TE model). The published code forced the pickup in
  the interval of the release time and the delivery in the interval of the due time.
- The fleet size is a parameter; the LB is only valid if it is at least the number
  of vehicles of an optimal solution.
- Capacity is optional (`capacity=None` ignores it) and, when given, is imposed.
- Sub-tours are removed with connectivity cuts, which stay valid when a route visits
  the same node twice inside one interval.

Two rules decide which problem is relaxed (`hub_rule`):
- 'commodity' (default, same problem as the LBBD of article 3): a hub must be opened
  only if a commodity goes through it; vehicles may visit a hub several times.
- 'vehicle' (TE model of article 2, constraint 24): a vehicle may visit a hub only
  if it is open, at most once.
"""
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import pyomo.environ as pyo

DEPOT = 's'


@dataclass
class RelaxationParams:
    n_intervals: int = 2
    n_vehicles: int = 5
    hub_cost: float = 100
    vehicle_cost: float = 500
    working_time: float = 480
    capacity: float = None          # None: no capacity constraint
    hub_rule: str = 'commodity'      # 'commodity' (LBBD) or 'vehicle' (article 2)
    aggregate_commodities: bool = False   # True: f without vehicle index (weaker, smaller)
    valid_inequalities: bool = True  # (B.2) and (B.3)
    time_limit: float = 10800
    threads: int = 2
    mem_gb: float = 10


@dataclass
class IntervalNetwork:
    """Nodes (n, j), vehicle arcs and, for each commodity, the arcs it may use."""
    length: float
    n_intervals: int
    nodes: list
    travel_arcs: list            # ((m, i), (n, j)), m != n, both physical nodes
    wait_arcs: list              # ((n, j), (n, j + 1))
    depot_out: list              # (DEPOT, (n, j))
    depot_in: list               # ((n, j), DEPOT)
    k_source: dict = field(default_factory=dict)   # k -> (o_k, j)
    k_sink: dict = field(default_factory=dict)     # k -> (d_k, j)
    k_travel: dict = field(default_factory=dict)   # k -> travel arcs usable by k
    k_wait_free: dict = field(default_factory=dict)     # k -> wait arcs without vehicle (o_k, d_k, hubs)
    k_wait_vehicle: dict = field(default_factory=dict)  # k -> wait arcs on board of a vehicle

    def start(self, j):
        return j * self.length

    def end(self, j):
        return (j + 1) * self.length

    def interval_of(self, t):
        return min(int(t // self.length), self.n_intervals - 1)


def build_network(inst, n_intervals):
    horizon = float(inst.k['u'].max())
    net = IntervalNetwork(horizon / n_intervals, n_intervals, [], [], [], [], [])
    J = range(n_intervals)
    s, e = net.start, net.end
    phys = inst.nodes
    hubs = set(inst.hubs)

    net.nodes = [(n, j) for n in phys for j in J]
    # Closed intervals: keeping an arc that is not needed only weakens the relaxation.
    for m in phys:
        for n in phys:
            if m == n:
                continue
            tau = inst.tau(m, n)
            for i in J:
                for j in J:
                    if j >= i and s(i) + tau <= e(j) and e(i) + tau >= s(j):
                        net.travel_arcs.append(((m, i), (n, j)))
    net.wait_arcs = [((n, j), (n, j + 1)) for n in phys for j in J if j + 1 < n_intervals]
    net.depot_out = [(DEPOT, (n, j)) for n in phys if n not in inst.destinations for j in J]
    net.depot_in = [((n, j), DEPOT) for n in phys if n not in inst.origins for j in J]

    for k, row in inst.k.iterrows():
        o, d, l, u = int(row.oid), int(row.did), float(row.l), float(row.u)
        net.k_source[k] = (o, net.interval_of(l))
        net.k_sink[k] = (d, net.interval_of(u))
        travel = []
        for (m, i), (n, j) in net.travel_arcs:
            if n == o or m == d:
                continue
            tau = inst.tau(m, n)
            lo = max(s(i), l + inst.tau(o, m), s(j) - tau)
            hi = min(e(i), u - tau - inst.tau(n, d), e(j) - tau)
            if lo <= hi:
                travel.append(((m, i), (n, j)))
        free, on_board = [], []
        for (n, j), (n2, j2) in net.wait_arcs:
            earliest = max(s(j), l + inst.tau(o, n))
            if earliest <= e(j) and earliest <= min(e(j2), u - inst.tau(n, d)) and s(j2) <= u - inst.tau(n, d):
                (free if n in (o, d) or n in hubs else on_board).append(((n, j), (n2, j2)))
        net.k_travel[k], net.k_wait_free[k], net.k_wait_vehicle[k] = travel, free, on_board
    return net


def build_model(inst, net, p):
    """MIP on the interval network. Arcs are indexed by integer ids (position in `m._arcs`)."""
    m = pyo.ConcreteModel(name='interval_relaxation')
    L = list(range(p.n_vehicles))
    K = inst.commodities
    H = inst.hubs
    hubs = set(H)

    arcs = net.travel_arcs + net.wait_arcs + net.depot_out + net.depot_in
    aid = {a: i for i, a in enumerate(arcs)}
    travel = [aid[a] for a in net.travel_arcs]
    depot_out = [aid[a] for a in net.depot_out]
    tau = {aid[a]: inst.tau(a[0][0], a[1][0]) for a in net.travel_arcs}
    cost = {aid[a]: inst.c(a[0][0], a[1][0]) for a in net.travel_arcs}
    m._arcs, m._L, m._net = arcs, L, net

    m.x = pyo.Var(range(len(arcs)), L, domain=pyo.Binary)
    m.y = pyo.Var(H, domain=pyo.Binary)

    # Commodity variables: on board of a vehicle (fv) or waiting without a vehicle (fw).
    board = {k: [aid[a] for a in net.k_travel[k] + net.k_wait_vehicle[k]] for k in K}
    free = {k: [aid[a] for a in net.k_wait_free[k]] for k in K}
    board_idx = [(a, k) for k in K for a in board[k]]
    if p.aggregate_commodities:
        m.fv = pyo.Var(board_idx, domain=pyo.Binary)
        fv_sum = lambda a, k: m.fv[a, k]
    else:
        m.fv = pyo.Var([(a, k, l) for a, k in board_idx for l in L], domain=pyo.Binary)
        fv_sum = lambda a, k: sum(m.fv[a, k, l] for l in L)
    m.fw = pyo.Var([(a, k) for k in K for a in free[k]], domain=pyo.Binary)

    # Objective (23): hubs, vehicles, routing.
    m.obj = pyo.Objective(sense=pyo.minimize, expr=
        sum(p.hub_cost * m.y[h] for h in H)
        + sum(p.vehicle_cost * m.x[a, l] for a in depot_out for l in L)
        + sum(cost[a] * m.x[a, l] for a in travel for l in L))

    # Vehicle flow (29) and one departure from the depot (30).
    v_in, v_out = defaultdict(list), defaultdict(list)
    for i, (t, h) in enumerate(arcs):
        v_out[t].append(i)
        v_in[h].append(i)
    V = range(len(net.nodes))
    m.vehicle_flow = pyo.Constraint(V, L, rule=lambda m, v, l:
        sum(m.x[a, l] for a in v_in[net.nodes[v]]) == sum(m.x[a, l] for a in v_out[net.nodes[v]]))
    m.one_departure = pyo.Constraint(L, rule=lambda m, l: sum(m.x[a, l] for a in depot_out) <= 1)

    # Working time (33): travel time of each vehicle.
    m.working_time = pyo.Constraint(L, rule=lambda m, l:
        sum(tau[a] * m.x[a, l] for a in travel) <= p.working_time)

    # Commodity flow (25)-(27): supply at the source node, demand at the sink node.
    k_in, k_out = defaultdict(list), defaultdict(list)
    for k in K:
        for a in board[k]:
            k_out[k, arcs[a][0]].append(('v', a))
            k_in[k, arcs[a][1]].append(('v', a))
        for a in free[k]:
            k_out[k, arcs[a][0]].append(('w', a))
            k_in[k, arcs[a][1]].append(('w', a))

    def term(k, kind, a):
        return fv_sum(a, k) if kind == 'v' else m.fw[a, k]

    def k_flow_rule(m, k, v):
        node = net.nodes[v]
        ends = (net.k_source[k], net.k_sink[k])
        if not k_in[k, node] and not k_out[k, node]:
            if node in ends:
                raise ValueError(f'Commodity {k} has no arc at {node}: the instance is infeasible.')
            return pyo.Constraint.Skip
        balance = (1 if node == ends[0] else 0) - (1 if node == ends[1] else 0)
        return (sum(term(k, *t) for t in k_out[k, node])
                - sum(term(k, *t) for t in k_in[k, node]) == balance)
    m.commodity_flow = pyo.Constraint(K, V, rule=k_flow_rule)

    # Linking (31): a commodity on board needs the vehicle on the arc.
    if p.aggregate_commodities:
        m.link = pyo.Constraint(board_idx, rule=lambda m, a, k: m.fv[a, k] <= sum(m.x[a, l] for l in L))
    else:
        m.link = pyo.Constraint([(a, k, l) for a, k in board_idx for l in L],
                                rule=lambda m, a, k, l: m.fv[a, k, l] <= m.x[a, l])

        # Vehicle-specific conservation (28) outside hubs, origin and destination.
        def k_vehicle_rule(m, k, v, l):
            node = net.nodes[v]
            if node[0] in hubs or node[0] in (net.k_source[k][0], net.k_sink[k][0]):
                return pyo.Constraint.Skip
            ins = [a for kind, a in k_in[k, node] if kind == 'v']
            outs = [a for kind, a in k_out[k, node] if kind == 'v']
            if not ins and not outs:
                return pyo.Constraint.Skip
            return sum(m.fv[a, k, l] for a in ins) == sum(m.fv[a, k, l] for a in outs)
        m.commodity_vehicle_flow = pyo.Constraint(K, V, L, rule=k_vehicle_rule)

    # Hubs.
    if p.hub_rule == 'vehicle':
        # (24): a vehicle visits a hub at most once and only if the hub is open.
        into = {h: [a for a in travel + depot_out if arcs[a][1][0] == h] for h in H}
        m.hub = pyo.Constraint(H, L, rule=lambda m, h, l: sum(m.x[a, l] for a in into[h]) <= m.y[h])
    elif p.hub_rule == 'commodity':
        # A commodity can enter a hub only if the hub is open (as in the LBBD master).
        idx = [(a, k) for k in K for a in board[k] if a in tau and arcs[a][1][0] in hubs]
        m.hub = pyo.Constraint(idx, rule=lambda m, a, k: fv_sum(a, k) <= m.y[arcs[a][1][0]])
    else:
        raise ValueError(f"Unknown hub_rule: {p.hub_rule}")

    # Capacity (32), only if requested.
    if p.capacity is not None:
        q = {k: float(inst.k.loc[k, 'nk']) for k in K}
        on_arc = defaultdict(list)
        for a, k in board_idx:
            on_arc[a].append(k)
        cap_arcs = [a for a in travel if on_arc[a]]
        if p.aggregate_commodities:
            m.capacity = pyo.Constraint(cap_arcs, rule=lambda m, a:
                sum(q[k] * m.fv[a, k] for k in on_arc[a]) <= p.capacity * sum(m.x[a, l] for l in L))
        else:
            m.capacity = pyo.Constraint(cap_arcs, L, rule=lambda m, a, l:
                sum(q[k] * m.fv[a, k, l] for k in on_arc[a]) <= p.capacity)

    # Valid inequalities (B.2) and (B.3).
    if p.valid_inequalities:
        vertical = defaultdict(list)
        for a in travel:
            if arcs[a][0][1] == arcs[a][1][1]:
                vertical[arcs[a][0][1]].append(a)
        m.vertical_time = pyo.Constraint(list(vertical), L, rule=lambda m, j, l:
            sum(tau[a] * m.x[a, l] for a in vertical[j]) <= net.length)
        m.commodity_time = pyo.Constraint(K, rule=lambda m, k:
            sum(tau[a] * fv_sum(a, k) for a in board[k] if a in tau)
            <= float(inst.k.loc[k, 'u'] - inst.k.loc[k, 'l']))

    # Symmetry breaking (21) and (22).
    m.order = pyo.Constraint(L, rule=lambda m, l:
        sum(m.x[a, l] for a in depot_out) <= sum(m.x[a, l - 1] for a in depot_out)
        if l > 0 else pyo.Constraint.Skip)
    m.first_vehicle = pyo.Constraint(expr=sum(m.x[a, 0] for a in depot_out) >= 1)

    m.cuts = pyo.ConstraintList()
    return m


def find_subtours(m, value):
    """(vehicle, node set, arc id) for each closed component of a route not linked to the depot."""
    found = []
    for l in m._L:
        used = [a for a in range(len(m._arcs)) if value(m.x[a, l]) > 0.5]
        if not used:
            continue
        parent = {}

        def root(v):
            parent.setdefault(v, v)
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v
        for a in used:
            t, h = m._arcs[a]
            parent[root(t)] = root(h)
        comps = defaultdict(set)
        for a in used:
            t, h = m._arcs[a]
            comps[root(t)].update((t, h))
        for nodes in comps.values():
            if DEPOT in nodes:
                continue
            inside = next(a for a in used if m._arcs[a][0] in nodes)
            found.append((l, frozenset(nodes), inside))
    return found


def add_connectivity_cut(m, nodes, arc):
    """For every vehicle: if it uses `arc` (inside `nodes`), it must also leave `nodes`."""
    leaving = [a for a, (t, h) in enumerate(m._arcs) if t in nodes and h not in nodes]
    return [m.cuts.add(sum(m.x[a, l] for a in leaving) >= m.x[arc, l]) for l in m._L]


def solve(inst, p, solver='gurobi', log=True):
    t0 = time.time()
    net = build_network(inst, p.n_intervals)
    m = build_model(inst, net, p)
    stats = {'cuts': 0, 'callbacks': 0, 'build_time': time.time() - t0,
             'n_variables': m.nvariables(), 'n_constraints': m.nconstraints(),
             'interval_length': net.length}
    if solver == 'gurobi':
        res = _solve_gurobi(m, p, stats, log)
    elif solver == 'highs':
        res = _solve_highs(m, p, stats)
    else:
        raise ValueError(f'Unknown solver: {solver}')
    res.update(stats)
    res['total_time'] = time.time() - t0
    if res['has_solution']:
        res['solution'] = extract_solution(m, inst)
    return res, m


def _solve_gurobi(m, p, stats, log):
    from gurobipy import GRB
    opt = pyo.SolverFactory('gurobi_persistent')
    opt.set_instance(m)
    opt.set_gurobi_param('TimeLimit', p.time_limit)
    opt.set_gurobi_param('Threads', p.threads)
    opt.set_gurobi_param('SoftMemLimit', p.mem_gb)
    opt.set_gurobi_param('LazyConstraints', 1)
    x_vars = list(m.x.values())

    def callback(cb_m, cb_opt, where):
        if where != GRB.Callback.MIPSOL:
            return
        stats['callbacks'] += 1
        cb_opt.cbGetSolution(vars=x_vars)
        for _, nodes, arc in find_subtours(m, pyo.value):
            for con in add_connectivity_cut(m, nodes, arc):
                cb_opt.cbLazy(con)
                stats['cuts'] += 1

    opt.set_callback(callback)
    m._opt, m._log = opt, log
    return _gurobi_optimize(m)


def _gurobi_optimize(m):
    result = m._opt.solve(m, tee=m._log, load_solutions=False)
    grb = m._opt._solver_model
    out = {'status': str(result.solver.termination_condition),
           'lb': grb.ObjBound, 'has_solution': grb.SolCount > 0,
           'solve_time': grb.Runtime, 'bb_nodes': grb.NodeCount}
    if out['has_solution']:
        m._opt.load_vars()
        out['relaxation_value'] = grb.ObjVal
        out['gap'] = grb.MIPGap
        if find_subtours(m, pyo.value):
            raise RuntimeError('The final solution still has sub-tours.')
    return out


def resume(m, time_limit):
    """Continue the Gurobi search of a relaxation stopped by its time limit (same model, same tree).

    Returns the new result, or None if the model was not solved with Gurobi."""
    if getattr(m, '_opt', None) is None or time_limit < 1:
        return None
    m._opt.set_gurobi_param('TimeLimit', time_limit)
    return _gurobi_optimize(m)


def _solve_highs(m, p, stats):
    """Cut loop for small tests without Gurobi: solve, add the violated cuts, repeat."""
    opt = pyo.SolverFactory('appsi_highs')
    opt.config.time_limit = p.time_limit
    t0 = time.time()
    while True:
        stats['callbacks'] += 1
        result = opt.solve(m)
        subtours = find_subtours(m, pyo.value)
        if not subtours:
            break
        for _, nodes, arc in subtours:
            stats['cuts'] += len(add_connectivity_cut(m, nodes, arc))
    value = pyo.value(m.obj)
    return {'status': str(result.solver.termination_condition), 'lb': value,
            'relaxation_value': value, 'gap': 0.0, 'has_solution': True,
            'solve_time': time.time() - t0}


def extract_solution(m, inst):
    """Routes on the interval network, commodity arcs and open hubs (input of the next steps)."""
    arcs, net = m._arcs, m._net
    routes = {}
    for l in m._L:
        used = [arcs[a] for a in range(len(arcs)) if pyo.value(m.x[a, l]) > 0.5]
        if not used:
            continue
        succ = defaultdict(list)
        for t, h in used:
            succ[t].append(h)
        seq, node = [], DEPOT
        while succ[node]:
            node = succ[node].pop()
            if node == DEPOT:
                break
            seq.append(list(node))
        routes[l] = seq
    commodity_arcs = defaultdict(list)
    for idx, var in m.fv.items():
        if pyo.value(var) > 0.5:
            a, k = idx[0], idx[1]
            l = idx[2] if len(idx) == 3 else None
            commodity_arcs[k].append({'from': list(arcs[a][0]), 'to': list(arcs[a][1]), 'vehicle': l})
    for (a, k), var in m.fw.items():
        if pyo.value(var) > 0.5:
            commodity_arcs[k].append({'from': list(arcs[a][0]), 'to': list(arcs[a][1]), 'vehicle': 'wait'})
    return {'routes': routes,
            'commodity_arcs': dict(commodity_arcs),
            'hubs': [h for h in inst.hubs if pyo.value(m.y[h]) > 0.5],
            'vehicles_used': len(routes)}
