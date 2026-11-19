"""
Randomised greedy constructive heuristic (article 2, Section 4.3.3 and Appendix C).

Each request is a pickup-and-delivery pair (origin, destination, quantity, release, due).
A route is started with a random request; then the request with the cheapest feasible
insertion (random among the `rcl` cheapest) is inserted, until none fits; then a new
route is started. Insertion feasibility uses forward time slacks (Savelsbergh, 1992;
Masson et al., 2013), so each test is O(1) after an O(n) update of the route.

Feasibility of a route: service at a pickup not before the release time, service at a
delivery not after the due time, travel time <= working time, and (optional) capacity.
There is no service time; the depot has zero travel time and cost.

Split commodities (`split_requests`): a commodity that went through hubs h1, ..., hm in
the relaxed solution becomes m + 1 requests o -> h1 -> ... -> d, and its time window is
divided in equal parts. Leg i is delivered at the hub before leg i + 1 is released, so
the legs are synchronised whatever routes serve them.
"""
import random
import time
from dataclasses import dataclass

INF = float('inf')


@dataclass
class Request:
    rid: int
    k: int            # original commodity
    leg: int          # position in the commodity's chain
    o: int
    d: int
    q: float
    release: float
    due: float


def requests_from_instance(inst, hub_seqs=None):
    """One request per commodity, or one per leg if `hub_seqs[k]` lists hubs to go through."""
    reqs = []
    for k, row in inst.k.iterrows():
        stops = [int(row.oid)] + list((hub_seqs or {}).get(k, [])) + [int(row.did)]
        l, u = float(row.l), float(row.u)
        n = len(stops) - 1
        width = (u - l) / n
        legs = [(stops[i], stops[i + 1], l + i * width, l + (i + 1) * width) for i in range(n)]
        if n > 1 and any(a == b or r + inst.tau(a, b) > dd for a, b, r, dd in legs):
            legs = [(int(row.oid), int(row.did), l, u)]      # the split does not fit: keep k whole
        for i, (a, b, r, dd) in enumerate(legs):
            reqs.append(Request(len(reqs), k, i, a, b, float(row.nk), r, dd))
    return reqs


class Route:
    def __init__(self, inst, working_time, capacity):
        self.inst, self.W, self.Q = inst, working_time, capacity
        self.stops = []           # (node, request, is_pickup)
        self.update()

    # --- route data -------------------------------------------------------------
    def earliest(self, stop):
        return stop[1].release if stop[2] else 0.0

    def latest(self, stop):
        r = stop[1]
        return r.due - self.inst.tau(r.o, r.d) if stop[2] else r.due

    def update(self):
        n, tau = len(self.stops), self.inst.tau
        self.S, self.wait, self.load = [0.0] * n, [0.0] * n, [0.0] * n
        self.travel, self.cost = 0.0, 0.0
        for i, st in enumerate(self.stops):
            arrive = self.earliest(st) if i == 0 else self.S[i - 1] + tau(self.stops[i - 1][0], st[0])
            self.S[i] = max(arrive, self.earliest(st))
            self.wait[i] = self.S[i] - arrive
            if i > 0:
                self.travel += tau(self.stops[i - 1][0], st[0])
                self.cost += self.inst.c(self.stops[i - 1][0], st[0])
            self.load[i] = (self.load[i - 1] if i else 0.0) + (st[1].q if st[2] else -st[1].q)
        self.F = [0.0] * n
        for i in range(n - 1, -1, -1):
            own = self.latest(self.stops[i]) - self.S[i]
            self.F[i] = own if i == n - 1 else min(own, self.wait[i + 1] + self.F[i + 1])

    def _arrival_delay(self, j, start_prev, node_prev):
        """Delay at old stop j when its predecessor becomes `node_prev` served at `start_prev`."""
        st = self.stops[j]
        new_start = max(start_prev + self.inst.tau(node_prev, st[0]), self.earliest(st))
        return max(0.0, new_start - self.S[j])

    # --- insertion ----------------------------------------------------------------
    def best_insertion(self, r):
        """(extra cost, i, j): pickup before old stop i, delivery before old stop j (j >= i)."""
        tau, c, n = self.inst.tau, self.inst.c, len(self.stops)
        best = (INF, None, None)
        if self.Q is not None and r.q > self.Q:
            return best
        for i in range(n + 1):
            prev = self.stops[i - 1] if i > 0 else None
            sp = r.release if prev is None else max(self.S[i - 1] + tau(prev[0], r.o), r.release)
            if sp > r.due - tau(r.o, r.d):
                if prev is not None and self.S[i - 1] > r.due:
                    break
                continue
            # delivery right after the pickup
            sd = sp + tau(r.o, r.d)
            if sd <= r.due:
                extra_t = tau(r.o, r.d) + (tau(prev[0], r.o) if prev else 0)
                extra_c = c(r.o, r.d) + (c(prev[0], r.o) if prev else 0)
                if i < n:
                    extra_t += tau(r.d, self.stops[i][0]) - (tau(prev[0], self.stops[i][0]) if prev else 0)
                    extra_c += c(r.d, self.stops[i][0]) - (c(prev[0], self.stops[i][0]) if prev else 0)
                ok = self.travel + extra_t <= self.W + 1e-9
                if ok and i < n:
                    ok = self._arrival_delay(i, sd, r.d) <= self.F[i] + 1e-9
                if ok and self.Q is not None:
                    ok = (self.load[i - 1] if i > 0 else 0.0) + r.q <= self.Q
                if ok and extra_c < best[0]:
                    best = (extra_c, i, i)
            if i == n:
                continue
            # delivery later: walk the delay of the pickup through stops i, i+1, ...
            delay = self._arrival_delay(i, sp, r.o)
            add_t = tau(r.o, self.stops[i][0]) + (tau(prev[0], r.o) - tau(prev[0], self.stops[i][0]) if prev else 0)
            add_c = c(r.o, self.stops[i][0]) + (c(prev[0], r.o) - c(prev[0], self.stops[i][0]) if prev else 0)
            max_load = (self.load[i - 1] if i > 0 else 0.0)
            for j in range(i + 1, n + 1):
                pj = self.stops[j - 1]
                if self.S[j - 1] + delay > self.latest(pj) + 1e-9:
                    break                                   # stop j-1 is late: later j are worse
                max_load = max(max_load, self.load[j - 1])
                if self.Q is not None and max_load + r.q > self.Q + 1e-9:
                    break
                sd = self.S[j - 1] + delay + tau(pj[0], r.d)
                if sd <= r.due:
                    t2 = tau(pj[0], r.d) + (tau(r.d, self.stops[j][0]) - tau(pj[0], self.stops[j][0]) if j < n else 0)
                    c2 = c(pj[0], r.d) + (c(r.d, self.stops[j][0]) - c(pj[0], self.stops[j][0]) if j < n else 0)
                    ok = self.travel + add_t + t2 <= self.W + 1e-9
                    if ok and j < n:
                        ok = self._arrival_delay(j, sd, r.d) <= self.F[j] + 1e-9
                    if ok and add_c + c2 < best[0]:
                        best = (add_c + c2, i, j)
                if j < n:                                   # delay carried to stop j
                    delay = max(0.0, delay - self.wait[j]) if delay > 0 else 0.0
        return best

    def insert(self, r, i, j):
        self.stops.insert(i, (r.o, r, True))
        self.stops.insert(j + 1, (r.d, r, False))
        self.update()


def construct(inst, reqs, working_time, capacity, rng, rcl=3):
    todo = list(reqs)
    rng.shuffle(todo)
    routes = []
    while todo:
        route = Route(inst, working_time, capacity)
        first = todo.pop(rng.randrange(len(todo)))
        route.insert(first, 0, 0)
        while todo:
            options = []
            for r in todo:
                cost, i, j = route.best_insertion(r)
                if i is not None:
                    options.append((cost, r.rid, r, i, j))
            if not options:
                break
            options.sort(key=lambda t: (t[0], t[1]))
            _, _, r, i, j = options[rng.randrange(min(rcl, len(options)))]
            route.insert(r, i, j)
            todo.remove(r)
        routes.append(route)
    return routes


def to_solution(routes):
    """Common solution format (see solution.py)."""
    sol_routes, legs = {}, {}
    for l, route in enumerate(routes):
        visits, where = [], {}
        for idx, (node, r, is_pick) in enumerate(route.stops):
            arrive = route.S[idx] - route.wait[idx]
            visits.append({'node': node, 'arrive': arrive, 'depart': route.S[idx]})
            where[r.rid, is_pick] = idx
        sol_routes[l] = visits
        for node, r, is_pick in route.stops:
            if is_pick:
                legs.setdefault(r.k, []).append((r.leg, {'vehicle': l, 'pickup': where[r.rid, True],
                                                         'drop': where[r.rid, False]}))
    legs = {k: [leg for _, leg in sorted(chain, key=lambda t: t[0])] for k, chain in legs.items()}
    return {'routes': sol_routes, 'legs': legs}


def rgch(inst, reqs, working_time=480, capacity=None, iterations=50, time_limit=60, seed=0,
         evaluate=None):
    """Best of `iterations` randomised constructions. `evaluate(sol) -> (cost, errors)`."""
    rng = random.Random(seed)
    best, best_cost, t0, done = None, INF, time.time(), 0
    for _ in range(iterations):
        sol = to_solution(construct(inst, reqs, working_time, capacity, rng))
        cost, errors = evaluate(sol)
        if errors:
            raise RuntimeError(f'RGCH built an infeasible solution: {errors[:3]}')
        if cost < best_cost:
            best, best_cost = sol, cost
        done += 1
        if time.time() - t0 > time_limit:
            break
    return best, best_cost, done
