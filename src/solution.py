"""
Common solution format and an independent feasibility check.

A solution is a dict:
    routes: {vehicle: [{'node': n, 'arrive': t, 'depart': t}, ...]}   (depot not listed)
    legs:   {k: [{'vehicle': l, 'pickup': i, 'drop': j}, ...]}        (in travel order)
The commodity k is loaded on `vehicle` at visit `pickup` and unloaded at visit `drop`
of that vehicle's route. Between two legs it waits at a hub.

`check` recomputes the cost from the routes, so every upper bound reported by the
pipeline is the cost of a solution that passed this check.
"""
EPS = 1e-6


def check(inst, sol, working_time=480, capacity=None, hub_cost=100, vehicle_cost=500):
    """Return (cost, errors). The solution is feasible if `errors` is empty."""
    errors = []
    routes, legs = sol['routes'], sol['legs']
    horizon = float(inst.k['u'].max())
    hubs = set(inst.hubs)

    routing, used_routes = 0.0, 0
    for l, visits in routes.items():
        if not visits:
            continue
        used_routes += 1
        travel = 0.0
        for i, v in enumerate(visits):
            if v['arrive'] < -EPS or v['depart'] > horizon + EPS:
                errors.append(f'vehicle {l}, visit {i}: time outside [0, {horizon}]')
            if v['arrive'] > v['depart'] + EPS:
                errors.append(f'vehicle {l}, visit {i}: departs before arriving')
            if i > 0:
                prev = visits[i - 1]
                tau = inst.tau(prev['node'], v['node'])
                travel += tau
                routing += inst.c(prev['node'], v['node'])
                if v['arrive'] < prev['depart'] + tau - EPS:
                    errors.append(f'vehicle {l}, visit {i}: arrives before {prev["depart"]} + {tau}')
        if travel > working_time + EPS:
            errors.append(f'vehicle {l}: travel time {travel:.1f} > {working_time}')

    hubs_used = set()
    load = {l: [0.0] * len(v) for l, v in routes.items()}
    for k, row in inst.k.iterrows():
        chain = legs.get(k, [])
        if not chain:
            errors.append(f'commodity {k}: not served')
            continue
        for n_leg, leg in enumerate(chain):
            l, p, d = leg['vehicle'], leg['pickup'], leg['drop']
            visits = routes.get(l, [])
            if not (0 <= p < d < len(visits)):
                errors.append(f'commodity {k}, leg {n_leg}: bad visit indices ({p}, {d}) on vehicle {l}')
                break
            for i in range(p, d):
                load[l][i] += float(row.nk)
            for i in range(p + 1, d + 1):          # arrivals with k on board
                if visits[i]['node'] in hubs:
                    hubs_used.add(visits[i]['node'])
            start, end = visits[p], visits[d]
            if n_leg == 0:
                if start['node'] != row.oid:
                    errors.append(f'commodity {k}: first leg does not start at its origin')
                if start['depart'] < row.l - EPS:
                    errors.append(f'commodity {k}: picked up before its release time')
            else:
                before = routes[chain[n_leg - 1]['vehicle']][chain[n_leg - 1]['drop']]
                if before['node'] != start['node']:
                    errors.append(f'commodity {k}: leg {n_leg} does not start where leg {n_leg - 1} ends')
                if start['node'] not in hubs:
                    errors.append(f'commodity {k}: transfer at a node that is not a hub')
                if before['arrive'] > start['depart'] + EPS:
                    errors.append(f'commodity {k}: leaves the hub before it arrives')
            if n_leg == len(chain) - 1:
                if end['node'] != row.did:
                    errors.append(f'commodity {k}: last leg does not end at its destination')
                if end['arrive'] > row.u + EPS:
                    errors.append(f'commodity {k}: delivered after its due time')

    if capacity is not None:
        for l, profile in load.items():
            if profile and max(profile) > capacity + EPS:
                errors.append(f'vehicle {l}: load {max(profile)} > capacity {capacity}')

    cost = vehicle_cost * used_routes + routing + hub_cost * len(hubs_used)
    return cost, errors


def summary(inst, sol, **kwargs):
    cost, errors = check(inst, sol, **kwargs)
    return {'cost': cost, 'feasible': not errors, 'errors': errors[:20],
            'vehicles': sum(1 for v in sol['routes'].values() if v),
            'hubs': sorted({v['node'] for vs in sol['routes'].values() for v in vs
                            if v['node'] in set(inst.hubs)})}
