"""
Read routes and commodity paths out of a solved network model (interval relaxation or RUBS).

For each vehicle the selected arcs form a closed walk from the depot; it is ordered with
Hierholzer's algorithm and cut into visits (consecutive network nodes with the same
physical node form one visit). For each commodity the selected arcs form a path from its
source to its sink; it is cut into legs (maximal runs on the same vehicle).
"""
from collections import defaultdict

import pyomo.environ as pyo

from interval_relaxation import DEPOT


def _closed_walk(arcs):
    """Order arcs (tail, head) into one walk that starts and ends at the depot."""
    out = defaultdict(list)
    for a in arcs:
        out[a[0]].append(a)
    stack, walk = [(DEPOT, None)], []
    while stack:
        node, arc = stack[-1]
        if out[node]:
            nxt = out[node].pop()
            stack.append((nxt[1], nxt))
        else:
            stack.pop()
            if arc is not None:
                walk.append(arc)
    walk.reverse()
    if len(walk) != len(arcs):
        raise RuntimeError('Vehicle arcs do not form one closed walk from the depot.')
    return walk


def vehicle_walks(m):
    """Visits of each vehicle and, for each of its arcs, the visit at its tail and at its head.

    Returns ({l: [{'node': n, 'te_nodes': [...]}, ...]}, {l: {arc: (tail_visit, head_visit)}})."""
    walks, arc_visits = {}, {}
    for l in m._L:
        arcs = [m._arcs[a] for a in range(len(m._arcs)) if pyo.value(m.x[a, l]) > 0.5]
        if not arcs:
            continue
        visits, where = [], {}
        for t, h in _closed_walk(arcs):
            current = len(visits) - 1
            if h == DEPOT:
                where[t, h] = (current, None)
            elif t == DEPOT or t[0] != h[0]:
                visits.append({'node': h[0], 'te_nodes': [h]})
                where[t, h] = (None if t == DEPOT else current, current + 1)
            else:                                   # waiting arc: same visit
                visits[-1]['te_nodes'].append(h)
                where[t, h] = (current, current)
        walks[l], arc_visits[l] = visits, where
    return walks, arc_visits


def commodity_paths(m, inst):
    """{k: [(arc, label)]} from source to sink; label = vehicle, None (aggregated) or 'wait'."""
    net = m._net
    out = defaultdict(list)
    for idx, var in m.fv.items():
        if pyo.value(var) > 0.5:
            out[idx[1], m._arcs[idx[0]][0]].append((m._arcs[idx[0]], idx[2] if len(idx) == 3 else None))
    for (a, k), var in m.fw.items():
        if pyo.value(var) > 0.5:
            out[k, m._arcs[a][0]].append((m._arcs[a], 'wait'))
    paths = {}
    for k in inst.commodities:
        node, path = net.k_source[k], []
        while node != net.k_sink[k]:
            if not out[k, node]:
                raise RuntimeError(f'Commodity {k}: path broken at {node}.')
            arc, label = out[k, node].pop()
            path.append((arc, label))
            node = arc[1]
        paths[k] = path
    return paths


def legs_from_paths(paths, arc_visits):
    """Cut each path into legs {k: [{'vehicle', 'pickup', 'drop'}]}; None if vehicles are unknown."""
    legs = {}
    for k, path in paths.items():
        chain, run = [], []
        for arc, label in path + [(None, 'end')]:
            if label is None:
                return None                         # aggregated model: no vehicle per arc
            if run and label != run[0][1]:
                if run[0][1] != 'wait':
                    l = run[0][1]
                    first, last = arc_visits[l][run[0][0]], arc_visits[l][run[-1][0]]
                    chain.append({'vehicle': l, 'pickup': first[0], 'drop': last[1]})
                run = []
            run.append((arc, label))
        legs[k] = chain
    return legs


def hub_sequences(paths, hubs):
    """Hubs where each commodity changes vehicle or waits, in order (used to split commodities)."""
    seqs = {}
    for k, path in paths.items():
        seq, prev = [], None
        for arc, label in path:
            node = arc[0][0]
            if prev is not None and label != prev and node in hubs and (not seq or seq[-1] != node):
                seq.append(node)
            prev = label
        seqs[k] = seq
    return seqs
