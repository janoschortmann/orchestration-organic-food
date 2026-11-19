"""Small checks of the interval relaxation, solved with HiGHS (no Gurobi licence needed)."""
from pathlib import Path

import pytest

from data import load_instance
from interval_relaxation import RelaxationParams, build_network, solve

FOLDER = Path(__file__).resolve().parents[1] / 'instances'


def small_instance(n_commodities=4):
    inst = load_instance(FOLDER, 2)
    inst.k = inst.k.iloc[:n_commodities]
    return inst


def test_commodity_can_wait_at_origin_and_destination():
    inst = small_instance()
    net = build_network(inst, 2)
    for k, row in inst.k.iterrows():
        waits = net.k_wait_free[k]
        assert any(a[0][0] == row.oid for a in waits), 'no waiting arc at the origin'
        assert any(a[0][0] == row.did for a in waits), 'no waiting arc at the destination'


def test_vehicle_index_gives_a_bound_at_least_as_strong():
    inst = small_instance(3)
    agg, _ = solve(inst, RelaxationParams(n_vehicles=2, aggregate_commodities=True), solver='highs')
    veh, _ = solve(inst, RelaxationParams(n_vehicles=2, aggregate_commodities=False), solver='highs')
    assert veh['lb'] >= agg['lb'] - 1e-6


def test_capacity_is_imposed():
    inst = small_instance(3)
    free, _ = solve(inst, RelaxationParams(n_vehicles=3, aggregate_commodities=True), solver='highs')
    tight, _ = solve(inst, RelaxationParams(n_vehicles=3, aggregate_commodities=True, capacity=10),
                     solver='highs')
    assert tight['lb'] >= free['lb'] - 1e-6
