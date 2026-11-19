"""End-to-end checks of the upper-bound stages, solved with HiGHS (no Gurobi licence needed)."""
import copy
from pathlib import Path

from data import load_instance
from interval_relaxation import RelaxationParams
from pipeline import PipelineParams, run
from rgch import requests_from_instance, rgch
from solution import check

FOLDER = Path(__file__).resolve().parents[1] / 'instances'


def test_rgch_with_split_commodities_is_feasible_and_checked():
    inst = load_instance(FOLDER, 6)
    seqs = {k: [inst.hubs[0]] for k in inst.commodities}
    reqs = requests_from_instance(inst, seqs)
    sol, cost, _ = rgch(inst, reqs, iterations=3, evaluate=lambda s: check(inst, s))
    assert check(inst, sol)[1] == []
    split = [k for k, chain in sol['legs'].items() if len(chain) > 1]
    assert split, 'no commodity was split'
    bad = copy.deepcopy(sol)
    first = bad['legs'][split[0]][0]
    visit = bad['routes'][first['vehicle']][first['drop']]
    visit['arrive'] = visit['depart'] = 1000          # arrives at the hub too late
    assert check(inst, bad)[1], 'the checker missed a late transfer'


def test_pipeline_closes_instance_2():
    inst = load_instance(FOLDER, 2)
    rec = run(inst, RelaxationParams(), PipelineParams(total_time=120), solver='highs', log=False)
    assert rec['solution_summary']['feasible']
    assert rec['lb'] <= rec['ub'] + 1e-6
    assert rec['optimal'] and abs(rec['ub'] - 2472) < 1e-6
