"""
Run the Relax-Solve-and-Fix method (article 2, Algorithm 1) on one instance; one JSON per run.

Examples:
    python scripts/run_rsf.py --instance 6 --tag test --total-time 600
    python scripts/run_rsf.py --instance 2 --tag test --solver highs --total-time 120
    python scripts/run_rsf.py --task-index 0 --tag rsf_3h --total-time 10800   (Slurm array)
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from data import load_instance  # noqa: E402
from interval_relaxation import RelaxationParams  # noqa: E402
from pipeline import PipelineParams, run  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument('--instance', type=int, help='instance id (cbl_<id>.csv)')
    which.add_argument('--task-index', type=int, help='Slurm array index; instance id = index + 1')
    p.add_argument('--folder', default=str(ROOT / 'instances'),
                   help='folder with cbl_<id>.csv (default: instances/ of this repository)')
    p.add_argument('--tag', default='test', help='experiment name; results go to results/<tag>/')
    p.add_argument('--total-time', type=float, default=10800, help='time budget of the whole method (s)')
    p.add_argument('--lb-share', type=float, default=0.7,
                   help='share of the remaining time given to the relaxation; RUBS gets the rest')
    p.add_argument('--heuristic-time', type=float, default=120, help='time limit of each RGCH stage (s)')
    p.add_argument('--intervals', type=int, default=2, help='number of intervals (lambda)')
    p.add_argument('--vehicles', type=int, default=None,
                   help='fleet of the relaxation (default: bound from the first RGCH solution)')
    p.add_argument('--capacity', default='none', help="'none' or the vehicle capacity Q")
    p.add_argument('--hub-rule', choices=['commodity', 'vehicle'], default='commodity',
                   help="'commodity': same problem as the LBBD; 'vehicle': constraint (24) of article 2")
    p.add_argument('--aggregate', action='store_true',
                   help='relaxation without vehicle index on commodities (smaller, weaker; skips the LP)')
    p.add_argument('--no-valid-ineq', action='store_true', help='do not add (B.2) and (B.3)')
    p.add_argument('--rubs-delta', type=float, default=60, help='time step of RUBS (min)')
    p.add_argument('--working-time', type=float, default=480)
    p.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 2)))
    p.add_argument('--mem-gb', type=float, default=10, help='Gurobi SoftMemLimit (GB)')
    p.add_argument('--solver', choices=['gurobi', 'highs'], default='gurobi')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--quiet', action='store_true', help='no solver log')
    return p.parse_args()


def main():
    args = parse_args()
    instance_id = args.instance if args.instance is not None else args.task_index + 1
    inst = load_instance(args.folder, instance_id)

    rp = RelaxationParams(
        n_intervals=args.intervals,
        n_vehicles=args.vehicles or 1,
        working_time=args.working_time,
        capacity=None if args.capacity.lower() == 'none' else float(args.capacity),
        hub_rule=args.hub_rule,
        aggregate_commodities=args.aggregate,
        valid_inequalities=not args.no_valid_ineq,
        threads=args.threads,
        mem_gb=args.mem_gb,
    )
    pp = PipelineParams(total_time=args.total_time, lb_share=args.lb_share,
                        heuristic_time=args.heuristic_time, rubs_delta=args.rubs_delta, seed=args.seed)

    started = time.strftime('%Y-%m-%d %H:%M:%S')
    rec = run(inst, rp, pp, solver=args.solver, log=not args.quiet, fleet_from_ub=args.vehicles is None)
    record = {'instance': instance_id, 'type': inst.size_class, 'K': len(inst.commodities),
              'nodes': len(inst.nodes), 'hubs': len(inst.hubs), 'tag': args.tag,
              'started': started, 'host': platform.node(), 'solver': args.solver, **rec}

    out_dir = ROOT / 'results' / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'rsf_{instance_id}.json'
    out_file.write_text(json.dumps(record, indent=2, default=_to_json))
    gap = f"{100 * record['gap']:.2f}%" if record['gap'] is not None else '-'
    print(f"instance {instance_id} ({inst.size_class}): LB = {record['lb']:.1f}, UB = {record['ub']:.1f} "
          f"[{record['ub_source']}], gap = {gap}, {record['total_time']:.0f} s -> {out_file}")


def _to_json(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    return str(obj)


if __name__ == '__main__':
    main()
