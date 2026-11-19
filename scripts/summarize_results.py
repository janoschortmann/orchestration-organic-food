"""
Table of results by instance and by size class.

    python scripts/summarize_results.py --tags rsf_3h rsf_6h rsf_9h
Writes results/summary_<tags>.csv and prints the mean by size class.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(tag):
    rows = []
    for f in sorted((ROOT / 'results' / tag).glob('rsf_*.json')):
        r = json.loads(f.read_text())
        rows.append({'tag': tag, 'instance': r['instance'], 'type': r['type'], 'K': r['K'],
                     'lb': r['lb'], 'ub': r['ub'], 'gap_%': None if r['gap'] is None else 100 * r['gap'],
                     'optimal': r['optimal'], 'ub_source': r['ub_source'], 'fleet': r['fleet'],
                     'vehicles': (r['solution_summary'] or {}).get('vehicles'),
                     'time_s': r['total_time']})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tags', nargs='+', required=True)
    args = p.parse_args()
    df = pd.DataFrame([row for tag in args.tags for row in load(tag)])
    if df.empty:
        print('No results found.')
        return
    out = ROOT / 'results' / f"summary_{'_'.join(args.tags)}.csv"
    df.sort_values(['tag', 'instance']).to_csv(out, index=False)
    by_class = df.groupby(['tag', 'type']).agg(n=('instance', 'count'), optimal=('optimal', 'sum'),
                                               gap_mean=('gap_%', 'mean'), time_mean=('time_s', 'mean'))
    print(by_class.round(2).to_string())
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()
