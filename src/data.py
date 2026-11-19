"""
Instance reader for the CRO-FDP (article 2) and the VTSP (article 3).

Reads the `cbl_<id>.csv` format by column name, so it works with the original
files (with latitude/longitude) and with the public files of vtsp-lbbd (without
locations, with a size class `type`).

Conventions are the same as `load_data` in vtsp-lbbd, so both methods solve
exactly the same instance:
- routing cost of arc (i, j) = ceil(routing_factor * distance)
- travel time of arc (i, j) = ceil(tt / dt) * dt  (dt = 1: exact minutes)
- the dummy depot has zero cost and zero travel time.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Instance:
    name: str
    size_class: str
    k: pd.DataFrame              # one row per commodity: oid, did, nk, l, u
    n: pd.DataFrame              # one row per node: Type (O, D, H)
    tt: dict = field(repr=False)     # (i, j) -> travel time (minutes)
    cost: dict = field(repr=False)   # (i, j) -> routing cost
    dist: dict = field(repr=False)   # (i, j) -> distance (as read)

    @property
    def nodes(self):
        return list(self.n.index)

    @property
    def hubs(self):
        return list(self.n.index[self.n['Type'] == 'H'])

    @property
    def origins(self):
        return list(self.n.index[self.n['Type'] == 'O'])

    @property
    def destinations(self):
        return list(self.n.index[self.n['Type'] == 'D'])

    @property
    def commodities(self):
        return list(self.k.index)

    def tau(self, i, j):
        """Travel time; zero for i == j."""
        return 0 if i == j else self.tt[i, j]

    def c(self, i, j):
        """Routing cost; zero for i == j."""
        return 0 if i == j else self.cost[i, j]


def _read_block(file_path, header, nrows, wanted):
    return pd.read_csv(file_path, header=header, nrows=nrows, index_col=0,
                       usecols=lambda col: col in wanted or str(col).startswith('Unnamed'))


def _matrix_to_dict(df):
    df = df.rename(columns={c: int(c) for c in df.columns})
    df.index = [int(i) for i in df.index]
    return {(int(i), int(j)): float(df.loc[i, j]) for i in df.index for j in df.columns if i != j}


def load_instance(folder, instance_id, dt=1, routing_factor=2):
    file_path = Path(folder) / f"cbl_{instance_id}.csv"
    if not file_path.is_file():
        raise FileNotFoundError(f"Instance file not found: {file_path}")

    head = pd.read_csv(file_path, nrows=1)
    num_k, num_n = int(head['k'][0]), int(head['n'][0])
    size_class = str(head['type'][0]) if 'type' in head.columns else ''

    k = _read_block(file_path, 2, num_k, ('oid', 'did', 'nk', 'l', 'u'))
    k = k.apply(pd.to_numeric)
    k[['oid', 'did']] = k[['oid', 'did']].astype(int)
    k.index = [int(i) for i in k.index]

    n = _read_block(file_path, 3 + num_k, num_n, ('Type',))
    n.index = [int(i) for i in n.index]

    d_m = pd.read_csv(file_path, header=4 + num_k + num_n, nrows=num_n, index_col=0)
    tt_m = pd.read_csv(file_path, header=5 + num_k + 2 * num_n, nrows=num_n, index_col=0)

    dist = _matrix_to_dict(d_m)
    cost = {a: float(np.ceil(v * routing_factor)) for a, v in dist.items()}
    tt = {a: float(np.ceil(v / dt) * dt) for a, v in _matrix_to_dict(tt_m).items()}

    inst = Instance(str(instance_id), size_class, k, n, tt, cost, dist)
    _check(inst)
    return inst


def _check(inst):
    nodes = set(inst.nodes)
    for kk, row in inst.k.iterrows():
        if row.oid not in nodes or row.did not in nodes:
            raise ValueError(f"Commodity {kk}: origin or destination is not a node.")
        if row.l + inst.tau(row.oid, row.did) > row.u:
            raise ValueError(f"Commodity {kk}: direct trip does not fit in its time window.")
    missing = [(i, j) for i in nodes for j in nodes if i != j and (i, j) not in inst.tt]
    if missing:
        raise ValueError(f"Travel-time matrix is missing arcs, e.g. {missing[:3]}")
