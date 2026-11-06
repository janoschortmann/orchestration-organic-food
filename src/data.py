# -*- coding: utf-8 -*-
"""
Data loading and helper functions for the collaborative routing models.

Based on the original data module (DataContainer, load_data, tt, cc).
The functions data_read, sol_read, lb_solution_read and SolContainer were
reconstructed from how the scripts in this folder use them; check them
against the original version if it is found.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

np.float_ = np.float64  # compatibility with NumPy 2 for older Pyomo versions


class DataContainer:
    """
    Container for the problem data: requests (k), nodes (n),
    travel-time matrix (tt_m) and distance matrix (d_m).

    Accepts four arguments, a single tuple (k_data, n_data, tt_matrix, d_matrix)
    as returned by data_read, or another DataContainer (as returned by load_data).
    """

    def __init__(self, k_data, n_data=None, tt_matrix=None, d_matrix=None):
        if n_data is None and isinstance(k_data, DataContainer):
            k_data, n_data, tt_matrix, d_matrix = k_data.k, k_data.n, k_data.tt_m, k_data.d_m
        elif n_data is None and isinstance(k_data, (tuple, list)):
            k_data, n_data, tt_matrix, d_matrix = k_data
        self.k = k_data
        self.n = n_data
        self.tt_m = tt_matrix
        self.d_m = d_matrix

        for col in ['l', 'u']:
            if col not in self.k.columns:
                raise KeyError(
                    f"Required column '{col}' not found in the request DataFrame (k_data).")
            self.k[col] = self.k[col].astype(int)

        self.k_nodes = self._k_nodes()

    def _k_nodes(self):
        """
        DataFrame with pickup, delivery and potential hub nodes for each request,
        with their time windows.
        """
        rows = {}
        hubs_id = list(self.n.loc[self.n['Type'] == "H"].index)
        for i, k in self.k.iterrows():
            pickup_upper = k.u - tt(k.oid, k.did, self.tt_m)
            delivery_lower = k.l + tt(k.oid, k.did, self.tt_m)
            rows[f"{i}P"] = [f"{i}P", int(k.oid), int(k.l), int(pickup_upper)]
            rows[f"{i}D"] = [f"{i}D", int(k.did), int(delivery_lower), int(k.u)]
            for h in hubs_id:
                hub_pickup_lower = k.l + tt(k.oid, h, self.tt_m)
                hub_delivery_upper = k.u - tt(h, k.did, self.tt_m)
                rows[f"{i}H{h}P"] = [f"{i}H{h}P", int(h), int(hub_pickup_lower), int(hub_delivery_upper)]
                rows[f"{i}H{h}D"] = [f"{i}H{h}D", int(h), int(hub_pickup_lower), int(hub_delivery_upper)]
        return pd.DataFrame.from_dict(rows, orient='index', columns=['id', 'n_id', 'l', 'u'])

    def update_value(self, index, column, value):
        """Update a value in the k_nodes DataFrame."""
        self.k_nodes.loc[index, column] = value

    def change_demand(self, percentage_change):
        """Change the demand ('nk') of all requests by a percentage (e.g. 20 or -15)."""
        if 'nk' not in self.k.columns:
            raise KeyError("The 'nk' column (demand) was not found in the request DataFrame.")
        multiplier = 1 + (percentage_change / 100.0)
        self.k['nk'] = np.ceil(self.k['nk'] * multiplier).clip(lower=1)

    def change_time(self, minutes_change):
        """Shift the latest delivery time ('u') of all requests by a number of minutes."""
        if 'u' not in self.k.columns:
            raise KeyError("The 'u' column (upper time limit) was not found in the request DataFrame.")
        self.k['u'] = self.k['u'] + minutes_change


def _read_instance(folder_path, instance_name):
    """Read the five blocks of an instance file and return them as DataFrames."""
    file_path = Path(folder_path) / f"cbl_{instance_name}.csv"
    try:
        inst_data = pd.read_csv(str(file_path), nrows=1)
        num_requests = int(inst_data.k[0])
        num_nodes = int(inst_data.n[0])
        if num_requests <= 0 or num_nodes <= 0:
            raise ValueError("Error in the format of the data file.")

        k_data = pd.read_csv(str(file_path), header=2, nrows=num_requests,
                             index_col=0, usecols=range(8))
        for col in ['oid', 'did']:
            k_data[col] = k_data[col].astype(int)
        n_data = pd.read_csv(str(file_path), header=3 + num_requests, nrows=num_nodes,
                             index_col=0, usecols=range(6))
        d_matrix = pd.read_csv(str(file_path), nrows=num_nodes, index_col=0,
                               header=4 + num_requests + num_nodes)
        d_matrix.rename(columns={i: int(i) for i in d_matrix.columns}, inplace=True)
        tt_matrix = pd.read_csv(str(file_path), nrows=num_nodes, index_col=0,
                                header=5 + num_requests + 2 * num_nodes)
        tt_matrix.rename(columns={i: int(i) for i in tt_matrix.columns}, inplace=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file {file_path} is empty")
    except Exception as e:
        raise ValueError(f"Error while reading {file_path}. Please check the file format: {e}")
    return k_data, n_data, tt_matrix, d_matrix


def data_read(gid, folder_path):
    """Return (k_data, n_data, tt_matrix, d_matrix) for instance `gid`."""
    return _read_instance(folder_path, gid)


def load_data(folder_path, instance_name):
    """Return a DataContainer for an instance."""
    return DataContainer(*_read_instance(folder_path, instance_name))


class SolContainer:
    """
    Container for a solution read from CSV files.

    Attributes:
        x: vehicle arcs, columns ['ae', 'l', 'a0', 'a3']
           (ae = time-expanded arc, l = vehicle, a0/a3 = origin/destination node).
        f: request arcs, columns ['ae', 'k', 'a0', 'a3'] (plus 'l' if present).
    Accepts either two DataFrames or a single tuple (x, f).
    """

    def __init__(self, x, f=None):
        if f is None and isinstance(x, (tuple, list)):
            x, f = x
        self.x = x
        self.x_lt = x  # name used in the R1 version of FCheck
        self.f = f


def _parse_arcs(df, dest_index):
    """Convert the 'ae' column to tuples and add origin (a0) and destination (a3) nodes."""
    df = df.drop(columns=[c for c in df.columns if str(c).startswith('Unnamed')])
    df['ae'] = df['ae'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
    df['a0'] = df['ae'].apply(lambda a: int(a[0]))
    df['a3'] = df['ae'].apply(lambda a: int(a[dest_index]))
    return df


def _read_solution_files(path_l, path_k):
    x = pd.read_csv(path_l)
    f = pd.read_csv(path_k)
    if len(x) == 0:
        return x, f
    # Interval model arcs have 6 elements (i, ti, ti2, j, tj, tj2): destination at index 3.
    # Fully time-expanded arcs have 4 elements (i, ti, j, tj): destination at index 2.
    first = ast.literal_eval(x['ae'].iloc[0]) if isinstance(x['ae'].iloc[0], str) else x['ae'].iloc[0]
    dest_index = 3 if len(first) == 6 else 2
    return _parse_arcs(x, dest_index), _parse_arcs(f, dest_index)


def sol_read(gid, rs_path, model='int'):
    """
    Read a solution written by ofndp_int:
    {rs_path}/{model}_cbl_{gid}_l.csv and {rs_path}/{model}_cbl_{gid}_k.csv
    """
    return _read_solution_files(Path(rs_path) / f"{model}_cbl_{gid}_l.csv",
                                Path(rs_path) / f"{model}_cbl_{gid}_k.csv")


def lb_solution_read(gid, folder_path, intervals):
    """
    Read a lower-bound solution:
    {folder_path}/{intervals}_int_cbl_{gid}_l.csv and ..._k.csv
    """
    return _read_solution_files(Path(folder_path) / f"{intervals}_int_cbl_{gid}_l.csv",
                                Path(folder_path) / f"{intervals}_int_cbl_{gid}_k.csv")


def tt(i, j, tt_m, dt=1):
    """Travel time between nodes i and j, rounded up to a multiple of dt. Node 0 is the depot."""
    if i == 0:
        return 0
    if j == 0:
        return dt
    return np.ceil(tt_m.loc[i][j] / dt) * dt


def cc(i, j, d_m):
    """Travel cost (distance) between nodes i and j. Node 0 is the depot."""
    if i == 0 or j == 0:
        return 0
    return d_m.loc[i][j]
