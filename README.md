# CRO-FDP: Relax-Solve-and-Fix (corrected implementation)

Corrected implementation of the solution method of

> Ocampo-Giraldo, D., Anaya-Arenas, A. M., Klibi, W., & Ortmann, J. (2026).
> Collaborative routing orchestration for organic food distribution.
> *Transportation Research Part E*, 214, 105060. https://doi.org/10.1016/j.tre.2026.105060

used as a benchmark for the logic-based Benders decomposition of
[vtsp-lbbd](https://github.com/dmocampog/vtsp-lbbd) (article 3). The changes will be
merged into the article's public repository (`janoschortmann/orchestration-organic-food`).

## Method (Algorithm 1 of the article)

`src/pipeline.py` runs the stages within a total time budget (`--total-time`):

0. **RGCH** on the original commodities (`src/rgch.py`): first upper bound; it also bounds
   the fleet of the relaxation, (UB - routing lower bound) / vehicle cost.
1. **Interval-based relaxation** (`src/interval_relaxation.py`): lower bound.
2. **Feasibility LP** (B.4–B.10, `src/feasibility.py`): if the relaxed routes can be
   scheduled, they are a solution with cost = LB, so it is optimal.
3. **RGCH on the split commodities**: legs o → hub → d with the time window split evenly.
4. **RUBS** (`src/rubs.py`): TE model on a 60-min grid restricted to the relaxed arcs and the
   hub arcs, with conservative rounding.

UB = best candidate. Every candidate is checked by `src/solution.py` (times, windows, transfers
at hubs, working time, capacity) and its cost is recomputed from the routes.

## Differences with the published code

- Vehicle capacity is optional (`--capacity none`, the setting of article 3).
- Exact travel times in the relaxation, so the lower bound is valid for the
  continuous-time problem.
- A commodity may wait at its origin, destination and hubs (constraint 31 of the TE model).
- The relaxed solution is checked with the LP of Appendix B; the split commodities are built
  correctly (several hubs allowed); RUBS imposes the constraints it declares; the best
  solution is verified and saved.
- Fleet size, costs and time limits are parameters; one JSON per run in `results/<tag>/`.
- Instances: the 100 public instances in `instances/` (same files as vtsp-lbbd: anonymous
  ids, no locations, shortest-path matrices), read by column name, with the same cost and
  time conventions as the LBBD (`ceil(2 * distance)`, minutes). See `instances/README.md`.

## Setup

```bash
conda env create -f environment.yml
conda activate cro-fdp-rsf
```

Without conda (e.g. on the cluster), a virtual environment in the repository works too:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Gurobi needs a licence (`grbgetkey` or a `gurobi.lic` file); HiGHS works without one.

## Running

```bash
python scripts/run_rsf.py --instance 6 --tag test --total-time 600
python scripts/run_rsf.py --instance 2 --tag test --solver highs --total-time 120   # without Gurobi
pytest -q
python scripts/summarize_results.py --tags test
```

Experiments on the cluster (total time 3 h, 6 h and 9 h per instance). The jobs inherit the
environment of the shell that submits them:

```bash
source .venv/bin/activate
module load gurobi/<version>              # the modules used on the server
bash scripts/slurm/smoke.sh               # instances 2 and 6, 10 min each
bash scripts/slurm/submit_experiments.sh  # rsf_3h, rsf_6h, rsf_9h
python scripts/summarize_results.py --tags rsf_3h rsf_6h rsf_9h
```

Instances are read from `instances/` (or `--folder`).
