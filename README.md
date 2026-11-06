# Collaborative Routing Orchestration for Organic Food Distribution

Companion code for the article:

> Ocampo-Giraldo, D., Anaya-Arenas, A. M., Klibi, W., & Ortmann, J. (2026).
> Collaborative routing orchestration for organic food distribution.
> *Transportation Research Part E: Logistics and Transportation Review*, 214, 105060.
> https://doi.org/10.1016/j.tre.2026.105060

> **Status:** this repository is being organized. The code below is the research code used for the article, uploaded as-is. A cleaner structure, documentation and tests will follow.

## The problem

Several small producers share their delivery operations. Each request must be picked up at an origin and delivered to a destination within a time window. Producers can coordinate routes and use shared transfer hubs, where loads are exchanged between vehicles. The goal is to decide which hubs to open and how vehicles are routed and synchronized, minimizing hub, vehicle and routing costs.

The models are formulated on time-expanded networks and solved with Gurobi through Pyomo.

## Repository structure

```
orchestration-organic-food/
├── src/            research code (see below)
├── instances/      instance files (see instances/README.md)
├── experiments/    scripts to reproduce the article's results (to come)
├── results/        output folder (not tracked)
├── requirements.txt
└── CITATION.cff
```

### Main files in `src/`

| File | What it does |
| --- | --- |
| `data.py` | Reads instances and solutions (`load_data`, `data_read`, `sol_read`, `DataContainer`, `SolContainer`) and provides travel-time and cost helpers (`tt`, `cc`). Solution readers were reconstructed and should be checked against the original runs. |
| `ofndp_mips.py` | Interval-based time-expanded MIP (`ofndp_int`), with subtour-elimination lazy cuts via a Gurobi callback. Gives a lower bound. |
| `mip_dt_3.py` | Fully time-expanded MIP (`mip_e_solution`), optionally restricted to the arcs of a given solution. |
| `mipc.py` | Compact (non time-expanded) MIP with big-M synchronization at hubs, used as a benchmark. Reads a different instance format. |
| `FCheck.py` | Feasibility check (`FS_Check`): verifies that a solution can be scheduled in time, including synchronization at hubs. |
| `RGCH.py` | Randomized greedy construction heuristic (`rgch`) based on cheapest insertion, with time-window and capacity checks. Can be run on its own: `python RGCH.py --instance_id 2`. |
| `int_end.py` | Main pipeline: heuristic, interval MIP, feasibility check and repair. Writes a summary CSV. |
| `solution_ub.py`, `solution_ub2.py` | Compute upper bounds by re-solving restricted models from a lower-bound solution. |

## Requirements

- Python 3.10+
- Gurobi with a valid license (free academic licenses are available from Gurobi)
- Python packages in `requirements.txt`

```bash
pip install -r requirements.txt
```

## Running an instance

From the `src/` folder:

```bash
cd src
python int_end.py
```

Instance and output paths, and the instance id, are currently set at the top of each script. Command-line arguments and a configuration file will be added.

## Citation

If you use this code, please cite the article above. GitHub's "Cite this repository" button uses `CITATION.cff`.

## License

MIT License. See `LICENSE`.
