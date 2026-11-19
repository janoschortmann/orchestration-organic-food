#!/bin/bash
# The three experiments: total time of the method 3 h, 6 h and 9 h per instance.
# From the root of the repository, with .venv active and the modules loaded:
#   bash scripts/slurm/submit_experiments.sh
mkdir -p logs
sbatch --array=0-99 --time=03:30:00 --job-name=rsf_3h scripts/slurm/run_array.sh rsf_3h 10800
sbatch --array=0-99 --time=06:30:00 --job-name=rsf_6h scripts/slurm/run_array.sh rsf_6h 21600
sbatch --array=0-99 --time=09:30:00 --job-name=rsf_9h scripts/slurm/run_array.sh rsf_9h 32400
