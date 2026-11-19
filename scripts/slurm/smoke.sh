#!/bin/bash
# Quick check before the long runs: instances 2 and 6, 5 minutes each, on the testing
# partition (limit 15 min). From the root of the repository, with .venv active and the
# modules loaded:
#   bash scripts/slurm/smoke.sh               # partition testing
#   bash scripts/slurm/smoke.sh optimum       # another partition
mkdir -p logs
PARTITION=${1:-testing}
sbatch --partition="$PARTITION" --array=1,5 --time=00:14:00 --job-name=rsf_smoke \
    scripts/slurm/run_array.sh rsf_smoke 300
