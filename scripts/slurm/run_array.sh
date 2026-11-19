#!/bin/bash
# Slurm job array: one task = one instance (task index i -> instance i + 1).
#
# Before sbatch, in the root of the repository:
#   source .venv/bin/activate          # the job inherits this environment
#   module load gurobi/<version>       # the modules you use on the server
#   mkdir -p logs                      # Slurm opens the log file before the script runs
#
#   sbatch --array=0-99 --time=03:30:00 scripts/slurm/run_array.sh rsf_3h 10800
#
# Arguments: <tag> <total time of the method in s> [extra options for run_rsf.py]
# The Slurm --time must exceed the total time (model building and writing take a few minutes).
#
#SBATCH --job-name=rsf
#SBATCH --partition=optimum
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --export=ALL
##SBATCH --account=CHANGE_ME     # uncomment if your server requires an account

TAG=${1:-test}
TOTAL_TIME=${2:-10800}
EXTRA=${3:-}

cd "$SLURM_SUBMIT_DIR"

if ! python -c "import gurobipy, pyomo" 2>/dev/null; then
    echo "Python environment not ready: activate .venv and load the modules before sbatch." >&2
    exit 1
fi

python scripts/run_rsf.py --task-index "$SLURM_ARRAY_TASK_ID" $EXTRA \
    --tag "$TAG" --total-time "$TOTAL_TIME" \
    --threads "$SLURM_CPUS_PER_TASK" --mem-gb 14 --quiet
