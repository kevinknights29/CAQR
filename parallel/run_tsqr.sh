#!/bin/bash
#SBATCH --job-name=tsqr_scaling
#SBATCH --nodes=1
#SBATCH --ntasks=4                  # 4 MPI processes
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/tsqr_%j.out
#SBATCH --error=logs/tsqr_%j.err
#SBATCH --partition=compute

# Usage: sbatch --export=ALL,M=10000,N=100 run_tsqr.sh
M=${M:-10000}   # default fallback if not passed
N=${N:-100}

MATRIX_FILE="matrix_${M}_${N}.txt"
RESULTS_FILE="results/tsqr_results.csv"

# Sanity check: m/p >> n
P=4
RATIO=$(( M / P ))
if [ "$RATIO" -le "$N" ]; then
    echo "WARNING: m/p ($RATIO) is not >> n ($N). Results may not be meaningful."
fi

# Load modules
module purge
module load gcc/15.2.0-gcc-8.5.0-r7c4jsu mpi/latest tbb/latest compiler-rt/latest mkl/latest

# Setup directories
mkdir -p logs results

# Generate the matrix
echo "[$(date)] Generating matrix: m=$M, n=$N"
./matrix_generator "$M" "$N"

if [ ! -f "$MATRIX_FILE" ]; then
    echo "ERROR: Matrix file '$MATRIX_FILE' was not created. Aborting."
    exit 1
fi

# Run TSQR, capture output
echo "[$(date)] Launching TSQR: m=$M, n=$N, p=$P"

TSQR_OUTPUT=$(mpirun -np "$P" ./tsqr "$M" "$N" "$MATRIX_FILE" 2>&1)
EXIT_CODE=$?

echo "$TSQR_OUTPUT"  # still echoes everything to the .out log

# Parse timing from TSQR printed output
WALL_MAX=$(echo "$TSQR_OUTPUT" | grep "Max (critical path)" | awk '{print $(NF-1)}')

if [ -z "$WALL_MAX" ]; then
    echo "WARNING: Could not parse [TIMING] output. Check tsqr stdout."
    WALL_MAX="NA"
fi

# Write results to CSV
# Write header if file doesn't exist yet
if [ ! -f "$RESULTS_FILE" ]; then
    echo "job_id,m,n,p,wall_max_s,exit_code" > "$RESULTS_FILE"
fi

# Append results to CSV
echo "${SLURM_JOB_ID},${M},${N},${P},${WALL_MAX},${EXIT_CODE}" >> "$RESULTS_FILE"

echo "[$(date)] Done. Results appended to $RESULTS_FILE"
