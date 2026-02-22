#!/bin/bash
#SBATCH --job-name=tsqr_scaling
#SBATCH --nodes=1
#SBATCH --ntasks=4                  # 4 MPI processes
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00             # Adjust as needed
#SBATCH --output=logs/tsqr_%j.out
#SBATCH --error=logs/tsqr_%j.err
#SBATCH --partition=your_partition  # Change to your cluster's partition

# ── Parameters passed via --export or environment variables ──────────────────
# Usage: sbatch --export=ALL,M=10000,N=100 run_tsqr.sh
M=${M:-10000}   # default fallback if not passed
N=${N:-100}

MATRIX_FILE="matrix_${M}_${N}.txt"
RESULTS_FILE="results/tsqr_results.csv"

# ── Sanity check: m/p >> n ────────────────────────────────────────────────────
P=4
RATIO=$(( M / P ))
if [ "$RATIO" -le "$N" ]; then
    echo "WARNING: m/p ($RATIO) is not >> n ($N). Results may not be meaningful."
fi

# ── Load modules (adapt to your cluster's module system) ──────────────────────
module purge
module load mpi/openmpi        # or intel-mpi, mpich, etc.

# ── Setup directories ─────────────────────────────────────────────────────────
mkdir -p logs results

# ── Generate the matrix ───────────────────────────────────────────────────────
echo "[$(date)] Generating matrix: m=$M, n=$N"
./matrix_generator "$M" "$N"

if [ ! -f "$MATRIX_FILE" ]; then
    echo "ERROR: Matrix file '$MATRIX_FILE' was not created. Aborting."
    exit 1
fi

# ── Run TSQR with timing ──────────────────────────────────────────────────────
echo "[$(date)] Launching TSQR: m=$M, n=$N, p=$P"

# /usr/bin/time writes to stderr by default; redirect to a temp file
TIME_OUTPUT=$( { /usr/bin/time -v \
    mpirun -np "$P" ./tsqr "$M" "$N" "$MATRIX_FILE" \
    ; } 2>&1 )

EXIT_CODE=$?

# ── Parse wall-clock time from /usr/bin/time -v output ───────────────────────
WALL_TIME=$(echo "$TIME_OUTPUT" | grep "Elapsed (wall clock)" | awk '{print $NF}')
USER_TIME=$(echo "$TIME_OUTPUT" | grep "User time"           | awk '{print $NF}')
SYS_TIME=$( echo "$TIME_OUTPUT" | grep "System time"         | awk '{print $NF}')
MAX_RSS=$(  echo "$TIME_OUTPUT" | grep "Maximum resident"     | awk '{print $NF}')

echo "$TIME_OUTPUT"  # echo full output to the .out log

# ── Append results to CSV ─────────────────────────────────────────────────────
# Write header if file doesn't exist yet
if [ ! -f "$RESULTS_FILE" ]; then
    echo "job_id,m,n,p,wall_time,user_time,sys_time,max_rss_kb,exit_code" > "$RESULTS_FILE"
fi

echo "${SLURM_JOB_ID},${M},${N},${P},${WALL_TIME},${USER_TIME},${SYS_TIME},${MAX_RSS},${EXIT_CODE}" >> "$RESULTS_FILE"

echo "[$(date)] Done. Results appended to $RESULTS_FILE"
