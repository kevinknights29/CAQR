#!/bin/bash
# Submits a sweep of TSQR jobs for scaling plots.
# Usage: bash submit_scaling.sh

P=4

# ── Define your scaling cases ─────────────────────────────────────────────────
# Format: "m n"  — keep m/p >> n in all cases

M_VALUES=(4000 8000 16000 32000 64000 128000)
N_VALUES=(50 100 200)   # vary n independently for strong/weak scaling

for N in "${N_VALUES[@]}"; do
    for M in "${M_VALUES[@]}"; do
        RATIO=$(( M / P ))
        if [ "$RATIO" -le "$N" ]; then
            echo "Skipping m=$M, n=$N: m/p=$RATIO is not >> n"
            continue
        fi

        echo "Submitting: m=$M, n=$N"
        sbatch --export=ALL,M="$M",N="$N" \
               --job-name="tsqr_${M}_${N}" \
               run_tsqr.sh
    done
done

echo "All jobs submitted. Monitor with: squeue -u $USER"
