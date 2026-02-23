//
// Created by Kevin Knights on 2/22/26.
//
/*
 * Communication-Avoiding QR (TSQR) factorization — generalized for p ≥ 2 processes.
 *
 *  Requirements:
 *    • p ≥ 2 MPI processes
 *    • m ≥ p  (each rank receives at least one row)
 *    • m ≥ n ≥ 1
 *
 *  Outputs (written by rank 0):
 *    R_final_<n>x<n>.txt  — the final upper-triangular R factor
 *
 *  Algorithm — TSQR binary reduction tree:
 *    Stage 0  Each rank independently QR-factorizes its local mb x n block.
 *    Tree     In ceil(log2(p)) rounds, pairs exchange their n x n R blocks;
 *             the receiver stacks [R_self ; R_partner] (2n x n) and
 *             re-factorizes.  The sender exits the tree.
 *    Gather   All Q reflectors (stage 0 and tree levels) are shipped to rank 0.
 *    Unroll   Rank 0 reverses the tree to reconstruct A = Q·R.
 */

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "mkl.h"
#include "mpi.h"

/*
 * Message-tag scheme:
 *  R blocks are exchanged during the tree reduction. One unique tag per
 *      level prevents messages from different levels from interfering.
 *  MPI mandates MPI_TAG_UB >= 32 767, so 30 levels (tags 100-130) is safe.
 */
#define TAG_R_LEVEL(l)   (100 + (l))

/* Allocation helpers (abort on failure) */
#define XMALLOC(bytes, rank) xmalloc_checked_((bytes), (rank), __func__, __LINE__)
#define XCALLOC(n_, sz, rank) xcalloc_checked_((n_), (sz), (rank), __func__, __LINE__)

void *xmalloc_checked_(const size_t bytes, const int rank, const char *fn, const int line) {
    void *p = malloc(bytes);
    if (!p && bytes > 0) {
        fprintf(stderr, "[Rank %d] malloc(%zu B) failed — %s:%d\n", rank, bytes, fn, line);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return p;
}

void *xcalloc_checked_(size_t n, size_t sz, int rank, const char *fn, int line) {
    void *p = calloc(n, sz);
    if (!p && n > 0 && sz > 0) {
        fprintf(stderr, "[Rank %d] calloc(%zu, %zu) failed — %s:%d\n", rank, n, sz, fn, line);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return p;
}

/* MPI error checker */
void mpi_check_(int rc, const char *expr, const int rank, const char *fn, const int line) {
    if (rc == MPI_SUCCESS) return;
    char msg[MPI_MAX_ERROR_STRING];
    int  len;
    MPI_Error_string(rc, msg, &len);
    fprintf(stderr, "[Rank %d] MPI error in '%s' (%s:%d): %s\n", rank, expr, fn, line, msg);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}
#define MPI_CHK(call, rank) mpi_check_((call), #call, (rank), __func__, __LINE__)

/* Function prototypes */
void row_decomp(int m, int size, int rank, int *start, int *end);
void remove_entries_below_diagonal(double *matrix, int m, int n);
void apply_Q_to_R(int rows, int n, double *q_data, double *tau, const double *C, double *result);

/* 
 * Binary-tree role predicates
 *  At each step s = 2^lvl, ranks are partitioned into pairs (r, r+s) where
 *      r % 2s == 0. The lower-index rank receives; the upper-index rank sends.
 *  Ranks that do not form a complete pair at a given level ("bystanders")
 *      carry their R_current unchanged to the next level.
 */

/** Returns 1 if rank is the receiver in its pair at this step. */
static inline int is_recv(int rank, int size, int step) {
    return (rank % (2 * step) == 0) && (rank + step < size);
}

/** Returns 1 if rank is the sender in its pair at this step. */
static inline int is_send(int rank, int size, int step) {
    (void)size;
    return (rank % (2 * step) == step);
}


int main(int argc, char *argv[]) {

    /* MPI initialization */
    MPI_CHK(MPI_Init(&argc, &argv), -1);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            fprintf(stderr, "Error: requires at least 2 MPI processes.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Argument parsing and broadcast */
    if (argc != 4) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <m> <n> <matrix_file>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int  dims[2]       = {0, 0};
    char filename[256] = {'\0'};

    if (rank == 0) {
        char *ep;
        dims[0] = (int)strtol(argv[1], &ep, 10);
        if (argv[1] == ep || *ep) {
            fprintf(stderr, "Cannot parse <m>.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        dims[1] = (int)strtol(argv[2], &ep, 10);
        if (argv[2] == ep || *ep) {
            fprintf(stderr, "Cannot parse <n>.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        strncpy(filename, argv[3], 255);
    }

    MPI_CHK(MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD), rank);
    const int m = dims[0];
    const int n = dims[1];

    /* Validate dimensions once all ranks know them */
    if (m < size || n < 1 || m < n) {
        if (rank == 0)
            fprintf(stderr,
                    "Error: require m >= size, m >= n >= 1. "
                    "Got m=%d, n=%d, size=%d.\n", m, n, size);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /* Read matrix (rank 0 only) */
    double *matrix = NULL;
    if (rank == 0) {
        FILE *fp = fopen(filename, "r");
        if (!fp) {
            fprintf(stderr, "Cannot open '%s': %s\n", filename, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        matrix = (double *)XMALLOC((size_t)(m * n) * sizeof(double), 0);
        for (int idx = 0; idx < m * n; idx++) {
            char buf[64], *ep;
            if (fscanf(fp, "%63s", buf) != 1) {
                fprintf(stderr, "Unexpected EOF at element %d.\n", idx);
                fclose(fp);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            errno = 0;
            matrix[idx] = strtod(buf, &ep);
            if (ep == buf || errno == ERANGE) {
                fprintf(stderr, "Bad numeric value at element %d.\n", idx);
                fclose(fp);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
        fclose(fp);
    }

    /* Row decomposition and scatter */
    int rs, re;
    row_decomp(m, size, rank, &rs, &re);

    /* mb = local block height.  Saved here; must not be modified after this. */
    const int mb = re - rs + 1;

    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = (int *)XMALLOC((size_t)size * sizeof(int), 0);
        displs     = (int *)XMALLOC((size_t)size * sizeof(int), 0);
        for (int r = 0; r < size; r++) {
            int rrs, rre;
            row_decomp(m, size, r, &rrs, &rre);
            sendcounts[r] = (rre - rrs + 1) * n;
            displs[r]     = rrs * n;
        }
    }

    double *local_matrix = (double *)XMALLOC((size_t)(mb * n) * sizeof(double), rank);

    MPI_CHK(MPI_Scatterv(matrix, sendcounts, displs, MPI_DOUBLE, local_matrix, mb * n, MPI_DOUBLE, 0, MPI_COMM_WORLD), rank);

    if (rank == 0) { free(sendcounts); free(displs); }

    /* Compute tree depth: ceil(log2(size)) */
    int num_levels = 0;
    for (int p = size; p > 1; p = (p + 1) / 2) num_levels++;

    /* 
     * ======================================================================
     * STAGE 0 — Local QR of the mb x n block on each rank.
     *
     * After factorization, we keep:
     *   Q0_data  : the packed Householder reflectors (mb x n) from dgeqrf
     *   tau0     : the scalar factors (min(mb,n) elements)
     *   R_current: the compact upper-triangular n x n R block (only the top
     *              n rows of the factorization, below-diagonal zeros stripped)
     *
     * Only R_current is forwarded into the reduction tree; Q0_data and tau0
     * are gathered at the end for reconstruction.
     * ====================================================================== 
     */

    const int tau0_n = (mb < n) ? mb : n;
    double *tau0 = (double *)XMALLOC((size_t) tau0_n * sizeof(double), rank);
    double *Q0_data = (double *)XMALLOC((size_t)(mb * n) * sizeof(double), rank);

    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, mb, n, local_matrix, n, tau0);

    /* Save Householder reflectors before extracting R */
    memcpy(Q0_data, local_matrix, (size_t)(mb * n) * sizeof(double));

    /*
     * Extract compact n x n R into R_current;
     * transmitting only n x n (not the full mb x n block) keeps tree communication cost O(n^2) per level.
     */
    double *R_current = (double *)XMALLOC((size_t)(n * n) * sizeof(double), rank);
    memcpy(R_current, local_matrix, (size_t)(n * n) * sizeof(double));
    remove_entries_below_diagonal(R_current, n, n);
    free(local_matrix); local_matrix = NULL;

    printf("[Stage 0] Rank %d: QR of local block (%d x %d) complete.\n", rank, mb, n);

    /* 
     * ======================================================================
     * TREE REDUCTION — ceil(log2(p)) rounds of pairwise R-block exchange.
     *
     * At level lvl (step = 2^lvl):
     *   Sender     — sends R_current to its partner rank (rank - step),
     *                frees R_current, and exits the tree (active = 0).
     *   Receiver   — receives R_partner, stacks [R_current ; R_partner]
     *                (2n x n), QR-factorizes, saves reflectors for
     *                reconstruction, and continues with the new n x n R.
     *   Bystander  — rank not paired at this level (only happens when p is
     *                not a power of 2); carries R_current unchanged forward.
     * ====================================================================== 
     */
    double **Q_tree   = (double **)XCALLOC((size_t)num_levels, sizeof(double *), rank);
    double **tau_tree = (double **)XCALLOC((size_t)num_levels, sizeof(double *), rank);

    int active = 1;
    for (int lvl = 0; lvl < num_levels && active; lvl++) {
        const int step = 1 << lvl;

        if (is_send(rank, size, step)) {
            /* Sender path */
            MPI_CHK(MPI_Send(R_current, n * n, MPI_DOUBLE, rank - step, TAG_R_LEVEL(lvl), MPI_COMM_WORLD), rank);
            free(R_current); R_current = NULL;
            active = 0;
            printf("[Level %d] Rank %d -> rank %d: sent R, exiting tree.\n", lvl, rank, rank - step);

        } else if (is_recv(rank, size, step)) {
            /* Receiver path */
            double *R_recv = (double *)XMALLOC((size_t)(n * n) * sizeof(double), rank);
            MPI_CHK(MPI_Recv(R_recv, n * n, MPI_DOUBLE, rank + step, TAG_R_LEVEL(lvl), MPI_COMM_WORLD, MPI_STATUS_IGNORE), rank);

            /* Stack [R_current (n x n) ; R_recv (n x n)] -> 2n x n */
            double *stacked = (double *)XMALLOC((size_t)(2 * n * n) * sizeof(double), rank);
            memcpy(stacked, R_current, (size_t)(n * n) * sizeof(double));
            memcpy(stacked + n * n, R_recv,    (size_t)(n * n) * sizeof(double));
            free(R_recv);
            free(R_current); R_current = NULL;

            /* QR of the 2n x n stacked matrix; tau size = min(2n, n) = n */
            double *tau_l = (double *)XMALLOC((size_t)n * sizeof(double), rank);
            LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, 2 * n, n, stacked, n, tau_l);

            /* Save Householder reflectors for later reconstruction */
            Q_tree[lvl]   = (double *)XMALLOC((size_t)(2 * n * n) * sizeof(double), rank);
            tau_tree[lvl] = tau_l;
            memcpy(Q_tree[lvl], stacked, (size_t)(2 * n * n) * sizeof(double));

            /* Update R_current to the new upper-triangular n x n R */
            R_current = (double *)XMALLOC((size_t)(n * n) * sizeof(double), rank);
            memcpy(R_current, stacked, (size_t)(n * n) * sizeof(double));
            remove_entries_below_diagonal(R_current, n, n);
            free(stacked);

            printf("[Level %d] Rank %d ← rank %d: received R, 2n x n QR complete.\n", lvl, rank, rank + step);
        }
        /* Bystanders (not paired at this level) carry R_current forward.    */
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* 
     * ======================================================================
     * GATHER — ship all Q reflectors to rank 0 for reconstruction.
     *
     * Stage-0 Q blocks have variable height (mb may differ across ranks),
     *  so MPI_Gatherv is required.
     *
     * Tree-level Q blocks are always 2n x n; ranks that were not receivers at
     * a given level contribute 0 elements.
     * ====================================================================== 
     */

    /* Gather per-rank block heights (mb) to rank 0 */
    int *all_mb = NULL;
    if (rank == 0) all_mb = (int *)XMALLOC((size_t)size * sizeof(int), 0);
    MPI_CHK(MPI_Gather(&mb, 1, MPI_INT, all_mb, 1, MPI_INT, 0, MPI_COMM_WORLD), rank);

    /* Gather stage-0 Q reflectors (variable size across ranks) */
    int *q0_cnt = NULL, *q0_dsp = NULL;
    int *t0_cnt = NULL, *t0_dsp = NULL;
    double *Q0_all = NULL, *tau0_all = NULL;

    if (rank == 0) {
        q0_cnt = (int *)XMALLOC((size_t)size * sizeof(int), 0);
        q0_dsp = (int *)XMALLOC((size_t)size * sizeof(int), 0);
        t0_cnt = (int *)XMALLOC((size_t)size * sizeof(int), 0);
        t0_dsp = (int *)XMALLOC((size_t)size * sizeof(int), 0);
        int qd = 0, td = 0;
        for (int r = 0; r < size; r++) {
            q0_cnt[r] = all_mb[r] * n;
            q0_dsp[r] = qd;  qd += q0_cnt[r];
            t0_cnt[r] = (all_mb[r] < n) ? all_mb[r] : n;
            t0_dsp[r] = td;  td += t0_cnt[r];
        }
        const int q0_tot = qd, t0_tot = td;
        Q0_all   = (double *)XMALLOC((size_t)q0_tot * sizeof(double), 0);
        tau0_all = (double *)XMALLOC((size_t)t0_tot * sizeof(double), 0);
    }

    MPI_CHK(MPI_Gatherv(Q0_data, mb * n, MPI_DOUBLE, Q0_all, q0_cnt, q0_dsp, MPI_DOUBLE, 0, MPI_COMM_WORLD), rank);
    MPI_CHK(MPI_Gatherv(tau0, tau0_n, MPI_DOUBLE, tau0_all, t0_cnt, t0_dsp, MPI_DOUBLE, 0, MPI_COMM_WORLD), rank);
    free(Q0_data); Q0_data = NULL;
    free(tau0);    tau0    = NULL;

    /* 
     * ======================================================================
     * Gather tree-level Q reflectors level by level
     *
     * For each level lvl:
     *   Receivers at that level contribute 2*n*n Q values and n tau values.
     *   All other ranks contribute 0.  MPI_Gatherv accepts NULL send buffers
     *   when the local count is 0, which is standard-conformant.
     *
     * recv_q_dsp[lvl][r] = element offset of rank r's Q contribution in
     *   Ql_all[lvl].  Used during reconstruction to locate Q(r, lvl).
     * The corresponding tau offset = recv_q_dsp[lvl][r] / (2*n) because
     * every receiver contributes exactly 2*n*n Q elements and n tau elements.
     * ======================================================================
     */
    double **Ql_all    = NULL;
    double **tauL_all  = NULL;
    int   **recv_q_dsp = NULL;

    if (rank == 0) {
        Ql_all     = (double **)XCALLOC((size_t)num_levels, sizeof(double *), 0);
        tauL_all   = (double **)XCALLOC((size_t)num_levels, sizeof(double *), 0);
        recv_q_dsp = (int    **)XMALLOC((size_t)num_levels * sizeof(int *),   0);
    }

    for (int lvl = 0; lvl < num_levels; lvl++) {
        const int step    = 1 << lvl;
        const int am_recv = is_recv(rank, size, step);
        const int my_qcnt = am_recv ? 2 * n * n : 0;
        const int my_tcnt = am_recv ? n          : 0;

        int    *cnt_q = NULL, *dsp_q = NULL;
        int    *cnt_t = NULL, *dsp_t = NULL;
        double *gq    = NULL, *gt    = NULL;

        if (rank == 0) {
            cnt_q = (int *)XMALLOC((size_t)size * sizeof(int), 0);
            dsp_q = (int *)XMALLOC((size_t)size * sizeof(int), 0);
            cnt_t = (int *)XMALLOC((size_t)size * sizeof(int), 0);
            dsp_t = (int *)XMALLOC((size_t)size * sizeof(int), 0);
            recv_q_dsp[lvl] = (int *)XMALLOC((size_t)size * sizeof(int), 0);

            int qd = 0, td = 0;
            for (int r = 0; r < size; r++) {
                const int recv_r = is_recv(r, size, step);
                cnt_q[r] = recv_r ? 2 * n * n : 0;
                dsp_q[r] = qd;  qd += cnt_q[r];
                cnt_t[r] = recv_r ? n          : 0;
                dsp_t[r] = td;  td += cnt_t[r];
                recv_q_dsp[lvl][r] = dsp_q[r];
            }
            const int q_tot = qd, t_tot = td;
            gq = (q_tot > 0)
                 ? (double *)XMALLOC((size_t)q_tot * sizeof(double), 0) : NULL;
            gt = (t_tot > 0)
                 ? (double *)XMALLOC((size_t)t_tot * sizeof(double), 0) : NULL;
            Ql_all[lvl]   = gq;
            tauL_all[lvl] = gt;
        }

        MPI_CHK(MPI_Gatherv(am_recv ? Q_tree[lvl]   : NULL, my_qcnt, MPI_DOUBLE, gq, cnt_q, dsp_q, MPI_DOUBLE, 0, MPI_COMM_WORLD), rank);
        MPI_CHK(MPI_Gatherv(am_recv ? tau_tree[lvl] : NULL, my_tcnt, MPI_DOUBLE, gt, cnt_t, dsp_t, MPI_DOUBLE, 0, MPI_COMM_WORLD), rank);

        if (rank == 0) { free(cnt_q); free(dsp_q); free(cnt_t); free(dsp_t); }

        /* Local copies are no longer needed after gathering */
        free(Q_tree[lvl]);   Q_tree[lvl]   = NULL;
        free(tau_tree[lvl]); tau_tree[lvl] = NULL;
    }
    free(Q_tree);   Q_tree   = NULL;
    free(tau_tree); tau_tree = NULL;

    /* ======================================================================
     * RECONSTRUCT A = Q0 Q1 … Q_L R  (rank 0 only)
     *
     * We maintain an array R_blocks[r] = the n x n intermediate R for rank r's
     * rows. Unrolling the tree from the highest level down splits each R block
     * into two sub-blocks by applying the stored Q factor.
     *
     * After all levels are unrolled, R_blocks[r] is the n x n R factor that
     * was the input to rank r's stage-0 QR.  Applying Q0[r] yields A_block[r].
     * ====================================================================== */
    if (rank == 0) {
        /* Snapshot of the final R (the n x n upper-triangular output) */
        double *R_final = (double *)XMALLOC((size_t)(n * n) * sizeof(double), 0);
        memcpy(R_final, R_current, (size_t)(n * n) * sizeof(double));

        /* Initialise R_blocks: rank 0 owns R_current; all others start NULL. */
        double **R_blocks = (double **)XCALLOC((size_t)size, sizeof(double *), 0);
        R_blocks[0] = R_current;
        R_current = NULL;   /* ownership transferred */

        /* Unroll from the highest level down to level 0 */
        for (int lvl = num_levels - 1; lvl >= 0; lvl--) {
            const int step = 1 << lvl;

            for (int r = 0; r < size; r++) {
                if (!is_recv(r, size, step) || !R_blocks[r]) continue;

                /*
                 * Locate Q and tau for receiver r at this level.
                 * Every receiver contributes exactly 2*n*n Q values, so
                 * tau_off = q_off / (2*n).
                 */
                const int q_off   = recv_q_dsp[lvl][r];
                const int tau_off = q_off / (2 * n);
                double *Q_lr   = Ql_all[lvl]   + q_off;
                double *tau_lr = tauL_all[lvl] + tau_off;

                /* Q_lr (2n x n) @ R_blocks[r] (n x n) → result (2n x n) */
                double *result = (double *)XMALLOC((size_t)(2 * n * n) * sizeof(double), 0);
                apply_Q_to_R(2 * n, n, Q_lr, tau_lr, R_blocks[r], result);

                /*
                 * Split result:
                 *   top n rows -> R_blocks[r]           (left child)
                 *   bottom n rows -> R_blocks[r + step] (right child)
                 */
                free(R_blocks[r]);
                R_blocks[r]        = (double *)XMALLOC((size_t)(n * n) * sizeof(double), 0);
                R_blocks[r + step] = (double *)XMALLOC((size_t)(n * n) * sizeof(double), 0);
                memcpy(R_blocks[r],        result,          (size_t)(n * n) * sizeof(double));
                memcpy(R_blocks[r + step], result + n * n,  (size_t)(n * n) * sizeof(double));
                free(result);
            }
        }

        /* Apply Q0[r] @ R_blocks[r] → A_block[r] */
        double *A_rec = (double *)XMALLOC((size_t)(m * n) * sizeof(double), 0);
        int dst_off = 0;
        for (int r = 0; r < size; r++) {
            apply_Q_to_R(all_mb[r], n,
                         Q0_all   + q0_dsp[r],
                         tau0_all + t0_dsp[r],
                         R_blocks[r],
                         A_rec + dst_off);
            dst_off += all_mb[r] * n;
            free(R_blocks[r]); R_blocks[r] = NULL;
        }
        free(R_blocks);

        /* Write R_final */
        {
            char fname[64];
            snprintf(fname, sizeof(fname), "TSQR_R_final_%dx%d.txt", n, n);
            FILE *fp = fopen(fname, "w");
            if (!fp) {
                perror("fopen TSQR_R_final");
            } else {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < n; j++)
                        fprintf(fp, "%.10lf%s",
                                R_final[i * n + j],
                                j < n - 1 ? ", " : "\n");
                }
                fclose(fp);
                printf("[Rank 0] Written: %s\n", fname);
            }
            free(R_final);
        }

        /* Numerical validation */
        double max_err = 0.0;
        for (int i = 0; i < m * n; i++) {
            double e = fabs(A_rec[i] - matrix[i]);
            if (e > max_err) max_err = e;
        }
        printf("[VALIDATION] Max reconstruction error : %.2e\n", max_err);
        printf("[VALIDATION] Machine epsilon (double) : %.2e\n", DBL_EPSILON);

        /* Cleanup (rank 0) */
        free(A_rec);
        free(matrix);
        free(all_mb);
        free(q0_cnt); free(q0_dsp);
        free(t0_cnt); free(t0_dsp);
        free(Q0_all); free(tau0_all);
        for (int lvl = 0; lvl < num_levels; lvl++) {
            free(Ql_all[lvl]);
            free(tauL_all[lvl]);
            free(recv_q_dsp[lvl]);
        }
        free(Ql_all);
        free(tauL_all);
        free(recv_q_dsp);
    }

    /* Cleanup (all non-root ranks) */
    if (R_current) { free(R_current); R_current = NULL; }

    MPI_CHK(MPI_Finalize(), rank);
    return EXIT_SUCCESS;
}

/**
 * @brief Row decomposition — assigns a contiguous range of rows to each rank.
 *
 * Rows are 0-indexed. The distribution ensures load balance:
 * ranks 0 … (m % size)−1 each receive one extra row (base_split + 1);
 * the remaining ranks receive base_split rows.
 *
 * @param m       Total number of rows.
 * @param size    Number of MPI processes.
 * @param rank    Rank of this process.
 * @param start   [out] First row index assigned to this rank (inclusive).
 * @param end     [out] Last row index assigned to this rank (inclusive).
 */
void row_decomp(const int m, const int size, const int rank,
                int *start, int *end) {
    const int base_split = m / size;
    const int remainder  = m % size;
    const int rows       = base_split + (rank < remainder ? 1 : 0);
    const int offset     = (rank < remainder)
                           ? rank * (base_split + 1)
                           : remainder * (base_split + 1) + (rank - remainder) * base_split;
    *start = offset;
    *end   = offset + rows - 1;
}

/**
 * @brief Zero out all entries strictly below the main diagonal (in-place).
 *
 * @param matrix  Row-major matrix buffer.
 * @param m       Number of rows.
 * @param n       Number of columns.
 */
void remove_entries_below_diagonal(double *matrix, const int m, const int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (i > j)
                matrix[(size_t)n * i + j] = 0.0;
}

/**
 * @brief Compute result = Q @ C, where Q is given in packed Householder form.
 *
 * Forms the explicit thin Q (rows x n) via dorgqr, then multiplies by C (n x n)
 * to yield result (rows x n).  All matrices are row-major.
 *
 * @param rows    Number of rows in the Q matrix.
 * @param n       Number of columns (also the width of C and result).
 * @param q_data  Packed Householder reflectors from dgeqrf (rows x n, row-major).
 * @param tau     Scalar factors from dgeqrf (min(rows,n) elements).
 * @param C       Input matrix (n x n, row-major).  Not modified.
 * @param result  Output matrix (rows x n, row-major).  Must be pre-allocated.
 */
void apply_Q_to_R(int rows, int n, double *q_data, double *tau,
                  const double *C, double *result) {
    /* 1. Form the explicit thin Q (rows x n) */
    double *Q = (double *)malloc((size_t)(rows * n) * sizeof(double));
    if (!Q) { MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE); }
    memcpy(Q, q_data, (size_t)(rows * n) * sizeof(double));
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, rows, n, n, Q, n, tau);

    /* 2. result (rows x n) = Q (rows x n)  x  C (n x n) */
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, n, n,
                1.0, Q, n,
                     C, n,
                0.0, result, n);
    free(Q);
}