#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/errno.h>
#include "mpi.h"
#include "mkl.h"


// Function Prototype
void row_decomp(const int m, const int size, const int rank, int *start, int *end);


int main(int argc, char *argv[]) {
	// Initialize MPI
	MPI_Init(&argc, &argv);
	int rank = 0;
	int size = 0;
	MPI_Comm_rank(&rank, MPI_COMM_WORLD);
	MPI_Comm_size(&size, MPI_COMM_WORLD);

	// Check command line arguments
	if (argc != 4) {
		if (rank == 0) {
			fprintf(stderr, "Usage: %s <m> <n> <matrix_file>\n", argv[0]);
		}
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

	// Parse command line argument from rank 0 (main processor)
	int m = 0;
	int n = 0;
	char filename[256];
	if (rank == 0) {
		char *end = NULL;
		m = (int) strtol(argv[1], &end, 10);
		if ((argv[1] == end) || (*end != '\0')) {
			fprintf(stderr, "Unable to parse m from command line...\n");
			MPI_Finalize();
			return EXIT_FAILURE;
		}

		n = (int) strtol(argv[2], &end, 10);
		if ((argv[2] == end) || (*end != '\0')) {
			fprintf(stderr, "Unable to parse n from command line...\n");
			MPI_Finalize();
			return EXIT_FAILURE;
		}

		strcpy(filename, argv[3]);
	}

	// Read Matrix from file with rank 0 (main processor)
	double *matrix = NULL;
	double *local_matrix = NULL;
	if (rank == 0) {
		FILE *file = fopen(filename, "r");
		if (file == NULL) {
			fprintf(stderr, "Unable to open: %s...\n", filename);
			MPI_Finalize();
			return EXIT_FAILURE;
		}

		matrix = malloc((long unsigned) (m * n) * sizeof(*matrix));
		if (matrix == NULL) {
			fprintf(stderr, "Unable to allocate memory for matrix...\n");
			MPI_Finalize();
			return EXIT_FAILURE;
		}

		for (size_t i = 0; i < (size_t) m; i++) {
			for (size_t j = 0; j < (size_t) n; j++) {
				char buffer[64];
				char *endptr;
				if (fscanf(file, "%63s", buffer) == 1) {
					matrix[((size_t) m * i) + j] = strtod(buffer, &endptr);
					if (endptr == buffer || errno == ERANGE) {
						fprintf(stderr, "Error converting number\n");
					}
				}
			}
		}
		fclose(file);
	}

	// Broadcast rows to all ranks
	MPI_Bcast(&m, 1, MPI_LONG, 0, MPI_COMM_WORLD);

	// Broadcast columns to all ranks
	MPI_Bcast(&n, 1, MPI_LONG, 0, MPI_COMM_WORLD);

	// Compute rows decomposition
	int starting_row = 0;
	int ending_row = 0;
	row_decomp((int) m, size, rank, &starting_row, &ending_row);

	// Calculate local length from decomposition (1-indexed, inclusive)
	int local_m = (ending_row - starting_row + 1);
	const int local_matrix_size = local_m * n;

	// Allocate local matrix
	local_matrix = malloc((long unsigned) local_matrix_size * sizeof(*local_matrix));

	// Partition and distribute matrix across processors
	// Prepare sendcounts and displacements arrays for Scatterv
	int *sendcounts = NULL;
	int *displs = NULL;
	if (rank == 0) {
		sendcounts = malloc((long unsigned) size * sizeof(*sendcounts));
		displs = malloc((long unsigned) size * sizeof(*displs));
		// Calculate sendcounts and displs for all ranks
		for (int i = 0; i < size; i++) {
			int _starting_row = 0;
			int _ending_row = 0;
			row_decomp(m, size, i, &_starting_row, &_ending_row);
			sendcounts[i] = (_starting_row - _ending_row + 1) * n;  // Number of elements
			displs[i] = _starting_row;
		}
	}

	// Scatter the matrix to all ranks
	MPI_Scatterv(matrix, sendcounts, displs, MPI_DOUBLE, local_matrix, (int) local_matrix_size, MPI_LONG, 0, MPI_COMM_WORLD);

	// Compute QR
	/* ?geqrf Computes the QR factorization of a general m-by-n matrix
	 * Input Parameters:
	 * matrix_layout	Specifies whether matrix storage layout is row major (LAPACK_ROW_MAJOR)
	 *						or column major (LAPACK_COL_MAJOR).
	 * m				The number of rows in the matrix A (m≥ 0).
	 * n				The number of columns in A (n≥ 0).
	 * a				Array a of size max(1, lda*n) for column major layout
	 *						and max(1, lda*m) for row major layout contains the matrix A.
	 * lda				The leading dimension of a; at least max(1, m) for column major layout
	 *						and at least max(1, n) for row major layout.
	 * Output Parameters:
	 * a				Overwritten by the factorization data as follows:
	 *						The elements on and above the diagonal of the array contain the
	 *						min(m,n)-by-n upper trapezoidal matrix R (R is upper triangular if m≥n);
	 *						the elements below the diagonal, with the array tau, present the orthogonal matrix Q
	 *						as a product of min(m,n) elementary reflectors (see Orthogonal Factorizations).
	 * tau				Array, size at least max (1, min(m, n)). Contains scalars that define elementary
	 *						reflectors for the matrix Q in its decomposition in a product of elementary reflectors
	 *						(see Orthogonal Factorizations).
	 */
	double *tau = malloc((long unsigned) (m > n ? n : m) * sizeof(*tau));
	LAPACKE_dgeqrf (LAPACK_ROW_MAJOR, local_m, n, matrix, n, tau);

	// Validate QR
	MPI_Finalize();
	return EXIT_SUCCESS;
}

/**
 * @brief Rows Decomposition of a matrix.
 * 	For example on rank 0, the output should be: s = 1, e = e_0.
 * 	On rank 1, the result should be: s = e_0 + 1, e = e_1.
 * 	On rank p−1 the result should be: s = e_{p−2} + 1, e = n.
 *
 * @param m			An int representing the number of rows.
 * @param size		An int representing the number of processors.
 * @param rank		An int representing the rank of the processor.
 * @param start		Pointer to an int to store starting index
 * @param end		Pointer to an int to store ending index
 */
void row_decomp(const int m, const int size, const int rank, int *start, int *end) {
	const int base_split = m / size;
	const int remainder = m % size;

	// Calculate how many rows this rank gets
	const int my_rows = base_split + (rank < remainder ? 1 : 0);

	// Calculate starting position (1-indexed)
	// Ranks [0, remainder-1] get (base_split + 1) elements each
	// Ranks [remainder, p-1] get base_split elements each
	int offset = 0;
	if (rank < remainder) {
		offset = rank * (base_split + 1);
	} else {
		offset = remainder * (base_split + 1) + (rank - remainder) * base_split;
	}

	*start = offset;  // Convert to 1-indexed
	*end = offset + my_rows;  // Last element for this rank
}