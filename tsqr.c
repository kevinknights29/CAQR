#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/errno.h>
#include "mkl.h"
#include "mpi.h"


// Function Prototype
void row_decomp(int m, int size, int rank, int *start, int *end);


int main(int argc, char *argv[]) {
	// Initialize MPI
	MPI_Init(&argc, &argv);
	int rank = 0;
	int size = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

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
		if (argv[1] == end || *end != '\0') {
			fprintf(stderr, "Unable to parse m from command line...\n");
			MPI_Finalize();
			return EXIT_FAILURE;
		}

		n = (int) strtol(argv[2], &end, 10);
		if (argv[2] == end || *end != '\0') {
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
			free(matrix);
			fclose(file);
			MPI_Finalize();
			return EXIT_FAILURE;
		}

		for (size_t i = 0; i < (size_t) m; i++) {
			for (size_t j = 0; j < (size_t) n; j++) {
				char buffer[64];
				char *endptr;
				if (fscanf(file, "%63s", buffer) == 1) {
					const size_t index = (size_t) n * i + j;
					matrix[index] = strtod(buffer, &endptr);
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
	row_decomp(m, size, rank, &starting_row, &ending_row);

	// Calculate local length from decomposition (1-indexed, inclusive)
	int local_m = ending_row - starting_row + 1;
	int local_matrix_size = local_m * n;

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
			sendcounts[i] = (_ending_row - _starting_row + 1) * n;  // Number of elements
			displs[i] = _starting_row;
		}
	}

	// Scatter the matrix to all ranks
	MPI_Scatterv(matrix, sendcounts, displs, MPI_DOUBLE, local_matrix, (int) local_matrix_size, MPI_LONG, 0, MPI_COMM_WORLD);

	// Clean up
	if (rank == 0) {
		free(matrix);
		free(sendcounts);
		free(displs);
	}

	// Stage 0: Independent computation of the QR factorization of each block row
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
	LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, local_m, n, local_matrix, n, tau);
	printf("[DEBUG] Rank %d computed QR with 'LAPACKE_dgeqrf' of matrix %d x %d\n", rank, local_m, n);

	// Stage 1: Group them into successive pairs and do the QR factorizations of grouped pairs in parallel
	double *incoming_matrix = NULL;
	int incoming_matrix_size = 0;
	double *stacked_matrix = NULL;

	// Processor 1 sends R matrix to Processor 0
	if (rank == 1) {
		const int receptor_rank = 0;
		MPI_Send(&local_matrix_size, 1, MPI_INT, receptor_rank, 90, MPI_COMM_WORLD);
		MPI_Send(local_matrix, local_matrix_size, MPI_DOUBLE, receptor_rank, 91, MPI_COMM_WORLD);
		printf("[DEBUG] Rank %d sent R matrix of size %d x %d\n", rank, local_m, n);

		// Cleanup
		free(local_matrix);
		free(tau);
	}

	// Processor 0 receives matrix
	if (rank == 0) {
		const int emissor_rank = 1;
		MPI_Recv(&incoming_matrix_size, 1, MPI_INT, emissor_rank, 90, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		incoming_matrix = malloc((long unsigned) incoming_matrix_size * sizeof(*incoming_matrix));
		if (incoming_matrix == NULL) {
			fprintf(stderr, "Unable to allocate memory for incoming matrix...\n");
			free(local_matrix);
			free(tau);
			MPI_Finalize();
			return EXIT_FAILURE;
		}
		MPI_Recv(incoming_matrix, incoming_matrix_size, MPI_DOUBLE, emissor_rank, 91, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("[DEBUG] Rank %d got R matrix of size %d x %d from Rank %d\n", rank, local_m, n, emissor_rank);
	}

	// Processor 3 sends R matrix to Processor 2
	if (rank == 3) {
		const int receptor_rank = 2;
		MPI_Send(&local_matrix_size, 1, MPI_INT, receptor_rank, 92, MPI_COMM_WORLD);
		MPI_Send(local_matrix, local_matrix_size, MPI_DOUBLE, receptor_rank, 93, MPI_COMM_WORLD);
		printf("[DEBUG] Rank %d sent R matrix of size %d x %d\n", rank, local_m, n);

		// Cleanup
		free(local_matrix);
		free(tau);
	}

	// Processor 2 receives matrix
	if (rank == 2) {
		const int emissor_rank = 3;
		MPI_Recv(&incoming_matrix_size, 1, MPI_INT, emissor_rank, 92, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		incoming_matrix = malloc((long unsigned) incoming_matrix_size * sizeof(*incoming_matrix));
		if (incoming_matrix == NULL) {
			fprintf(stderr, "Unable to allocate memory for incoming matrix...\n");
			free(local_matrix);
			free(tau);
			MPI_Finalize();
			return EXIT_FAILURE;
		}
		MPI_Recv(incoming_matrix, incoming_matrix_size, MPI_DOUBLE, emissor_rank, 93, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("[DEBUG] Rank %d got R matrix of size %d x %d from Rank %d\n", rank, local_m, n, emissor_rank);
	}

	// Processors 0 and 2 stack local_matrix and incoming_matrix
	if (rank == 0 || rank == 2) {
		stacked_matrix = malloc((long unsigned) (local_matrix_size + incoming_matrix_size) * sizeof(*stacked_matrix));
		if (stacked_matrix == NULL) {
			fprintf(stderr, "Unable to allocate memory for stacked matrix...\n");
			free(local_matrix);
			free(incoming_matrix);
			free(tau);
			MPI_Finalize();
			return EXIT_FAILURE;
		}
		memcpy(stacked_matrix, local_matrix, (long unsigned) local_matrix_size * sizeof(*local_matrix));
		memcpy(stacked_matrix + incoming_matrix_size, incoming_matrix, (long unsigned) incoming_matrix_size * sizeof(*incoming_matrix));

		// Cleanup
		free(local_matrix);
		free(incoming_matrix);
		free(tau);

		// Update pointers
		local_matrix = stacked_matrix;
		stacked_matrix = NULL;
		tau = NULL;
		incoming_matrix = NULL;
		local_m = (local_matrix_size + incoming_matrix_size) / n;
		local_matrix_size = local_m * n;
		printf("[DEBUG] Rank %d stacked R matrices resulting in a matrix %d x %d\n", rank, local_m, n);
	}

	// Stage 2: Independent computation of the QR factorization of each stacked block row
	if (rank == 0 || rank == 2) {
		LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, local_m, n, local_matrix, n, tau);
		printf("[DEBUG] Rank %d computed QR with 'LAPACKE_dgeqrf' of matrix %d x %d\n", rank, local_m, n);
	}

	// Stage 3: Group them into successive pairs and do the QR factorizations of grouped pairs in parallel
	// Processor 2 sends R matrix to Processor 0
	if (rank == 2) {
		const int receptor_rank = 0;
		MPI_Send(&local_matrix_size, 1, MPI_INT, receptor_rank, 94, MPI_COMM_WORLD);
		MPI_Send(local_matrix, local_matrix_size, MPI_DOUBLE, receptor_rank, 95, MPI_COMM_WORLD);
		printf("[DEBUG] Rank %d sent R matrix of size %d x %d\n", rank, local_m, n);

		// Cleanup
		free(local_matrix);
		free(tau);
	}

	// Processor 0 receives matrix
	if (rank == 0) {
		const int emissor_rank = 2;
		MPI_Recv(&incoming_matrix_size, 1, MPI_INT, emissor_rank, 94, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		incoming_matrix = malloc((long unsigned) incoming_matrix_size * sizeof(*incoming_matrix));
		if (incoming_matrix == NULL) {
			fprintf(stderr, "Unable to allocate memory for incoming matrix...\n");
			free(local_matrix);
			free(tau);
			MPI_Finalize();
			return EXIT_FAILURE;
		}
		MPI_Recv(incoming_matrix, incoming_matrix_size, MPI_DOUBLE, emissor_rank, 95, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("[DEBUG] Rank %d got R matrix of size %d x %d from Rank %d\n", rank, local_m, n, emissor_rank);
	}

	// Processors 0 stacks local_matrix and incoming_matrix
	if (rank == 0) {
		stacked_matrix = malloc((long unsigned) (local_matrix_size + incoming_matrix_size) * sizeof(*stacked_matrix));
		if (stacked_matrix == NULL) {
			fprintf(stderr, "Unable to allocate memory for stacked matrix...\n");
			free(local_matrix);
			free(incoming_matrix);
			free(tau);
			MPI_Finalize();
			return EXIT_FAILURE;
		}
		memcpy(stacked_matrix, local_matrix, (long unsigned) local_matrix_size * sizeof(*local_matrix));
		memcpy(stacked_matrix + incoming_matrix_size, incoming_matrix, (long unsigned) incoming_matrix_size * sizeof(*incoming_matrix));

		// Cleanup
		free(local_matrix);
		free(incoming_matrix);
		free(tau);

		// Update pointers
		local_matrix = stacked_matrix;
		stacked_matrix = NULL;
		tau = NULL;
		incoming_matrix = NULL;
		local_m = (local_matrix_size + incoming_matrix_size) / n;
		local_matrix_size = local_m * n;
		printf("[DEBUG] Rank %d stacked R matrices resulting in a matrix %d x %d\n", rank, local_m, n);
	}

	// Validate QR
	if (rank == 0) {
		char output_filename[256];
		sprintf(output_filename, "caqr_matrix_%d_%d.txt", local_m, n);
		FILE *file = fopen(output_filename, "w");
		if (file == NULL) {
			fprintf(stderr, "Unable to open: %s...\n", output_filename);
			free(local_matrix);
			return EXIT_FAILURE;
		}
		for (size_t i = 0; i < (size_t) local_m; i++) {
			for (size_t j = 0; j < (size_t) n; j++) {
				const size_t index = (size_t) n * i + j;
				fprintf(file, "%.10lf", matrix[index]);
				if (j < (size_t) (n - 1)) {
					fprintf(file, ", ");
				} else {
					fprintf(file, "\n");
				}

			}
		}
		fclose(file);
		printf("[DEBUG] Rank %d wrote resulting R matrix %d x %d\n", rank, local_m, n);
	}

	// Cleanup
	if (rank == 0) {
		free(local_matrix);
	}
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
	const int rows = base_split + (rank < remainder ? 1 : 0);

	// Calculate starting position (1-indexed)
	// Ranks [0, remainder-1] get (base_split + 1) elements each
	// Ranks [remainder, p-1] get base_split elements each
	int offset = 0;
	if (rank < remainder) {
		offset = rank * (base_split + 1);
	} else {
		offset = remainder * (base_split + 1) + (rank - remainder) * base_split;
	}

	*start = offset;
	*end = offset + rows - 1;  // Last element for this rank
}