#define _XOPEN_SOURCE 500
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main(const int argc, char *argv[]) {
	// Takes two parameters m and n
	if (argc != 3) {
		fprintf(stderr, "Usage: %s <m> <n>\n", argv[0]);
		return EXIT_FAILURE;
	}

	char *end = NULL;
	const int m = (int) strtol(argv[1], &end, 10);
	if (argv[1] == end || *end != '\0') {
		fprintf(stderr, "Unable to parse m from command line...\n");
		return EXIT_FAILURE;
	}

	const int n = (int) strtol(argv[2], &end, 10);
	if (argv[2] == end || *end != '\0') {
		fprintf(stderr, "Unable to parse n from command line...\n");
		return EXIT_FAILURE;
	}

	// Generates a matrix of size (m x n) filled with random numbers
	srandom((unsigned) time(NULL));
	double *matrix = malloc((long unsigned) (m * n) * sizeof(*matrix));
	if (matrix == NULL) {
		fprintf(stderr, "Unable to allocate memory for matrix...\n");
		return EXIT_FAILURE;
	}
	for (size_t i = 0; i < (size_t) m; i++) {
		for (size_t j = 0; j < (size_t) n; j++) {
			const double scaling_factor  = 100.0;
			const size_t index = (size_t) n * i + j;
			matrix[index] = scaling_factor * ((double) random() / RAND_MAX);
		}
	}

	// Writes the matrix to a file
	char filename[100];
	sprintf(filename, "matrix_%d_%d.txt", m, n);
	FILE *file = fopen(filename, "w");
	if (file == NULL) {
		fprintf(stderr, "Unable to open: %s...\n", filename);
		free(matrix);
		return EXIT_FAILURE;
	}
	for (size_t i = 0; i < (size_t) m; i++) {
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
	free(matrix);
	printf("Wrote %d x %d matrix to: '%s'\n", m, n, filename);

	return EXIT_SUCCESS;
}
