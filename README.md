# CAQR

Implementation of the Communication-Optimal TSQR algorithm based on Demmel et al. [1] for Case Studies in HPC.

## Repository Structure

```
.
├── README.md
├── parallel/       # MPI-based parallel implementation (C)
│   ├── matrix_generator.c
│   ├── tsqr.c
│   └── Makefile
└── sequential/     # Sequential implementation (Jupyter Notebook)
```

## Parallel Implementation

### Environment

The parallel version was developed and tested on Seagul. 

The following modules must be loaded before building or running:

```bash
module load gcc/15.2.0-gcc-8.5.0-r7c4jsu mpi/latest tbb/latest compiler-rt/latest mkl/latest
```

### Building

From the `parallel/` directory:

```bash
make
```

This compiles both `matrix_generator` and `tsqr`.

### Running

To generate a matrix and run TSQR on it manually:

```bash
./matrix_generator <rows> <cols>
mpirun -np 4 ./tsqr <rows> <cols> <matrix_file>
```

### Reproducing Test Results

A `make test` target is provided that generates a 16×3 matrix and runs TSQR on it with 4 MPI processes:

```bash
make test
```

This is equivalent to:

```bash
./matrix_generator 16 3
mpirun -np 4 ./tsqr 16 3 matrix_16_3.txt
```

### Cleaning Up

```bash
make clean
```

Removes compiled binaries and any generated matrix `.txt` files.

## Sequential Implementation

The `sequential/` directory contains a Jupyter Notebook with a step-by-step sequential implementation of the TSQR algorithm. Open and run the notebook in order to reproduce the sequential results.

## Reference

[1] J. Demmel, L. Grigori, M. Hoemmen, and J. Langou, "Implementing communication-optimal parallel and sequential QR factorizations," *arXiv preprint arXiv:0809.2407 [math.NA]*, 2008. https://doi.org/10.48550/arXiv.0809.2407