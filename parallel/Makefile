# Requires: module load gcc/15.2.0-gcc-8.5.0-r7c4jsu mpi/latest tbb/latest compiler-rt/latest mkl/latest

CC = gcc
MPI_CC = mpicc
CC_FLAGS = -std=c23 -Wall -Wextra -Wpedantic -Wconversion
MKL_FLAGS = -m64 -I "${MKLROOT}/include" -L "${MKLROOT}/lib"
LD_FLAGS = -lmkl_rt -Wl,--no-as-needed -lpthread -lm -ldl

all: matrix_generator tsqr

matrix_generator: matrix_generator.c
	$(CC) $(CC_FLAGS) -o matrix_generator{,.c}

tsqr: tsqr.c
	$(MPI_CC) $(CC_FLAGS) $(MKL_FLAGS) -o tsqr{,.c} $(LD_FLAGS)

.PHONY:
	clean test

clean:
	rm -f matrix_generator matrix_*.txt tsqr

test: matrix_generator tsqr
	./matrix_generator 16 3
	mpirun -np 4 ./tsqr 16 3 matrix_16_3.txt