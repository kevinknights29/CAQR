# Requires: module load gcc/15.2.0-gcc-8.5.0-r7c4jsu
# Requires: module load tbb/latest
# Requires: module load compiler-rt/latest
# Requires: module load mkl/latest

CC = gcc
C_FLAGS = -std=c23 -Wall -Wextra -Wpedantic -Wconversion

all: matrix_generator tsqr

matrix_generator: matrix_generator.c
	$(CC) $(C_FLAGS) -o matrix_generator{,.c}

tsqr: tsqr.c
	$(CC) $(C_FLAGS) -o tsqr{,.c}

.PHONY:
	clean

clean:
	rm -f matrix_generator matrix_*.txt tsqr
