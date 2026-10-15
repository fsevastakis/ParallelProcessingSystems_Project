CC=mpicc
CFLAGS=-O3 -lm -Wall
#RES=-DPRINT_RESULTS
CONV=-DTEST_CONV

all: jacobi seidelsor redblacksor

jacobi: mpi_jacobi.c utils.c
	$(CC) $(CFLAGS) $(RES) $(CONV) mpi_jacobi.c utils.c -o jacobi

seidelsor: mpi_gauss_seidel.c utils.c
	$(CC) $(CFLAGS) $(RES) mpi_gauss_seidel.c utils.c -o seidelsor

redblacksor: mpi_redblack.c utils.c
	$(CC) $(CFLAGS) $(RES) mpi_redblack.c utils.c -o redblacksor

clean:
	rm -f jacobi seidelsor redblacksor
