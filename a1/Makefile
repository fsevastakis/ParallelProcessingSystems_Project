all: omp_Game_Of_Life

omp_Game_Of_Life: Game_Of_Life.c
	gcc -O3 -fopenmp -o omp_Game_Of_Life Game_Of_Life.c

clean: 
	rm omp_Game_Of_Life
