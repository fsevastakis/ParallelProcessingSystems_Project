#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_kmeans

## Output and error files
#PBS -o run_kmeans_reduction.out
#PBS -e run_kmeans_reduction.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=4

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=4
export GOMP_CPU_AFFINITY='0-3'
./kmeans_omp_reduction -s 256 -n 1 -c 4 -l 10
