#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_kmeans

## Output and error files
#PBS -o run_kmeans_naive_delta.out
#PBS -e run_kmeans_naive_delta.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=64

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=64
#export GOMP_CPU_AFFINITY='0-15'
./kmeans_omp_naive_delta -s 256 -n 16 -c 16 -l 10
