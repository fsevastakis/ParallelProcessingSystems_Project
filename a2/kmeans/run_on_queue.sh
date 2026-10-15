#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_kmeans

## Output and error files
#PBS -o run_kmeans.out
#PBS -e run_kmeans.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=64

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=64
export GOMP_CPU_AFFINITY='0-63'
#./kmeans_omp_nosync_lock -s 32 -n 16 -c 16 -l 10
#./kmeans_omp_pthread_mutex_lock -s 32 -n 16 -c 16 -l 10 
#./kmeans_omp_pthread_spin_lock -s 32 -n 16 -c 16 -l 10
#./kmeans_omp_tas_lock -s 32 -n 16 -c 16 -l 10
#./kmeans_omp_ttas_lock -s 32 -n 16 -c 16 -l 10
#./kmeans_omp_array_lock -s 32 -n 16 -c 16 -l 10
#./kmeans_omp_clh_lock  -s 32 -n 16 -c 16 -l 10
./kmeans_omp_critical -s 32 -n 16 -c 16 -l 10
