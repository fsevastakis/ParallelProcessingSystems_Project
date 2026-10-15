#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_mpi_kmeans

## Output and error files
#PBS -o run_mpi_kmeans.out
#PBS -e run_mpi_kmeans.err

## How many machines should we get? 
#PBS -l nodes=8:ppn=8

## Start 
## Run make in the src folder (modify properly)

module load openmpi/1.8.3
cd $PBS_O_WORKDIR
mpirun -np 64 --mca btl tcp,self ./kmeans_mpi -s 256 -n 16 -c 16 -l 10 

