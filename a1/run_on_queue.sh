#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_omp_Game_Of_Life

## Output and error files
#PBS -o run_omp_Game_Of_Life.out
#PBS -e run_omp_Game_Of_Life.err

## How many machines should we get? 
#PBS -l nodes=1:ppn=8

##How long should the job run for?
#PBS -l walltime=00:10:00

## Start 
## Run make in the src folder (modify properly)

module load openmp
cd $PBS_O_WORKDIR
export OMP_NUM_THREADS=8
./omp_Game_Of_Life 4096 1000

