#!/bin/bash

## Give the Job a descriptive name
#PBS -N run_fw

## Output and error files
#PBS -o run_fw.out
#PBS -e run_fw.err

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
#./fw 2048
./fw_sr 1024 128
# ./fw_tiled <SIZE> <BSIZE>
