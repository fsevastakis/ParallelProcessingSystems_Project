#!/bin/bash

## Give the Job a descriptive name
#PBS -N makejob

## Output and error files
#PBS -o outmakejob
#PBS -e errormakejob

## How many machines should we get?
#PBS -l nodes=1:ppn=1

#PBS -l walltime=00:05:00

## Start 
module load openmpi/1.8.3 
## Run make in the src folder (modify properly)
cd $PBS_O_WORKDIR
make
