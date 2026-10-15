#!/bin/bash

## Give the Job a descriptive name
#PBS -N runjob

## Output and error files
#PBS -o outrunjob
#PBS -e errorrunjob

## How many machines should we get?
#PBS -l nodes=1:ppn=1

#PBS -l walltime=00:15:00

## Start 
## Run make in the src folder (modify properly)
cd $PBS_O_WORKDIR
./jacobi 1024
#./seidelsor 1024
#./redblacksor 1024
