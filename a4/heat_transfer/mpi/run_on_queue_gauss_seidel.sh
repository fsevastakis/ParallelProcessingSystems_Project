#!/bin/bash

## Give the Job a descriptive name
#PBS -N runjob

## Output and error files
#PBS -o outrunjob_gauss_seidel
#PBS -e errorrunjob_gauss_seidel

## How many machines should we get?
#PBS -l nodes=8:ppn=8

#PBS -l walltime=00:25:00

## Start 
## Run make in the src folder (modify properly)
module load openmpi/1.8.3
cd $PBS_O_WORKDIR
#mpirun --mca btl tcp,self -np 32 ./seidelsor 6144 6144 4 8

sizes='2048 4096 6144'
threads=('1' '2' '4' '8' '16' '32' '64')
threads1=('1' '1' '2' '2' '4' '4' '8')
threads2=('1' '2' '2' '4' '4' '8' '8')
 
for size in $sizes
do 
    for i in "${!threads[@]}"
    do
        echo "size: $size, threads: ${threads[i]}"
        mpirun --mca btl tcp,self -np "${threads[i]}" ./seidelsor $size $size "${threads1[i]}" "${threads2[i]}"
    done
done

