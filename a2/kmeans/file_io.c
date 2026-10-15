#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* read(), close() */
// TODO: remove comment from following line
#include <omp.h>

#include "kmeans.h"

double * dataset_generation(int numObjs, int numCoords)
{
    double * objects = NULL;
    long i, j;
    // Random values that will be generated will be between 0 and 10.
    double val_range = 10;

    /* allocate space for objects[][] and read all objects */
    objects = (typeof(objects)) malloc(numObjs * numCoords * sizeof(*objects));

    /*
     * Hint : Could dataset generation be performed in a more "NUMA-Aware" way?
     *        Need to place data "close" to the threads that will perform operations on them.
     *        reminder : First-touch data placement policy
     */

    for (i=0; i<numObjs; i++)
    {
        unsigned int seed = i;
        for (j=0; j<numCoords; j++)
        {
            objects[i*numCoords + j] = (rand_r(&seed) / ((double) RAND_MAX)) * val_range;
            if (_debug && i == 0)
                printf("object[i=%ld][j=%ld]=%f\n",i,j,objects[i*numCoords + j]);
        }
    } 

    return objects;
}
