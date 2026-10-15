#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "kmeans.h"

// square of Euclid distance between two multi-dimensional points
inline static double euclid_dist_2(int    numdims,  /* no. dimensions */
                                 double * coord1,   /* [numdims] */
                                 double * coord2)   /* [numdims] */
{
    int i;
    double ans = 0.0;

    for(i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return ans;
}

inline static int find_nearest_cluster(int     numClusters, /* no. clusters */
                                       int     numCoords,   /* no. coordinates */
                                       double * object,      /* [numCoords] */
                                       double * clusters)    /* [numClusters][numCoords] */
{
    int index, i;
    double dist, min_dist;

    // find the cluster id that has min distance to object 
    index = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters);

    for(i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, &clusters[i*numCoords]);
        // no need square root 
        if (dist < min_dist) { // find the min and its array index
            min_dist = dist;
            index    = i;
        }
    }
    return index;
}

void kmeans(double * objects,         /* in: [numObjs][numCoords] */
            int     numCoords,        /* no. coordinates */
            int     numObjs,          /* no. objects */
            int     numClusters,      /* no. clusters */
            double   threshold,       /* minimum fraction of objects that change membership */
            long    loop_threshold,   /* maximum number of iterations */
            int   * membership,       /* out: [numObjs] */
            double * clusters)        /* out: [numClusters][numCoords] */
{
    int i, j;
    int index, loop=0;
    double timing = 0;

    /* Every variable has its "rank_" version, which is used to store local data,
     * and its "new" version, which is used to store global data.
     */
    double rank_delta, delta = 0;                // fraction of objects whose clusters change in each loop 
    int * rank_newClusterSize, * newClusterSize; // [numClusters]: no. objects assigned in each new cluster 
    double * rank_newClusters, *newClusters;     // [numClusters][numCoords] 
    
    // Get rank of this process    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // initialize membership
    for (i=0; i<numObjs; i++)
        membership[i] = -1;

    // initialize rank_newClusterSize and rank_newClusters to all 0 
    rank_newClusterSize = (typeof(rank_newClusterSize)) calloc(numClusters, sizeof(*rank_newClusterSize));
    rank_newClusters    = (typeof(rank_newClusters))  calloc(numClusters * numCoords, sizeof(*rank_newClusters));
    newClusterSize      = (typeof(newClusterSize)) calloc(numClusters, sizeof(*newClusterSize));
    newClusters         = (typeof(newClusters))  calloc(numClusters * numCoords, sizeof(*newClusters));

    timing = wtime();
    do {
        // before each loop, set cluster data to 0
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++)
                rank_newClusters[i*numCoords + j] = 0.0;
            rank_newClusterSize[i] = 0;
        }

        rank_delta = 0.0;

        for (i=0; i<numObjs; i++) {
            // find the array index of nearest cluster center 
            index = find_nearest_cluster(numClusters, numCoords, &objects[i*numCoords], clusters);
            
            // if membership changes, increase rank_delta by 1 
            if (membership[i] != index)
                rank_delta += 1.0;
            
            // assign the membership to object i 
            membership[i] = index;
            
            // update new cluster centers : sum of objects located within
            rank_newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                rank_newClusters[index*numCoords + j] += objects[i*numCoords + j];
        }

        /*
         * TODO: Perform reduction of cluster data (rank_newClusters, rank_newClusterSize) from local arrays to shared.
         */

	// Allreduce rather than Reduce: every rank needs the new cluster centers for its next iteration.
	MPI_Allreduce(rank_newClusters, newClusters, numClusters*numCoords, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(rank_newClusterSize, newClusterSize, numClusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // average the sum and replace old cluster centers with newClusters
        for (i=0; i<numClusters; i++) {
            if (newClusterSize[i] > 0) {
                for (j=0; j<numCoords; j++) {
                    clusters[i*numCoords + j] = newClusters[i*numCoords + j] / newClusterSize[i];
                }
            }
        }
        
        /*
         * TODO: Perform reduction from rank_delta variable to delta variable, that will be used for convergence check.
         */
	MPI_Allreduce(&rank_delta, &delta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Get fraction of objects whose membership changed during this loop. This is used as a convergence criterion.
        delta /= numObjs;
        loop++;
        printf("\r\tcompleted loop %d", loop);
        //fflush(stdout);
    } while (delta > threshold && loop < loop_threshold);
    
    timing = wtime() - timing;
    if (rank == 0) fprintf(stdout, "        nloops = %3d   (total = %7.4fs)  (per loop = %7.4fs)\n", loop, timing, timing/loop);
 
    free(rank_newClusters);
    free(rank_newClusterSize);
    free(newClusters);
    free(newClusterSize);
}
