#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"
/*
 * TODO: include openmp header file
 */ 
#include <omp.h>

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

inline static int find_nearest_cluster(int      numClusters, /* no. clusters */
                                       int      numCoords,   /* no. coordinates */
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

void kmeans(double * objects,          /* in: [numObjs][numCoords] */
            int      numCoords,        /* no. coordinates */
            int      numObjs,          /* no. objects */
            int      numClusters,      /* no. clusters */
            double   threshold,        /* minimum fraction of objects that change membership */
            long     loop_threshold,   /* maximum number of iterations */
            int    * membership,       /* out: [numObjs] */
            double * clusters)         /* out: [numClusters][numCoords] */
{
    int i, j, k;
    int index, loop=0;
    double timing = 0;

    double delta;          // fraction of objects whose clusters change in each loop 
    int * newClusterSize; // [numClusters]: no. objects assigned in each new cluster 
    double * newClusters;  // [numClusters][numCoords] 
    int nthreads;         // no. threads 

    nthreads = omp_get_max_threads();
    printf("OpenMP Kmeans - Reduction\t(number of threads: %d)\n", nthreads);

    // initialize membership
    for (i=0; i<numObjs; i++)
        membership[i] = -1;

    // initialize newClusterSize and newClusters to all 0 
    newClusterSize = (typeof(newClusterSize)) calloc(numClusters, sizeof(*newClusterSize));
    newClusters = (typeof(newClusters))  calloc(numClusters * numCoords, sizeof(*newClusters));

    // Each thread calculates new centers using a private space. After that, thread 0 does an array reduction on them.
    int * local_newClusterSize[nthreads];  // [nthreads][numClusters] 
    double * local_newClusters[nthreads];   // [nthreads][numClusters][numCoords]

    /*
     * Hint for false-sharing
     * This is noticed when numCoords is low (and neighboring local_newClusters exist close to each other).
     * Allocate local cluster data with a "first-touch" policy.
     */
    // Initialize local (per-thread) arrays (and later collect result on global arrays)
    // Allocating in parallel (iteration k runs on thread k) means each thread's arrays come from its
    // own malloc arena, not right next to another thread's (false sharing), and are first touched,
    // so placed on the NUMA node, by the thread that uses them.
    #pragma omp parallel for private(k)
    for (k=0; k<nthreads; k++)
    {
        local_newClusterSize[k] = (typeof(*local_newClusterSize)) calloc(numClusters, sizeof(**local_newClusterSize));
        local_newClusters[k] = (typeof(*local_newClusters)) calloc(numClusters * numCoords, sizeof(**local_newClusters));
    }

    timing = wtime();
    do {
        // before each loop, set cluster data to 0
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++)
                newClusters[i*numCoords + j] = 0.0;
            newClusterSize[i] = 0;
        }

        delta = 0.0;
        double local_delta[nthreads];
        /* 
         * TODO: Initiliaze local cluster data to zero (separate for each thread)
         */
	for (k=0; k<nthreads; k++) {
		for (i=0; i<numClusters; i++) {
            		for (j=0; j<numCoords; j++)
                		local_newClusters[k][i*numCoords + j] = 0.0;
            	local_newClusterSize[k][i] = 0;
        	}
	}
       
        for (k=0; k<nthreads; k++) local_delta[k]=0.0;
           
	int tid;	// selects this thread's private arrays and local_delta entry, so no atomics are needed

	#pragma omp parallel for private(i,j,index,tid)

        for (i=0; i<numObjs; i++)
        {
            // find the array index of nearest cluster center 
            index = find_nearest_cluster(numClusters, numCoords, &objects[i*numCoords], clusters);
            
	    tid = omp_get_thread_num();
            // if membership changes, increase delta by 1 
            if (membership[i] != index)
		local_delta[tid] += 1.0;
            
            // assign the membership to object i 
            membership[i] = index;
            
            // update new cluster centers : sum of all objects located within (average will be performed later) 
            /* 
             * TODO: Collect cluster data in local arrays (local to each thread)
             *       Replace global arrays with local per-thread
             */ 	
            local_newClusterSize[tid][index]++;
            for (j=0; j<numCoords; j++)
               local_newClusters[tid][index*numCoords + j] += objects[i*numCoords + j];

        }

        /*
         * TODO: Reduction of cluster data from local arrays to shared.
         *       This operation will be performed by one thread
         */
	
	// reduction: done serially by the master thread, after the parallel loop's implicit barrier
        for (k=0; k<nthreads; k++) {
		for (i=0; i<numClusters; i++) {
			for (j=0; j<numCoords; j++)
	    			newClusters[i*numCoords + j] += local_newClusters[k][i*numCoords + j];
			newClusterSize[i] += local_newClusterSize[k][i];
		}
          delta += local_delta[k];
	}
 

        // average the sum and replace old cluster centers with newClusters 
        for (i=0; i<numClusters; i++) {
            if (newClusterSize[i] > 0) {
                for (j=0; j<numCoords; j++) {
                    clusters[i*numCoords + j] = newClusters[i*numCoords + j] / newClusterSize[i];
                }
            }
        }

        // Get fraction of objects whose membership changed during this loop. This is used as a convergence criterion.
        delta /= numObjs;

        loop++;
        printf("\r\tcompleted loop %d", loop);
        fflush(stdout);
    } while (delta > threshold && loop < loop_threshold);
    timing = wtime() - timing;
    printf("\n        nloops = %3d   (total = %7.4fs)  (per loop = %7.4fs)\n", loop, timing, timing/loop);

    for (k=0; k<nthreads; k++)
    {
        free(local_newClusterSize[k]);
        free(local_newClusters[k]);
    }
    free(newClusters);
    free(newClusterSize);
}
