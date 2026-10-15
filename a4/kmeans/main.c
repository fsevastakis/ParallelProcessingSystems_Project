#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */
#include <mpi.h>

int _debug;
#include "kmeans.h"

static void usage(char *argv0) {
    char *help =
        "Usage: %s [switches]\n"
        "       -c num_clusters    : number of clusters (must be > 1)\n"
        "       -s size            : size of examined dataset\n"
        "       -n num_coords      : number of coordinates\n"
        "       -t threshold       : threshold value (default : 0.001)\n"
        "       -l loop_threshold  : iterations threshold (default : 10)\n"
        "       -d                 : enable debug mode\n"
        "       -h                 : print this help information\n";
    fprintf(stderr, help, argv0);
    exit(-1);
}

int main(int argc, char **argv)
{
    long i, j, opt;
    extern char* optarg;
    extern int optind;

    long    numClusters=0, numCoords=0, numObjs=0;
    long    rank_numObjs=0;
    int   * membership;     // [rank_numObjs] this array will contain membership information for this rank's objects
    int   * tot_membership; // [numObjs]      this array will contain membership information for all objects
    double * objects;       // [numObjs * numCoords] data  objects
    double * clusters;      // [numClusters * numCoords] cluster center
    double   dataset_size = 0, threshold;
    long    loop_threshold;
    double  io_timing_read;

    /* some default values */
    _debug         = 0;
    threshold      = 0.001;
    loop_threshold = 10;
    numClusters    = 0;

    while ( (opt = getopt(argc,argv,"n:t:l:c:s:dh")) != EOF) {
        switch (opt) {
            case 'c': numClusters = atol(optarg);
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'l': loop_threshold=atol(optarg);
                      break;
            case 's': dataset_size=atof(optarg);
                      break;
            case 'n': numCoords=atol(optarg);
                      break;
            case 'd': _debug = 1;
                      break;
            case 'h':
            default: usage(argv[0]);
                      break;
        }
    }
    if (numClusters <= 1) {
        usage(argv[0]);
    }

    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    numObjs = (dataset_size*1024*1024) / (numCoords*sizeof(double));

    if (numObjs < numClusters) {
        if (rank == 0) printf("Error: number of clusters must be larger than the number of data points to be clustered.\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) printf("dataset_size = %.2f MB    numObjs = %ld    numCoords = %ld    numClusters = %ld\n", dataset_size, numObjs, numCoords, numClusters);

    objects = dataset_generation(numObjs, numCoords, &rank_numObjs);

//    printf("Rank: %d	", rank);
//    printf("Number of objects: %d \n", rank_numObjs);

    // Allocate space for clusters (coordinates of cluster centers)
    clusters = (double*)  malloc(numClusters * numCoords * sizeof(double));

    // The first numClusters elements are selected as initial centers. Only rank 0 needs to calculate this, and later broadcast it to all ranks.
    if (rank == 0) {
        for (i=0; i<numClusters; i++)
            for (j=0; j<numCoords; j++)
                clusters[i*numCoords + j] = objects[i*numCoords + j];

        // check initial cluster centers for repetition 
        if (check_repeated_clusters(numClusters, numCoords, clusters) == 0) {
            printf("Error: some initial clusters are repeated. Please select distinct initial centers\n");
            MPI_Finalize();
            return 1;
        }
        
        printf("Initial cluster centers:\n");
        for (i=0; i<numClusters; i++) {
            printf("(0) clusters[%ld] =",i);
            for (j=0; j<numCoords; j++)
                printf(" %6.6f", clusters[i*numCoords + j]);
            printf("\n");
        }
        
    }

    /*
     * TODO: Broadcast initial cluster positions to all ranks
     */
    MPI_Bcast(clusters, numClusters*numCoords, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // membership: the cluster id for each data object
    membership = (int*) malloc(rank_numObjs * sizeof(int));
    tot_membership = (int*) malloc(numObjs * sizeof(int));

    // start the core computation
    /* 
     * TODO: Fix number of objects that this kmeans function call will process
     */
    kmeans(objects, numCoords, rank_numObjs, numClusters, threshold, loop_threshold, membership, clusters);

    
    if (rank == 0) {    
        printf("Final cluster centers:\n");
        for (i=0; i<numClusters; i++) {
            printf("clusters[%ld] = ",i);
            for (j=0; j<numCoords; j++)
                printf("%6.6f ", clusters[i*numCoords + j]);
            printf("\n");
        }
    }
    

    // Gather membership information from all ranks to tot_membership
    int recvcounts[size], displs[size];
    int r;
    if (rank == 0) {
        /* TODO: Calculate recvcounts and displs, which will be used to gather data from each rank.
         * Hint: recvcounts: number of elements received from each rank
         *       displs: displacement of each rank's data 
         */
	for (r=0; r<size; r++) {
                recvcounts[r] = (r < (numObjs % size)) ? ((numObjs/size) + 1) : (numObjs/size);
                displs[r] = (r == 0) ? 0 : (displs[r - 1] + recvcounts[r - 1]);
	}
    }
    /* 
     * TODO: Broadcast the recvcounts and displs arrays to other ranks.
     */
    MPI_Bcast(recvcounts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs, size, MPI_INT, 0, MPI_COMM_WORLD);

    /*
     * TODO: Gather membership information from every rank. (hint: each rank may send different number of objects)
     */
    MPI_Gatherv(membership, rank_numObjs, MPI_INT, tot_membership, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (_debug && rank == 0)
        for (i = 0; i < numObjs; ++i)
            fprintf(stderr, "%d\n", tot_membership[i]);
    
    free(objects);
    free(membership);
    free(tot_membership);
    free(clusters);

    MPI_Finalize();
    return 0;
}
