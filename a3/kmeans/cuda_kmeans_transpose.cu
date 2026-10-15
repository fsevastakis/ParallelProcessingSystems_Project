#include <stdio.h>
#include <stdlib.h>

#include "kmeans.h"
#include "alloc.h"
#include "error.h"

#ifdef __CUDACC__
inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        // cudaGetErrorString() isn't always very helpful. Look up the error
        // number in the cudaError enum in driver_types.h in the CUDA includes
        // directory for a better explanation.
        error("CUDA Error %d: %s\n", e, cudaGetErrorString(e));
    }
}

inline void checkLastCudaError() {
    checkCuda(cudaGetLastError());
}
#endif

__device__ int get_tid(){
         int globalId = threadIdx.x + blockIdx.x * blockDim.x;
      return globalId; /* TODO: copy me from naive version... */
}

/* square of Euclid distance between two multi-dimensional points using column-base format */
__host__ __device__ inline static
 double euclid_dist_2_transpose(int numCoords,
                    int    numObjs,
                    int    numClusters,
                    double *objects,     // [numCoords][numObjs]
                    double *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    int i;
    double ans=0.0;

        /* TODO: Calculate the euclid_dist of elem=objectId of objects from elem=clusterId from clusters, but for column-base format!!! */
     for(i=0; i<numCoords; i++){ 
             // in the serial we have a for loop for all dimensions ,now we just just take the distance between object and cluster 
            ans += (objects[objectId+i*numObjs]-clusters[clusterId+i*numClusters])*(objects[objectId+i*numObjs]-clusters[clusterId+i*numClusters]);
           // ans += (objects[i][objectId]-clusters[i][clusterId])*(objects[i][objectId]-clusters[i][clusterId]);
        }

    return(ans);
}


__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          double *objects,           //  [numCoords][numObjs]
                          double *deviceClusters,    //  [numCoords][numClusters]
                          int *deviceMembership,          //  [numObjs]
                          double *devdelta)
{
	/* TODO: copy me from naive version... */
     /* Get the global ID of the thread. */
    int tid = get_tid();

        /* TODO: Maybe something is missing here... should all threads run this? */
    if (tid < numObjs) { // max tid we can have 
        int   index, i;
        double dist, min_dist;
        int objectId , clusterId ;
        int local_tid = threadIdx.x; // eg local_tid take values :0,1,2,3,0,1,2,3,... global_tid take values:0,1,2,3,4,5,6,7,...
        extern __shared__ int devDelta_arr[]; // extern because we dont know how many elemetns has

        /* find the cluster id that has min distance to object */
        index = 0;
        /* TODO: call min_dist = euclid_dist_2(...) with correct objectId/clusterId */
        objectId  = tid; // every CUDA core have one tid  (global_tid)
        clusterId = 0; // just for the first cluster to initialize min_dist ! 
        min_dist = euclid_dist_2_transpose(numCoords, numObjs,numClusters,objects,deviceClusters,objectId,clusterId);
        
        for (i=1; i<numClusters; i++) {
            /* TODO: call dist = euclid_dist_2(...) with correct objectId/clusterId */
              objectId  = tid; // we are in the same CUDA core so the same object --this line it doesnt need it 
              clusterId = i;  // we will comptude for this one object(tid) the distance from all  clusters 
              dist = euclid_dist_2_transpose(numCoords, numObjs,numClusters,objects,deviceClusters,objectId,clusterId );
            /* no need square root */
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }
       
         devDelta_arr[local_tid]=0; 
       if (deviceMembership[tid] != index) {
                /* TODO: Maybe something is missing here... is this write safe? */
              // (*devdelta)+= 1.0; //what will hapend if 2 differnts tid go to do devdelta++ in the same time???
             // here we want atomic , but we choose better solution with shared memory array
               devDelta_arr[local_tid] += 1;
        }

        /* assign the deviceMembership to object objectId */
        deviceMembership[tid] = index;
      
        __syncthreads();
         i = blockDim.x/2;
        while(i!=0) { 
            if(local_tid < i) 
               devDelta_arr[local_tid] +=  devDelta_arr[local_tid+i];
            __syncthreads();
            i /=2;
        }
        if (local_tid == 0) { 
            atomicAdd(devdelta, devDelta_arr[0]);
        }

    }
}

//
//  ----------------------------------------
//  DATA LAYOUT
//
//  objects         [numObjs][numCoords]
//  clusters        [numClusters][numCoords]
//  dimObjects      [numCoords][numObjs]
//  dimClusters     [numCoords][numClusters]
//  newClusters     [numCoords][numClusters]
//  deviceObjects   [numCoords][numObjs]
//  deviceClusters  [numCoords][numClusters]
//  ----------------------------------------
//
/* return an array of cluster centers of size [numClusters][numCoords]       */            
void kmeans_gpu(	double *objects,      /* in: [numObjs][numCoords] */
		               	int     numCoords,    /* no. features */
		               	int     numObjs,      /* no. objects */
		               	int     numClusters,  /* no. clusters */
		               	double   threshold,    /* % objects change membership */
		               	long    loop_threshold,   /* maximum number of iterations */
		               	int    *membership,   /* out: [numObjs] */
						double * clusters,   /* out: [numClusters][numCoords] */
						int blockSize)  
{
    double timing = wtime(), timing_internal, timer_min = 1e42, timer_max = 0; 
	int    loop_iterations = 0; 
     double timing_gpu_part,timing_cpu_gpu_transfers,timing_gpu_cpu_transfers,timing_cpu_part;////////////////////////// new
     double t1 =0.0,t2=0.0,t3=0.0,t4=0.0;
     int      i, j, index, loop=0;
     int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    double  delta = 0, *dev_delta_ptr;          /* % of objects change their clusters */
    
    /* TODO: Transpose dims */
    double  **dimObjects =  (double**)calloc_2d(numCoords,numObjs,sizeof(double));      //calloc_2d(...) -> [numCoords][numObjs]
    double  **dimClusters = (double**)calloc_2d(numCoords,numClusters,sizeof(double));  //calloc_2d(...) -> [numCoords][numClusters]
    double  **newClusters = (double**)calloc_2d(numCoords,numClusters,sizeof(double));  //calloc_2d(...) -> [numCoords][numClusters]    

    double *deviceObjects;
    double *deviceClusters;
    int *deviceMembership;

    printf("\n|-----------Transpose GPU Kmeans------------|\n\n");


    printf("####################################");

    //  TODO: Copy objects given in [numObjs][numCoords] layout to new [numCoords][numObjs] layout
    // input  objects [numObjs][numCoords]  --transpose-->  ouput dimObjects  [numCoords][numObjs]
        for(int i =0 ;i< numObjs;i++ )
           for (int j =0;j<numCoords;j++){
               dimObjects[j][i] = objects[i*numCoords+j]; //objetcs[i][j]
}
	
    /* pick first numClusters elements of objects[] as initial cluster centers*/
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }
	
    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL); 
    
    timing = wtime() - timing;
    printf("t_alloc: %lf ms\n\n", 1000*timing);
    timing = wtime(); 

    /* TODO: Calculate Grid size, e.g. number of blocks. */ 
    const unsigned int numThreadsPerClusterBlock = (numObjs > blockSize)? blockSize: numObjs;
    const unsigned int numClusterBlocks = (numObjs + numThreadsPerClusterBlock - 1) / numThreadsPerClusterBlock;  //Grid Size= Threads per Block/Total Number of Threads -new 
    const unsigned int clusterBlockSharedDataSize = numThreadsPerClusterBlock*sizeof(int); // is zero because we dont use shared memory   or we must see what we want in shared memory
    //actually the only shared variable we want all blocks see is delta so we put only that in mainmemory  one for every thread

// this correct because we just want to malloc "lines" of memory in GPU just with correct size       
    checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(double)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(double)));
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&dev_delta_ptr, sizeof(double)));
    timing = wtime() - timing;
    printf("t_alloc_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime(); 
   
/*This line of code is copying the contents of the 2D array dimObjects from the host to the device,
    starting from the first element of the array (dimObjects[0]). The data size being copied is numObjs * numCoords * sizeof(double) bytes. 
    After this operation, the data is available in the deviceObjects array on the GPU. */ 
    checkCuda(cudaMemcpy(deviceObjects, dimObjects[0], numObjs*numCoords*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership, numObjs*sizeof(int), cudaMemcpyHostToDevice));

    timing = wtime() - timing;
    printf("t_get_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime();   
    
    do {
    	timing_internal = wtime();

		/* GPU part: calculate new memberships */
         timing_cpu_gpu_transfers = wtime();		        
        /* TODO: Copy clusters to deviceClusters checkCuda(cudaMemcpy(...)); */
        checkCuda(cudaMemcpy(deviceClusters, dimClusters[0], numClusters*numCoords*sizeof(double), cudaMemcpyHostToDevice)); //new 
        checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(double)));          
         timing_cpu_gpu_transfers  =  wtime() - timing_cpu_gpu_transfers;
	 t1 +=timing_cpu_gpu_transfers;

		//printf("Launching find_nearest_cluster Kernel with grid_size = %d, block_size = %d, shared_mem = %d KB\n", numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize/1000);
 timing_gpu_part = wtime();       
 find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters, deviceObjects, deviceClusters, deviceMembership, dev_delta_ptr);

        cudaDeviceSynchronize(); checkLastCudaError();
		//printf("Kernels complete for itter %d, updating data in CPU\n", loop);
	 timing_gpu_part =wtime() - timing_gpu_part ;//new
	 t2 += timing_gpu_part;

          timing_gpu_cpu_transfers = wtime();	
		/* TODO: Copy deviceMembership to membership
        checkCuda(cudaMemcpy(...)); */
checkCuda(cudaMemcpy(membership,  deviceMembership ,numObjs*sizeof(int), cudaMemcpyDeviceToHost)); //new
    
    	/* TODO: Copy dev_delta_ptr to &delta
        checkCuda(cudaMemcpy(...)); */
checkCuda(cudaMemcpy(&delta,  dev_delta_ptr  ,sizeof(double), cudaMemcpyDeviceToHost));//new !!the delta is not pointer so want by value
  timing_gpu_cpu_transfers  = wtime() - timing_gpu_cpu_transfers;
  t3 +=timing_gpu_cpu_transfers;

		/* CPU part: Update cluster centers*/
          timing_cpu_part = wtime();  		
        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
            index = membership[i];
			
            /* update new cluster centers : sum of objects located within */
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[j][index] += objects[i*numCoords + j];
        }
 
        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 0)
                    dimClusters[j][i] = newClusters[j][i] / newClusterSize[i];
                newClusters[j][i] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }

        delta /= numObjs;
       	//printf("delta is %f - ", delta);
        loop++; 
         timing_cpu_part =wtime() - timing_cpu_part ;//new
	 t4 += timing_cpu_part;
        //printf("completed loop %d\n", loop);
		timing_internal = wtime() - timing_internal; 
		if ( timing_internal < timer_min) timer_min = timing_internal; 
		if ( timing_internal > timer_max) timer_max = timing_internal; 
	} while (delta > threshold && loop < loop_threshold);
    
    /*TODO: Update clusters using dimClusters. Be carefull of layout!!! clusters[numClusters][numCoords] vs dimClusters[numCoords][numClusters] */
    // input dimClusters[numCoords][numClusters]  ---transpose--->  ouput clutsers[numClusters][cumCoords]
  	for( i =0 ;i<numClusters;i++ )
           for ( j =0;j<numCoords;j++)
               clusters[i*numCoords+j] =  dimClusters[j][i]; //!beware clusters is 1D   clusters[i][j] =  dimClusters[j][i] 
	
    timing = wtime() - timing;
   //only for debug
   //      printf("Final  cluster centers:\n", loop);
   //      for (i=0; i<numClusters; i++) {
   //          printf("clusters[%ld] = ",i);
   //          for (j=0; j<numCoords; j++)
   //              printf("%6.2f ",clusters[i*numCoords + j]); 
   //            //   printf("%6.2f ",dimClusters[j][i]);
   //          printf("\n");
   //      }

	

    printf("nloops = %d  : total = %lf ms\n\t-> t_loop_avg = %lf ms\n\t-> t_loop_min = %lf ms\n\t-> t_loop_max = %lf ms\n\n|-------------------------------------------|\n", 
    	loop, 1000*timing, 1000*timing/loop, 1000*timer_min, 1000*timer_max);

    // Print the values
    printf("timing_gpu_part: %lf\n",1000*t1);
    printf("timing_cpu_gpu_transfers: %lf\n",1000*t2);
    printf("timing_gpu_cpu_transfers: %lf\n", 1000*t3);
    printf("timing_cpu_part: %lf\n", 1000*t4);
    // end of the print values


	char outfile_name[1024] = {0}; 
	sprintf(outfile_name, "Execution_logs/silver1-V100_Sz-%lu_Coo-%d_Cl-%d.csv", numObjs*numCoords*sizeof(double)/(1024*1024), numCoords, numClusters);
	FILE* fp = fopen(outfile_name, "a+");
	if(!fp) error("Filename %s did not open succesfully, no logging performed\n", outfile_name); 
	fprintf(fp, "%s,%d,%lf,%lf,%lf\n", "Transpose", blockSize, timing/loop, timer_min, timer_max);
	fclose(fp); 
	
    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(deviceMembership));

    free(dimObjects[0]);
    free(dimObjects);
    free(dimClusters[0]);
    free(dimClusters);
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return;
}


