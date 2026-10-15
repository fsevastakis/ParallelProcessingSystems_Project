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
    double ans1;	
    /* same as transpose */
             /* TODO: Calculate the euclid_dist of elem=objectId of objects from elem=clusterId from clusters, but for column-base format!!! */
         for(i=0; i<numCoords; i++){     
            ans1 = objects[numObjs*i+objectId]-clusters[numClusters*i+clusterId]; 
            ans += ans1*ans1;
           // ans += (objects[i][objectId]-clusters[i][clusterId])*(objects[i][objectId]-clusters[i][clusterId]);
        }

    return(ans);
}

__global__ static
void find_nearest_cluster(int numCoords,
                          int numObjs,
                          int numClusters,
                          double *deviceobjects,           //  [numCoords][numObjs]
/*                          
                          TODO: If you choose to do (some of) the new centroid calculation here, you will need some extra parameters here (from "update_centroids").
*/                        int *devicenewClusterSize,           //  [numClusters]
                          double *devicenewClusters,    //  [numCoords][numClusters]

                          double *deviceClusters,    //  [numCoords][numClusters]
                          int *deviceMembership,          //  [numObjs]
                          double *devdelta)
{
 
/*The reason the code works with char but not with double has to do with pointer arithmetic and alignment issues.
In C and C++, pointer arithmetic depends on the size of the data type being pointed to. When you use a char* pointer and perform arithmetic on it,
the compiler assumes that each increment or decrement in the pointer will move by one byte, because char is one byte in size. */

extern __shared__ char sharedMemory[];

    // Calculate the size of each array in shared memory
    int deltaSize = blockDim.x * sizeof(int);
    int clustersSize = numClusters * numCoords * sizeof(double);
    int newClustersSize = numClusters * numCoords * sizeof(double);
    int newClustersSizeSize = numClusters * sizeof(int);

    // Pointers to access shared memory
    int *devDelta_ptr = (int *)sharedMemory;
    double *shmemClusters = (double *)(sharedMemory + deltaSize);
    double *shmemNEWClusters = (double *)(sharedMemory + deltaSize + clustersSize);
    int *shmemNEWClustersSIZE = (int *)(sharedMemory + deltaSize + clustersSize + newClustersSize);

     /* TODO: copy me from shared version... */ 
         int i = threadIdx.x;
   	 int b = blockDim.x;
    while(i < numCoords*numClusters) {
            shmemClusters[i] = deviceClusters[i];
            shmemNEWClusters[i] = 0; //new 
            if(i<numClusters)  shmemNEWClustersSIZE[i] = 0; //new 
             i += b;               
	 }
     
    __syncthreads(); 

    int tid = get_tid(); 

    if (tid < numObjs) {
        int   index, i,j;
        double dist, min_dist;
	int objectId , clusterId ;
        int local_tid = threadIdx.x;

        index = 0;
        objectId  = tid;
        clusterId = 0;
        min_dist = euclid_dist_2_transpose(numCoords, numObjs,numClusters,deviceobjects,shmemClusters,objectId,clusterId);

        for (i=1; i<numClusters; i++) {
            objectId  = tid;  
            clusterId = i;   
            dist = euclid_dist_2_transpose(numCoords, numObjs,numClusters,deviceobjects,shmemClusters,objectId,clusterId );
 
            if (dist < min_dist) { /* find the min and its array index */
                min_dist = dist;
                index    = i;
            }
        }

        devDelta_ptr[local_tid]=0;
        if (deviceMembership[tid] != index) {devDelta_ptr[local_tid] = 1;}
      
        deviceMembership[tid] = index;  
     //NEW
       atomicAdd(&shmemNEWClustersSIZE[index],1); 

        for(j =0;j<numCoords;j++) {
            //devicenewClusters[j][index] += deviceobjects[j][tid];
            atomicAdd(&shmemNEWClusters[j*numClusters+index],deviceobjects[j*numObjs+tid]);

        }
 
        __syncthreads();

     //reduction for delta
         i = blockDim.x/2;
        while(i!=0) {
            if(local_tid < i)
               devDelta_ptr[local_tid] +=  devDelta_ptr[local_tid+i];
            __syncthreads();
            i /=2;
        }
        if (local_tid == 0) {atomicAdd(devdelta, devDelta_ptr[0]);}
		    
        //reduction of shared arrays --NEW 
         i = threadIdx.x;
         b = blockDim.x;
        while(i < numCoords*numClusters) {
        	atomicAdd(&devicenewClusters[i],shmemNEWClusters[i]);
            if(i<numClusters) atomicAdd(&(devicenewClusterSize[i]),shmemNEWClustersSIZE[i]);
            i += b;
	    }    
    }
}

__global__ static
void update_centroids(int numCoords,
                          int numClusters,
                          int *devicenewClusterSize,           //  [numClusters]
                          double *devicenewClusters,    //  [numCoords][numClusters]
                          double *deviceClusters)    //  [numCoords][numClusters])
{

    /* TODO: additional steps for calculating new centroids in GPU? */
    int global_tid= get_tid() ; //+1;  // every thread will update one value from clustes array 
    int i,j,done,temp; // lets say i run numClusters and j numCoords
    // with the transpose rows are i and columns are j 
    i =0; // give as the cluster 
    j =0; //give as the Coord 

  if(global_tid <numCoords*numClusters) {   
//    done =0;
//    while(done == 0) { 
//    if (global_tid > numClusters*(j+1)) {
//        j++;
//        }
//     else done = 1 ;         
//    }
//now i have different i,j for every tid  

     i = global_tid%numClusters;
    	
    if(devicenewClusterSize[i]>0) { 
       // deviceClusters[j][i] = devicenewClusters[j][i]/devicenewClusterSize[i];
       // deviceClusters[ j*numClusters +i] = devicenewClusters[j*numClusters+i]/devicenewClusterSize[i];
       deviceClusters[global_tid] = devicenewClusters[global_tid]/devicenewClusterSize[i];
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
    int      i, j, index, loop=0;
    double  delta = 0, *dev_delta_ptr;          /* % of objects change their clusters */
 
   /* TODO: Copy me from transpose version*/
    double  **dimObjects =  (double**)calloc_2d(numCoords,numObjs,sizeof(double));      //calloc_2d(...) -> [numCoords][numObjs]
    double  **dimClusters = (double**)calloc_2d(numCoords,numClusters,sizeof(double));  //calloc_2d(...) -> [numCoords][numClusters]
    double  **newClusters = (double**)calloc_2d(numCoords,numClusters,sizeof(double));  //calloc_2d(...) -> [numCoords][numClusters] 

    printf("\n|-----------Full-offload GPU Kmeans------------|\n\n");
    
    /* TODO: Copy me from transpose version*/
	for(int i =0 ;i< numObjs;i++ )
           for (int j =0;j<numCoords;j++)
               dimObjects[j][i] = objects[i*numCoords+j]; //objetcs[i][j]

    
    double *deviceObjects;
    double *deviceClusters, *devicenewClusters;
    int *deviceMembership;
    int *devicenewClusterSize; /* [numClusters]: no. objects assigned in each new cluster */
    // devicenewClusters and devicenewClusters is new!
    
    /* pick first numClusters elements of objects[] as initial cluster centers*/
    for (i = 0; i < numCoords; i++) {
        for (j = 0; j < numClusters; j++) {
            dimClusters[i][j] = dimObjects[i][j];
        }
    }
	
    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;
    
    timing = wtime() - timing;
    printf("t_alloc: %lf ms\n\n", 1000*timing);
    timing = wtime(); 
    const unsigned int numThreadsPerClusterBlock = (numObjs > blockSize)? blockSize: numObjs;
    const unsigned int numClusterBlocks =(numObjs + numThreadsPerClusterBlock -1) / numThreadsPerClusterBlock; /* TODO: Calculate Grid size, e.g. number of blocks. */
	/*	Define the shared memory needed per block.
    	- BEWARE: We can overrun our shared memory here if there are too many
    	clusters or too many coordinates! 
    	- This can lead to occupancy problems or even inability to run. 
    	- Your exercise implementation is not requested to account for that (e.g. always assume deviceClusters fit in shmemClusters */
    const unsigned int clusterBlockSharedDataSize = numThreadsPerClusterBlock*sizeof(int) + 2*numCoords*numClusters*sizeof(double) + numClusters*sizeof(int); //new 

    cudaDeviceProp deviceProp;
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);

    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
        error("Your CUDA hardware has insufficient block shared memory to hold all cluster centroids\n");
    }
           
    checkCuda(cudaMalloc(&deviceObjects, numObjs*numCoords*sizeof(double)));
    checkCuda(cudaMalloc(&deviceClusters, numClusters*numCoords*sizeof(double)));
    checkCuda(cudaMalloc(&devicenewClusters, numClusters*numCoords*sizeof(double))); //nre
    checkCuda(cudaMalloc(&devicenewClusterSize, numClusters*sizeof(int))); //new 
    checkCuda(cudaMalloc(&deviceMembership, numObjs*sizeof(int)));
    checkCuda(cudaMalloc(&dev_delta_ptr, sizeof(double)));
 
    timing = wtime() - timing;
    printf("t_alloc_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime();  
       
    checkCuda(cudaMemcpy(deviceObjects, dimObjects[0],
              numObjs*numCoords*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceMembership, membership,
              numObjs*sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceClusters, dimClusters[0],
                  numClusters*numCoords*sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(devicenewClusterSize, 0, numClusters*sizeof(int))); //new 
    free(dimObjects[0]);
      
    timing = wtime() - timing;
    printf("t_get_gpu: %lf ms\n\n", 1000*timing);
    timing = wtime();   
    
    do {
        timing_internal = wtime(); 
        checkCuda(cudaMemset(dev_delta_ptr, 0, sizeof(double)));          
 //	checkCuda(cudaMemset(devicenewClusterSize, 0, numClusters*sizeof(int)));
 //       checkCuda(cudaMemset(devicenewClusters, 0, numClusters*numCoords*sizeof(int)));  
	
        /* TODO: change invocation if extra parameters needed */
        find_nearest_cluster
            <<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numCoords, numObjs, numClusters,
             deviceObjects, devicenewClusterSize, devicenewClusters, deviceClusters, deviceMembership, dev_delta_ptr);
        

        cudaDeviceSynchronize(); checkLastCudaError();
		//printf("Kernels complete for itter %d, updating data in CPU\n", loop);
    
    	/* TODO: Copy dev_delta_ptr to &delta
        checkCuda(cudaMemcpy(...)); */
        checkCuda(cudaMemcpy(&delta,  dev_delta_ptr  ,sizeof(double), cudaMemcpyDeviceToHost));
         // here i dont send anymore the deviceMembership  to CPU but use them throu global Memory in gpu
         //by the second global kernel function 

     	const unsigned int update_centroids_block_sz = (numCoords* numClusters > blockSize) ? blockSize: numCoords* numClusters;  /* TODO: can use different blocksize here if deemed better */
     	const unsigned int update_centroids_dim_sz = (numCoords* numClusters +update_centroids_block_sz -1)/update_centroids_block_sz ; /* TODO: calculate dim for "update_centroids" and fire it  --new*/ 
     	update_centroids<<< update_centroids_dim_sz, update_centroids_block_sz, 0 >>>
        (numCoords, numClusters, devicenewClusterSize, devicenewClusters, deviceClusters);  
        cudaDeviceSynchronize(); checkLastCudaError();   
                       
        delta /= numObjs;
       	//printf("delta is %f - ", delta);
        loop++; 
        //printf("completed loop %d\n", loop);
		timing_internal = wtime() - timing_internal; 
		if ( timing_internal < timer_min) timer_min = timing_internal; 
		if ( timing_internal > timer_max) timer_max = timing_internal; 
	} while (delta > threshold && loop < loop_threshold);
                  	
    checkCuda(cudaMemcpy(membership, deviceMembership,
                 numObjs*sizeof(int), cudaMemcpyDeviceToHost));     
    checkCuda(cudaMemcpy(dimClusters[0], deviceClusters,
                 numClusters*numCoords*sizeof(double), cudaMemcpyDeviceToHost));  
    //reverse transpose here                                
	for (i=0; i<numClusters; i++) {
		for (j=0; j<numCoords; j++) {
		    clusters[i*numCoords + j] = dimClusters[j][i];
		}
	}
	
    timing = wtime() - timing;

//only for debug
	printf("Final  cluster centers:\n", loop);
        for (i=0; i<numClusters; i++) {
            printf("clusters[%ld] = ",i);
            for (j=0; j<numCoords; j++)
                printf("%f ",dimClusters[j][ i]);
            printf("\n");
        }

    printf("nloops = %d  : total = %lf ms\n\t-> t_loop_avg = %lf ms\n\t-> t_loop_min = %lf ms\n\t-> t_loop_max = %lf ms\n\n|-------------------------------------------|\n", 
    	loop, 1000*timing, 1000*timing/loop, 1000*timer_min, 1000*timer_max);

	char outfile_name[1024] = {0}; 
	sprintf(outfile_name, "Execution_logs/silver1-V100_Sz-%lu_Coo-%d_Cl-%d.csv", numObjs*numCoords*sizeof(double)/(1024*1024), numCoords, numClusters);
	FILE* fp = fopen(outfile_name, "a+");
	if(!fp) error("Filename %s did not open succesfully, no logging performed\n", outfile_name); 
	fprintf(fp, "%s,%d,%lf,%lf,%lf\n", "All_GPU", blockSize, timing/loop, timer_min, timer_max);
	fclose(fp); 
	
    checkCuda(cudaFree(deviceObjects));
    checkCuda(cudaFree(deviceClusters));
    checkCuda(cudaFree(devicenewClusters));
    checkCuda(cudaFree(devicenewClusterSize));
    checkCuda(cudaFree(deviceMembership));

    return;
}


