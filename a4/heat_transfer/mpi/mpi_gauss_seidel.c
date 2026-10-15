#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"
#include "utils.h"


int main(int argc, char ** argv) {
    int rank,size;
    int global[2],local[2]; //global matrix dimensions and local matrix dimensions (2D-domain, 2D-subdomain)
    int global_padded[2];   //padded global matrix dimensions (if padding is not needed, global_padded=global)
    int grid[2];            //processor grid dimensions
    int i,j,t;
    int global_converged=0,converged=0; //flags for convergence, global and per process
    MPI_Datatype dummy;     //dummy datatype used to align user-defined datatypes in memory
    double omega; 			//relaxation factor - useless for Jacobi
    int f;
    struct timeval tts,ttf,tcs,tcf;   //Timers: total-> tts,ttf, computation -> tcs,tcf
    double ttotal=0,tcomp=0,total_time,comp_time;
    
    double ** U, ** u_current, ** u_previous, ** swap; //Global matrix, local current and previous matrices, pointer to swap between current and previous
   

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    //----Read 2D-domain dimensions and process grid dimensions from stdin----//

    if (argc!=5) {
        fprintf(stderr,"Usage: mpirun .... ./exec X Y Px Py");
        exit(-1);
    }
    else {
        global[0]=atoi(argv[1]);
        global[1]=atoi(argv[2]);
        grid[0]=atoi(argv[3]);
        grid[1]=atoi(argv[4]);
    }

    //----Create 2D-cartesian communicator----//
	//----Usage of the cartesian communicator is optional----//

    MPI_Comm CART_COMM;         //CART_COMM: the new 2D-cartesian communicator
    int periods[2]={0,0};       //periods={0,0}: the 2D-grid is non-periodic
    int rank_grid[2];           //rank_grid: the position of each process on the new communicator
		
    MPI_Cart_create(MPI_COMM_WORLD,2,grid,periods,0,&CART_COMM);    //communicator creation
    MPI_Cart_coords(CART_COMM,rank,2,rank_grid);	                //rank mapping on the new communicator

    //----Compute local 2D-subdomain dimensions----//
    //----Test if the 2D-domain can be equally distributed to all processes----//
    //----If not, pad 2D-domain----//

    int pad[2];		// used later for the jacobi-for-loops' boundries
    for (i=0;i<2;i++) {
        if (global[i]%grid[i]==0) {
            local[i]=global[i]/grid[i];
            global_padded[i]=global[i];
            pad[i]=0;
	}
        else {
            local[i]=(global[i]/grid[i])+1;
            global_padded[i]=local[i]*grid[i];
	    pad[i]=global_padded[i]-global[i];
        }
    }


	//Initialization of omega
    omega=2.0/(1+sin(3.14/global[0]));

    //----Allocate global 2D-domain and initialize boundary values----//
    //----Rank 0 holds the global 2D-domain----//
    if (rank==0) {
        U=allocate2d(global_padded[0],global_padded[1]);   
        init2d(U,global[0],global[1]);
    }

    //----Allocate local 2D-subdomains u_current, u_previous----//
    //----Add a row/column on each size for ghost cells----//

    u_previous=allocate2d(local[0]+2,local[1]+2);
    u_current=allocate2d(local[0]+2,local[1]+2);   
       
    //----Distribute global 2D-domain from rank 0 to all processes----//
         
 	//----Appropriate datatypes are defined here----//
	/*****The usage of datatypes is optional*****/
    
    //----Datatype definition for the 2D-subdomain on the global matrix----//

    MPI_Datatype global_block;
    MPI_Type_vector(local[0],local[1],global_padded[1],MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&global_block);
    MPI_Type_commit(&global_block);

    //----Datatype definition for the 2D-subdomain on the local matrix----//

    MPI_Datatype local_block;
    MPI_Type_vector(local[0],local[1],local[1]+2,MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&local_block);
    MPI_Type_commit(&local_block);

    //----Rank 0 defines positions and counts of local blocks (2D-subdomains) on global matrix----//
    int * scatteroffset, * scattercounts;
    double ** U_start;
    if (rank==0) {
	U_start=&U[0][0];
        scatteroffset=(int*)malloc(size*sizeof(int));
        scattercounts=(int*)malloc(size*sizeof(int));
        for (i=0;i<grid[0];i++)
            for (j=0;j<grid[1];j++) {
                scattercounts[i*grid[1]+j]=1;
                scatteroffset[i*grid[1]+j]=(local[0]*local[1]*grid[1]*i+local[1]*j);
            }
    }


    //----Rank 0 scatters the global matrix----//
    
    //----Rank 0 scatters the global matrix----//

	//*************TODO*******************//



	/*Fill your code here*/



	/*Make sure u_current and u_previous are
		both initialized*/

    MPI_Scatterv(U_start, scattercounts, scatteroffset, global_block, &u_previous[1][1], 1, local_block, 0, MPI_COMM_WORLD);
    MPI_Scatterv(U_start, scattercounts, scatteroffset, global_block, &u_current[1][1], 1, local_block, 0, MPI_COMM_WORLD);

     //************************************//


    if (rank==0)
        free2d(U);

 
     
	//----Define datatypes or allocate buffers for message passing----//

	//*************TODO*******************//



	/*Fill your code here*/

    
    // Halo datatypes: h_block = one interior row (local[1] contiguous doubles),
    // v_block = one interior column (local[0] doubles, one per row of length local[1]+2).
    MPI_Datatype h_block;
    MPI_Type_vector(1,local[1],0,MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&h_block);
    MPI_Type_commit(&h_block);

    MPI_Datatype v_block;
    MPI_Type_vector(local[0],1,local[1]+2,MPI_DOUBLE,&dummy);
    MPI_Type_create_resized(dummy,0,sizeof(double),&v_block);
    MPI_Type_commit(&v_block);


	//************************************//


    //----Find the 4 neighbors with which a process exchanges messages----//

	//*************TODO*******************//
    // 99 marks a missing neighbour (process on the global boundary). Fine for the <= 64
    // processes we ran; MPI_PROC_NULL would be the general choice.
    int north = 99, south = 99, east = 99, west = 99;



	/*Fill your code here*/


	/*Make sure you handle non-existing
		neighbors appropriately*/

    int coord[2];
    if (rank_grid[0] != 0) {coord[0] = rank_grid[0]-1; coord[1] = rank_grid[1]; MPI_Cart_rank(CART_COMM, coord, &north);}
    if (rank_grid[0] != grid[0] - 1) {coord[0] = rank_grid[0]+1; coord[1] = rank_grid[1]; MPI_Cart_rank(CART_COMM, coord, &south);}
    if (rank_grid[1] != 0) {coord[0] = rank_grid[0]; coord[1] = rank_grid[1]-1; MPI_Cart_rank(CART_COMM, coord, &west);}
    if (rank_grid[1] != grid[1] - 1) {coord[0] = rank_grid[0]; coord[1] = rank_grid[1]+1; MPI_Cart_rank(CART_COMM, coord, &east);}

	//************************************//
//printf("My rank is %d, my coords are (%d, %d) and my neighbours are: %d (north), %d (south), %d (east), %d (west) \n", rank, rank_grid[0], rank_grid[1], north, south, east, west);



    //---Define the iteration ranges per process-----//
	//*************TODO*******************//

    int i_min,i_max,j_min,j_max;



	/*Fill your code here*/

	
    // Local row/column 1 of a process on the north/west edge is the fixed global boundary,
    // so it starts at 2. On the south/east edge the last pad[] rows/columns are padding and
    // the one before them is the boundary, so it stops at local - pad - 1.
    (rank_grid[0] != 0) ? (i_min = 1) : (i_min = 2);	// north boundry ?
    (rank_grid[1] != 0) ? (j_min = 1) : (j_min = 2);	// west boundry ?
    (rank_grid[0] != grid[0] - 1) ? (i_max = local[0]) : (i_max = local[0] - pad[0] - 1);	// south boundry ? 
    (rank_grid[1] != grid[1] - 1) ? (j_max = local[1]) : (j_max = local[1] - pad[1] - 1); 	// east boundry ? 


	/*Three types of ranges:
		-internal processes
		-boundary processes
		-boundary processes and padded global array
	*/


//printf("My rank is %d and my boundries are: i -> (%d, %d), j -> (%d, %d) \n", rank, i_min, i_max, j_min, j_max);


	//************************************//

    MPI_Barrier(MPI_COMM_WORLD);


 	//----Computational core----//   
	gettimeofday(&tts, NULL);
    #ifdef TEST_CONV
    for (t=0;t<T && !global_converged;t++) {
    #endif
    #ifndef TEST_CONV
    #undef T
    #define T 256
    for (t=0;t<T;t++) {
    #endif


	 	//*************TODO*******************//
     
 

		/*Fill your code here*/


		/*Compute and Communicate*/

		/*Add appropriate timers for computation*/


    // Unlike Jacobi, swap first: u_previous now holds the last iteration's values
    // and u_current is the array this iteration writes into.
    swap=u_previous;
    u_previous=u_current;
    u_current=swap;


    int lastrow = local[0];
    int lastcol = local[1];

    /* Gauss-Seidel reads the NEW values of its north/west neighbours (u_current)
     * and the OLD values of its south/east neighbours (u_previous). So each process:
     *   1. sends its old first row/column to north/west (their south/east halo),
     *   2. receives the old halo from south/east into u_previous,
     *   3. receives the new halo from north/west into u_current; this blocks until
     *      those neighbours have finished the current iteration (wavefront),
     *   4. computes, then sends its new last row/column to south/east (their step 3).
     * All messages use tag 0; MPI keeps per-pair order, so they match up iteration by iteration. */
    MPI_Request requests[6];
    int count = 0;

    // Senders: old values //
    if (north != 99) {
	MPI_Isend(&u_previous[1][1], 1, h_block, north, 0, MPI_COMM_WORLD, &requests[count]);
	count++;
    }
    if (west != 99) {
	MPI_Isend(&u_previous[1][1], 1, v_block, west, 0, MPI_COMM_WORLD, &requests[count]);
	count++;
    }

    // Receivers: old values from south/east //
    if (south != 99) {
	MPI_Irecv(&u_previous[lastrow+1][1], 1, h_block, south, 0, MPI_COMM_WORLD, &requests[count]);
	count++;
    }
    if (east != 99) {
	MPI_Irecv(&u_previous[1][lastcol+1], 1, v_block, east, 0, MPI_COMM_WORLD, &requests[count]);
	count++;
    }

    // Receivers: new values from north/west, into u_current (a row from north, a column from west) //
    if (north != 99) {
        MPI_Irecv(&u_current[0][1], 1, h_block, north, 0, MPI_COMM_WORLD, &requests[count]);
        count++;
    }
    if (west != 99) {
        MPI_Irecv(&u_current[1][0], 1, v_block, west, 0, MPI_COMM_WORLD, &requests[count]);
        count++;
    }

    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);


    gettimeofday(&tcs,NULL);

    //Gauss-Seidel
    int new_count = 0; MPI_Request new_requests[2];
    int i,j;
    for (i=i_min; i<=i_max; i++)
	for (j=j_min; j<=j_max; j++)
            u_current[i][j]=u_previous[i][j]+(u_current[i-1][j]+u_previous[i+1][j]+u_current[i][j-1]+u_previous[i][j+1]-4*u_previous[i][j])*omega/4.0;

    gettimeofday(&tcf,NULL);

    // Senders: new values, for the south/east neighbours' step 3 //
    if (south != 99) {
	MPI_Isend(&u_current[lastrow][1], 1, h_block, south, 0, MPI_COMM_WORLD, &new_requests[new_count]);
	new_count++;
    }
    if (east != 99) {
        MPI_Isend(&u_current[1][lastcol], 1, v_block, east, 0, MPI_COMM_WORLD, &new_requests[new_count]);
        new_count++;
    }
    MPI_Waitall(new_count, new_requests, MPI_STATUSES_IGNORE);

    tcomp+=(tcf.tv_sec-tcs.tv_sec)+(tcf.tv_usec-tcs.tv_usec)*0.000001;



		#ifdef TEST_CONV
	        if (t%C==0) {
			//*************TODO**************//
			/*Test convergence*/
			converged = converge(u_previous,u_current,i_min,i_max,j_min,j_max);
			MPI_Allreduce(&converged,&global_converged,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
//		printf("Round %d: Rank %d, converged %d \n", t, rank, converged);
		}
		#endif


		//************************************//
 
         
    // Keeps all processes in step every iteration. Not needed for correctness (the halo
    // exchange already synchronises neighbours), and it is counted in the total time.
    MPI_Barrier(MPI_COMM_WORLD);
       
    }
//    printf("Rank %d, converged %d \n", rank, converged);
    gettimeofday(&ttf,NULL);

    ttotal=(ttf.tv_sec-tts.tv_sec)+(ttf.tv_usec-tts.tv_usec)*0.000001;

    MPI_Reduce(&ttotal,&total_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    MPI_Reduce(&tcomp,&comp_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);



    //----Rank 0 gathers local matrices back to the global matrix----//
   
    if (rank==0) {
            U=allocate2d(global_padded[0],global_padded[1]);
	    U_start=&U[0][0];
    }


	//*************TODO*******************//



	/*Fill your code here*/


    MPI_Gatherv(&(u_current[1][1]), 1, local_block, U_start, scattercounts, scatteroffset, global_block, 0, MPI_COMM_WORLD);


     //************************************//


	
   

	//----Printing results----//

	//**************TODO: Change "Jacobi" to "GaussSeidelSOR" or "RedBlackSOR" for appropriate printing****************//
    if (rank==0) {
        printf("GaussSeidelSOR X %d Y %d Px %d Py %d Iter %d ComputationTime %lf TotalTime %lf midpoint %lf\n",global[0],global[1],grid[0],grid[1],t,comp_time,total_time,U[global[0]/2][global[1]/2]);
	
        #ifdef PRINT_RESULTS
        char * s=malloc(50*sizeof(char));
        sprintf(s,"resGaussSeidelSORMPI_%dx%d_%dx%d",global[0],global[1],grid[0],grid[1]);
        fprint2d(s,U,global[0],global[1]);
        free(s);
        #endif

    }
    MPI_Finalize();
    return 0;
}
