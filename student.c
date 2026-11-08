#include "utils_1d.h"

#include <stdlib.h>   // standard library: dynamic memory management (malloc, free)
#include <math.h>     // math library: for error calculations
#include <mpi.h>      // mpi librar: message passing types, functions, communicator def.
/* Implement the MPI solver here. You may add includes and helper functions.
 * Keep the explicit finite difference scheme and the public signature.
 * MPI is initialized/finalized by the driver. The full result and cumulative
 * error must be available on rank 0 when this function returns.
 */

 /**
 * @brief Solve the heat equation using an explicit finite difference scheme.
 *
 * This functions solves the heat equation iterating over ITER time steps.
 *
 * @param heat_array Array of `double` values representing the temperatures of the points.
 * @param err Pointer to a `double` value representing the error in the simulation.
 * @param num_pts Number of points in the simulation.
 *
 * @return 0 on success, -1 on error.
 */
int heat_eqn_solver_1d(double *heat_array, double *err, size_t num_pts){

  int my_rank = 0;  // process rank (0 - comm_size - 1)

  int comm_size = 1;   // number of running MPI processes

  // check for invalid input pointers or zero grid point count
  if (heat_array == NULL || err == NULL || num_pts == 0) {
    return -1;
  }

  // get the rank of this process
  if (MPI_Comm_rank(MPI_COMM_WORLD, &my_rank) != MPI_SUCCESS) {
    return -1;
  }

  //query the communicator to know how many processes exist
  if (MPI_Comm_size(MPI_COMM_WORLD, &comm_size) != MPI_SUCCESS) {
    return -1;  /* if MPI call fails */
  }

  //make sure every process has at least one point
  if ((size_t)comm_size > num_pts) {
    return -1;
  }

  //Setup the parameters
  double dx = 1.0/(num_pts+1);    //calculates spatial grid spacing
  double dt = 0.5*dx*dx;          //calculate the stable time step size
  size_t beg = 1;

  size_t base_num = num_pts / (size_t)comm_size;    //calculate base number of pointa assigned to each rank

  size_t rem_pts = num_pts % (size_t)comm_size;     //calculate remaining points that do not divide evenly across processes

  size_t local_num = base_num + ((size_t)my_rank < rem_pts ? 1 : 0);     // number of points this rank computes (ranks below rem_pts take one extra point)

  size_t end = beg + local_num - 1;

  size_t start_indx = 1 + (size_t)my_rank * base_num + ((size_t)my_rank < rem_pts ? (size_t)my_rank : rem_pts);  // 1-based start index where this ranks partition begins in heat_array

  double *local_curr = (double *)malloc((local_num + 2) * sizeof(double));  // allocate local working array sized for local_num points + 2 ghost/halo cells

  // Utilize an auxiliary array to store the new temperatures
  double *aux = (double *) malloc((local_num + 2) * sizeof(double));    // + 2 ghost/halo cells

  //check aux memory allocation
  if (aux == NULL) {
    free(local_curr);
    return -1;     //allocation fail
  }
  //check local_curr allocation
  if (local_curr == NULL) {
    //free aux array prevents leaks
    free(aux);

    return -1;   //allocation error
  }

  //fill local_curr array with values from heat_array
  for (size_t j = beg; j <= end; j++) {

    local_curr[j] = heat_array[start_indx + (j - beg)];   //copy initial temp point from heat_array into local_curr array
  }

  local_curr[0] = 0.0;    // left ghost cell at index 0 at 0.0

  local_curr[end + 1] = 0.0;    // right ghost cell at index end + 1 to 0.0

  int left_neigh_rank;    //determine left neighbor rank or set MPI_PROC_NULL if its at the left boundary
  if (my_rank > 0) {
    left_neigh_rank = my_rank - 1;
  }else {
    left_neigh_rank = MPI_PROC_NULL;
  }

  int right_neigh_rank;    //determine right neighbor rank or set MPI_PROC_NULL if its at the right boundary
  if (my_rank < comm_size - 1) {
    right_neigh_rank = my_rank + 1;
  }else {
    right_neigh_rank = MPI_PROC_NULL;    //indicates that there isnt a process near that direction
  }

  double local_err = 0.0;    //accumulates error across all time steps locally

  //simulation time steps
  for (size_t i = 0; i < ITER; i++){

    //exchange boundary points with left and right neighbor processes

    //communication towards the right (even ranks send first, odd ranks receive first)
    if (my_rank % 2 == 0) {

      //even ranks send their right boundary point to the right neighbor
      MPI_Send(&local_curr[end], 1, MPI_DOUBLE, right_neigh_rank, 0, MPI_COMM_WORLD);

      //even ranks receive the left ghost cell from left neighbor
      MPI_Recv(&local_curr[beg - 1], 1, MPI_DOUBLE, left_neigh_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    } else {

      //odd ranks receive left ghost cell from left neighbor
      MPI_Recv(&local_curr[beg - 1], 1, MPI_DOUBLE, left_neigh_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      //odd ranks send their right boundary point to the right neighbor
      MPI_Send(&local_curr[end], 1, MPI_DOUBLE, right_neigh_rank, 0, MPI_COMM_WORLD);

    }

    //communication towards the left (odd ranks send first, even ranks receive first)
    if (my_rank % 2 == 0) {

      //even ranks receive the right ghost cell from the right neighbor
      MPI_Recv(&local_curr[end + 1], 1, MPI_DOUBLE,
               right_neigh_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      //even ranks send their left boundary point to the left neighbor
      MPI_Send(&local_curr[beg], 1, MPI_DOUBLE,
               left_neigh_rank, 1, MPI_COMM_WORLD);

    } else {

      //odd ranks send their left boundary point to the left neighbor
      MPI_Send(&local_curr[beg], 1, MPI_DOUBLE,
               left_neigh_rank, 1, MPI_COMM_WORLD);

      //odd ranks receive the right ghost cell from the right neighbor
      MPI_Recv(&local_curr[end + 1], 1, MPI_DOUBLE,
               right_neigh_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    // Set boundary conditions to 0
    if (my_rank == 0) {

      aux[beg - 1] = 0.0;   //fix left boundary in aux
      local_curr[beg - 1] = 0.0;   //fix left halo cell in local_curr
    }

    if (my_rank == comm_size - 1) {

      aux[end + 1] = 0.0;    //fix right boundary in aux
      local_curr[end + 1] = 0.0;   //fix right halo cell in local_curr
    }


    // Compute temperatures at new time step
    for (size_t j = beg; j <= end; j++){
      //changed from heat_array to local_curr so that we can use current and neighbor values
      aux[j] = local_curr[j] + dt/(dx*dx)*(local_curr[j-1] - 2.0*local_curr[j] + local_curr[j+1]);

      //accumulate error contribution for THIS step on THIS rank (thats why i didnt use *err)
      local_err += dx*fabs(aux[j] - local_curr[j]);
    }

    // update local temperature values
    for (size_t j = beg; j <= end; j++){
      local_curr[j] = aux[j];
    }
  }

  free(aux);

  double total_err = 0.0;    //total cumulative error across all ranks

  //reduce and sum local_err from all ranks into total_err on rank 0
  MPI_Reduce(&local_err, &total_err, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  //we need to deposit final error and set boundary values in heat array on rank 0
  if (my_rank == 0) {

    *err = total_err;     //set global error
    heat_array[0] = 0.0;  //set global left boundary
    heat_array[num_pts + 1] = 0.0;  //set global right boundary
  }

  int *recvcounts = NULL;    //stores counts received from each rank for MPI_Gatherv

  int *displs = NULL;    //stores array displacementes in heat_array for MPI_Gatherv

  //metadata arrays are allocated and initialized only on rank 0
  if (my_rank == 0) {

    recvcounts = (int *) malloc((size_t)comm_size * sizeof(int));   //allocate memory for recvcounts array

    displs = (int *) malloc((size_t)comm_size * sizeof(int));       //allocate memory for displs array

    //verify if metadata array allocations succeded
    if (recvcounts == NULL || displs == NULL) {

      free(local_curr);
      free(recvcounts);
      free(displs);

      return -1;
    }

    //precompute chunk sizes and displacements for all ranks
    for (int p = 0; p < comm_size; p++) {

      //calc numbers of elements for rank p
      size_t p_local_num = base_num + ((size_t)p < rem_pts ? 1 : 0);

      //calc start index for rank p
      size_t p_start_indx = 1 + (size_t)p * base_num + ((size_t)p < rem_pts ? (size_t)p : rem_pts);

      //store element count for rank p
      recvcounts[p] = (int)p_local_num;

      //store displacement index for rank p
      displs[p] = (int)p_start_indx;
    }
  }

  //gather all local segments from rank buffers into heat_array on rank 0
  MPI_Gatherv(&local_curr[beg], (int)local_num, MPI_DOUBLE, heat_array, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  free(local_curr);    //free local buffer

  //free metadata arrays on rank 0
  if (my_rank == 0) {
    free(recvcounts);
    free(displs);
  }


  return 0;
}
