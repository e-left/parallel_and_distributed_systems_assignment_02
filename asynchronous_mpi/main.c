#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "knn.h"
#include "test.h"
#include <sys/time.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // -------------------------------------

    // load test
    test_input test = test_mnist(world_rank, world_size);

    printf("[%d of %d] printing test\n", world_rank + 1, world_size);
    print_test(test);

    MPI_Barrier(MPI_COMM_WORLD);

    // run test
    // note that the function has a barrier in the end
    printf("[%d of %d] running\n", world_rank + 1, world_size);
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    long long start = te.tv_sec*1000LL + te.tv_usec/1000; // calculate milliseconds
    knnresult result = distrAllkNN(test.X, test.n, test.d, test.k);
    gettimeofday(&te, NULL); // get current time
    long long end = te.tv_sec*1000LL + te.tv_usec/1000; // calculate milliseconds

    // observe results in the end, in order
    int temp = 5;
    if(world_rank != 0) {
        MPI_Recv(&temp, 1, MPI_INT, (world_rank - 1) % world_size, 69, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    printf("[%d of %d] printing result\n", world_rank + 1, world_size);
    print_result(result);
    MPI_Send(&temp, 1, MPI_INT, (world_rank + 1) % world_size, 69, MPI_COMM_WORLD);

    // -------------------------------------

    // free result
    free_knnresult(result);
    // free test
    free_test(test);

    MPI_Barrier(MPI_COMM_WORLD);

    printf("[%d of %d] Test took %lld miliseconds to complete \n", world_rank + 1, world_size, end - start);

    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}
