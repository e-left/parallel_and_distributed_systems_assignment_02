#include "knn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cblas.h>
#include <mpi.h>

void free_knnresult(knnresult k) {
    free(k.nidx);
    free(k.ndist);
}

knnresult kNN(double *X, double *Y, int n, int m, int d, int k) {
    // create ndist and nidx for the final knnresult
    int *nidx = (int *)malloc(m * k * sizeof(int));
    if (nidx == NULL) {
        printf("Error while creating index array\n");
        exit(1);
    }
    double *ndist = (double *)malloc(m * k * sizeof(double));
    if (ndist == NULL) {
        printf("Error while creating distances array\n");
        exit(1);
    }

    for(int i = 0; i < m * k; i++) {
        nidx[i] = -1;
        ndist[i] = DBL_MAX;
    }

    // create matrix D
    // A = (X . X) * e * e'
    // B = -2 * X * Y'
    // C = e * e' * (Y . Y)'
    // D = A + B + C
    double* A = (double*) malloc(m * n * sizeof(double));
    if (A == NULL) {
        printf("Error while creating matrix A\n");
        exit(1);
    }

    for(int i = 0; i < m; i++) {
        double sum = 0;
        for (int j = 0; j < d; j++) {
            // row i, column j
            sum += X[i * d + j] * X[i * d + j];
        }

        for(int j = 0; j < n; j++) {
            // row i, column j
            A[i * n + j] = sum;
        }
    }

    double* B = (double*) malloc(m * n * sizeof(double));
    if (B == NULL) {
        printf("Error while creating matrix B\n");
        exit(1);
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, d, -2, X, d, Y, d, 0, B, n);

    double* C = (double*) malloc(m * n * sizeof(double));
    if (C == NULL) {
        printf("Error while creating matrix C\n");
        exit(1);
    }

    for(int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < d; j++) {
            // row i, column j
            sum += Y[i * d + j] * Y[i * d + j];
        }

        for(int j = 0; j < m; j++) {
            // row j, column i
            C[j * n + i] = sum;
        }
    }

    double* D = (double*) malloc(m * n * sizeof(double));
    if (D == NULL) {
        printf("Error while creating matrix D\n");
        exit(1);
    }

    for(int i = 0; i < m * n; i++) {
        D[i] = sqrt(A[i] + B[i] + C[i]);
    }

    // find the k smallest distances and their indices for each point in X
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            // find largest in distances array
            double largest_distance = 0.0;
            int largest_index = -1;
            for(int l = 0; l < k; l++) {
                // row i, column l
                if(ndist[i * k + l] > largest_distance) {
                    largest_distance = ndist[i * k + l];
                    largest_index = l;
                }
            }
            // if the distance we check is smaller than the current largest, replace it
            // row i, column j
            if(D[i * n + j] < largest_distance) {
                ndist[i * k + largest_index] = D[i * n + j];
                nidx[i * k + largest_index] = j;
            }
        }
    }

    // free matrices
    free(A);
    free(B);
    free(C);
    free(D);

    // construct the final object and return it
    knnresult result;
    result.nidx = nidx;
    result.ndist = ndist;
    result.m = m;
    result.k = k;

    return result;
}

knnresult distrAllkNN(double *X, int n, int d, int k) {
    // initializations
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int offset = world_rank * n;

    knnresult result;
    result.k = k;
    result.m = n;
    result.nidx = (int *)malloc(n * k * sizeof(int));
    if (result.nidx == NULL) {
        printf("Error while creating index array\n");
        exit(1);
    }
    for(int i = 0; i < n * k; i++) {
        result.nidx[i] = -1;
    }
    result.ndist = (double *)malloc(n * k * sizeof(double));
    if (result.ndist == NULL) {
        printf("Error while creating distances array\n");
        exit(1);
    }
    for(int i = 0; i < n * k; i++) {
        result.ndist[i] = DBL_MAX;
    }

    // initialize Y with X for the first round
    double* Y = X;

    double* Z = (double *)malloc(n * d * sizeof(double));
    if (Z == NULL) {
        printf("Error while creating incoming distances array\n");
        exit(1);
    }

    // perform the algorithm
    printf("[%d of %d] beginning computations\n", world_rank + 1, world_size);
    for(int i = 0; i < world_size; i++) {
        printf("[%d of %d] round %d of %d\n", world_rank + 1, world_size, i + 1, world_size);
        // funny tag haha
        // send current points to next process
        MPI_Request send_req;
        MPI_Isend(Y, n * d, MPI_DOUBLE, (world_rank + 1) % world_size, 69, MPI_COMM_WORLD, &send_req);

        // receive points from previous process
        MPI_Request recv_req;
        MPI_Irecv(Z, n * d, MPI_DOUBLE, (world_rank - 1) % world_size, 69, MPI_COMM_WORLD, &recv_req);

        // obtain results for current data piece
        knnresult res_iteration = kNN(X, Y, n, n, d, k);
        // keep only k better from k (previous) and k(current)
        for(int j = 0; j < n; j++) {
            for(int l = 0; l < k; l++) {
                // find largest value on previous dataset
                int largest_prev_index = -1;
                double largest_prev_distance = 0.0;
                for(int m = 0; m < k; m++) {
                    // row j (point j), column m (neigboor m)
                    if (result.ndist[j * k + m] > largest_prev_distance) {
                        largest_prev_distance = result.ndist[j * k + m];
                        largest_prev_index = m;
                    }
                } 

                // if currently checked value is smaller than the largest from previously, replace it
                if (res_iteration.ndist[j * k + l] <= largest_prev_distance) {
                    // replace ndist and nidx with new, smaller value
                    result.ndist[j * k + largest_prev_index] = res_iteration.ndist[j * k + l];
                    result.nidx[j * k + largest_prev_index] = offset + res_iteration.nidx[j * k + l];
                }
            }
        }
        free_knnresult(res_iteration);

        // ensure send and receive have finished
        MPI_Status send_status;
        MPI_Status recv_status;

        MPI_Wait(&send_req, &send_status);
        MPI_Wait(&recv_req, &recv_status);

        // free old Y if not first iteration (i == 0: Y = X)
        // replace Y with new Z
        if(i != 0) {
            free(Y);
        }
        Y = Z;

        // reallocate Z
        Z = (double *)malloc(n * d * sizeof(double));
        if (Z == NULL) {
            printf("Error while creating incoming distances array\n");
            exit(1);
        }

        offset = (offset + n) % (n * world_size);
    }

    // free everything 
    printf("[%d of %d] cleaning up\n", world_rank + 1, world_size);
    free(Y);
    free(Z);

    // wait for everyone to finish, then return
    MPI_Barrier(MPI_COMM_WORLD);
    return result;
}