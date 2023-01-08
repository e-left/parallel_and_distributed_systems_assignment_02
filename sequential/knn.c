#include "knn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cblas.h>

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