#include <stdlib.h>
#include <stdio.h>
#include "test.h"
#include "knn.h"

test_input test_0() {
    test_input t;

    t.d = 2;
    t.k = 2;
    t.m = 2;
    t.n = 6;

    t.X = (double*) malloc(t.m * t.d * sizeof(double));
    if(t.X == NULL) {
        printf("Error allocating X array\n");
        exit(1);
    }
    double tempX[4] = {
        1.0, 1.0,
        4.0, 5.0
    };
    for(int i = 0; i < t.m * t.d; i++) {
        t.X[i] = tempX[i];
    }

    t.Y = (double*) malloc(t.n * t.d * sizeof(double));
    if(t.Y == NULL) {
        printf("Error allocating Y array\n");
        exit(1);
    }
    double tempY[12] = {
        1.5, 2.5,
        1.0, 2.0,
        1.5, 3.0,
        3.0, 5.0,
        4.0, 4.0,
        4.0, 2.0
    };
    for(int i = 0; i < t.n * t.d; i++) {
        t.Y[i] = tempY[i];
    }

    return t;
}

test_input test_1() {
    test_input t;

    t.d = 3;
    t.k = 3;
    t.m = 8;
    t.n = 8;

    t.X = (double*) malloc(t.m * t.d * sizeof(double));
    if(t.X == NULL) {
        printf("Error allocating X array\n");
        exit(1);
    }
    double temp[24] = {
        0, 0, 0,
        1, 0, 0,
        1, 1, 0,
        0, 1, 0,
        0, 0, 1,
        1, 0, 1,
        1, 1, 1,
        0, 1, 1
    };
    for(int i = 0; i < t.m * t.d; i++) {
        t.X[i] = temp[i];
    }

    t.Y = (double*) malloc(t.n * t.d * sizeof(double));
    if(t.Y == NULL) {
        printf("Error allocating Y array\n");
        exit(1);
    }
    for(int i = 0; i < t.n * t.d; i++) {
        t.Y[i] = temp[i];
    }

    return t;
}

test_input test_mnist() {
    test_input t;

    char* filename = "./datasets/MNIST_txt/MNIST_train.txt";

    int d = 784;
    // lets say 10 nearest neighboors
    int k = 10;
    int n = 10000;

    t.d = d;
    t.k = k;
    t.n = n;
    t.m = n;

    double* X = (double*) malloc(n * d * sizeof(double));
    if(X == NULL) {
        printf("error allocating X array\n");
        exit(1);
    }

    double* Y = (double*) malloc(n * d * sizeof(double));
    if(Y == NULL) {
        printf("error allocating Y array\n");
        exit(1);
    }

    // here read it, all at once
    FILE *f = fopen(filename, "r");
    if (f == NULL) {
        printf("Error opening file\n");
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < (d + 1); j++) {
            int t;
            fscanf(f, "%d,", &t);
            // discard first number (label)
            if (j != 0) {
                X[i * d + j] = t * 1.0;
                Y[i * d + j] = t * 1.0;
            } 
        }
    }

    t.X = X;
    t.Y = Y;

    return t;
}

void print_test(test_input test) {
    printf("Test\n");
    printf("m = %d, n = %d, d = %d, k = %d\n", test.m, test.n, test.d, test.k);
    printf("Points in X are: ");
    for(int i = 0; i < test.m; i++) {
        printf("[%d](", i);
        for(int j = 0; j < test.d; j++) {
            printf("%f", test.X[i * test.d + j]);
            if(j != test.d - 1) {
                printf(", ");
            }
        }
        printf(")");
        if (i != test.m - 1) {
            printf(", ");
        }
    }
    printf("\n");
    printf("Points in Y are: ");
    for(int i = 0; i < test.n; i++) {
        printf("[%d](", i);
        for(int j = 0; j < test.d; j++) {
            printf("%f", test.Y[i * test.d + j]);
            if(j != test.d - 1) {
                printf(", ");
            }
        }
        printf(")");
        if (i != test.n - 1) {
            printf(", ");
        }
    }
    printf("\n");
}

void print_result(knnresult result) {
    printf("Result\n");
    printf("m = %d, k = %d\n", result.m, result.k);
    for (int i = 0; i < result.m; i++) {
        printf("For point at X[%d], the %d-nearest-neighboors are ", i, result.k);
        for (int j = 0; j < result.k; j++) {
            printf("Y[%d],", result.nidx[i * result.k + j]);
        }
        printf(" with distances");
        for (int j = 0; j < result.k; j++) {
            printf(" %f", result.ndist[i * result.k + j]);
            if (j != result.k - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

void free_test(test_input test) {
    free(test.X);
    free(test.Y);
}
