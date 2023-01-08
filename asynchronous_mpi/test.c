#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include "test.h"
#include "knn.h"

test_input test_0(int pi, int p) {
    test_input t;

    int n = 6;
    int d = 2;
    int k = 2;

    t.d = d;
    t.k = k;

    double* X = (double*) malloc(n * d * sizeof(double));
    if(X == NULL) {
        printf("Error allocating X array\n");
        exit(1);
    }
    double temp[12] = {
        1.0, 1.0,
        4.0, 5.0,
        2.0, 2.0,
        3.0, 1.0,
        4.0, 6.0,
        2.0, 1.5,
    };
    for(int i = 0; i < n * d; i++) {
        X[i] = temp[i];
    }

    // split it
    if (n % p == 0) {
        t.n = n / p;
        t.X = (double*) malloc(t.n * t.d * sizeof(double));
        if(t.X == NULL) {
            printf("error allocating X array\n");
            exit(1);
        } 
        int start = t.n * t.d *  pi;
        for (int i = 0; i < t.n * t.d; i++) {
            t.X[i] = X[start + i];
        } 
    } else {
        t.n = (n / p) + 1;
        t.X = (double*) malloc(t.n * t.d * sizeof(double));
        if(t.X == NULL) {
            printf("error allocating X array\n");
            exit(1);
        } 
        if (pi == p - 1) {
            // find end, fill with points on inf (DBL_MAX)
            int start = t.n * t.d * pi;
            int end = n * t.d;
            for(int i = 0; i < (end - start); i++) {
                t.X[i] = X[start + i]; 
            } 
            for(int i = (end - start); i < t.n * t.d; i++) {
                t.X[i] = DBL_MAX;
            }
        } else {
            int start = t.n * pi;
            for (int i = 0; i < t.n * t.d; i++) {
                t.X[i] = X[start + i];
            } 
        }
    }

    return t;
}

test_input test_1(int pi, int p) {
    test_input t;

    int d = 3;
    int k = 3;
    int n = 8;

    t.d = d;
    t.k = k;

    double* X = (double*) malloc(n * d * sizeof(double));
    if(X == NULL) {
        printf("error allocating X array\n");
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
    for(int i = 0; i < n * d; i++) {
        X[i] = temp[i];
    }


    // split it
    if (n % p == 0) {
        t.n = n / p;
        t.X = (double*) malloc(t.n * t.d * sizeof(double));
        if(t.X == NULL) {
            printf("error allocating X array\n");
            exit(1);
        } 
        int start = t.n * t.d *  pi;
        for (int i = 0; i < t.n * t.d; i++) {
            t.X[i] = X[start + i];
        } 
    } else {
        t.n = (n / p) + 1;
        t.X = (double*) malloc(t.n * t.d * sizeof(double));
        if(t.X == NULL) {
            printf("error allocating X array\n");
            exit(1);
        } 
        if (pi == p - 1) {
            // find end, fill with points on inf (DBL_MAX)
            int start = t.n * t.d * pi;
            int end = n * t.d;
            for(int i = 0; i < (end - start); i++) {
                t.X[i] = X[start + i]; 
            } 
            for(int i = (end - start); i < t.n * t.d; i++) {
                t.X[i] = DBL_MAX;
            }
        } else {
            int start = t.n * pi;
            for (int i = 0; i < t.n * t.d; i++) {
                t.X[i] = X[start + i];
            } 
        }
    }

    return t;
}

void print_test(test_input test) {
    printf("Test\n");
    printf("n = %d, d = %d, k = %d\n", test.n, test.d, test.k);
    printf("Points in X are: ");
    for(int i = 0; i < test.n; i++) {
        printf("[%d](", i);
        for(int j = 0; j < test.d; j++) {
            printf("%f", test.X[i * test.d + j]);
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

test_input test_mnist(int pi, int p) {
    test_input t;

    char* filename = "./datasets/MNIST_txt/MNIST_train.txt";

    int d = 784;
    // lets say 10 nearest neighboors
    int k = 10;
    int n = 10000;

    t.d = d;
    t.k = k;

    double* X = (double*) malloc(n * d * sizeof(double));
    if(X == NULL) {
        printf("error allocating X array\n");
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
            } 
        }
    }

    // split it
    if (n % p == 0) {
        t.n = n / p;
        t.X = (double*) malloc(t.n * t.d * sizeof(double));
        if(t.X == NULL) {
            printf("error allocating X array\n");
            exit(1);
        } 
        int start = t.n * t.d *  pi;
        for (int i = 0; i < t.n * t.d; i++) {
            t.X[i] = X[start + i];
        } 
    } else {
        t.n = (n / p) + 1;
        t.X = (double*) malloc(t.n * t.d * sizeof(double));
        if(t.X == NULL) {
            printf("error allocating X array\n");
            exit(1);
        } 
        if (pi == p - 1) {
            // find end, fill with points on inf (DBL_MAX)
            int start = t.n * t.d * pi;
            int end = n * t.d;
            for(int i = 0; i < (end - start); i++) {
                t.X[i] = X[start + i]; 
            } 
            for(int i = (end - start); i < t.n * t.d; i++) {
                t.X[i] = DBL_MAX;
            }
        } else {
            int start = t.n * pi;
            for (int i = 0; i < t.n * t.d; i++) {
                t.X[i] = X[start + i];
            } 
        }
    }

    return t;
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
}
