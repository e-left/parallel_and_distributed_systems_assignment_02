#ifndef TEST_H
#define TEST_H

#include "knn.h"

typedef struct {
    double* X;
    double* Y;
    int n;
    int m;
    int d;
    int k;
} test_input;

test_input test_0();
test_input test_1();
test_input test_mnist();
test_input test_2d_grid();
test_input test_3d_grid();
test_input test_4d_grid();
test_input test_5d_grid();

void print_test(test_input test);

void print_result(knnresult result);

void free_test(test_input test);

#endif