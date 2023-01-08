#ifndef TEST_H
#define TEST_H

#include "knn.h"

typedef struct {
    double* X;
    int n;
    int d;
    int k;
} test_input;

test_input test_0(int pi, int p);
test_input test_1(int pi, int p);
test_input test_mnist(int pi, int p);

void print_test(test_input test);

void print_result(knnresult result);

void free_test(test_input test);

#endif