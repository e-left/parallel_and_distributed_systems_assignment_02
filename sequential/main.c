#include "knn.h"
#include "test.h"
#include <stdio.h>
#include <sys/time.h>

int main(int argc, char **argv) {
    // test setup
    test_input test = test_5d_grid();

    // run test
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    long long start = te.tv_sec*1000LL + te.tv_usec/1000; // calculate milliseconds
    knnresult result = kNN(test.X, test.Y, test.n, test.m, test.d, test.k);
    gettimeofday(&te, NULL); // get current time
    long long end = te.tv_sec*1000LL + te.tv_usec/1000; // calculate milliseconds

    // observe results
    print_test(test);
    print_result(result);

    // free test
    free_test(test);
    // free result
    free_knnresult(result);

    // log time
    printf("Test took %lld miliseconds to complete\n", end - start);

    return 0;
}