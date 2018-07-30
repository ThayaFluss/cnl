#ifndef TEST_MATRIX_UTIL
#define TEST_MATRIX_UTIL
#include "matrix_util.h"
//#include <cblas.h>
//int test_blas(int result);

int test_my_zgemm(int result);

int test_inv2by2(int result);

int test_inv2by2inv2by2_overwrite(int result);
int test_outer(int result);


int test_my_zaxpy(int result);

int test_my_zdot(int result);

#endif
