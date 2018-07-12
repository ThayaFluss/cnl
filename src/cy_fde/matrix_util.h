#ifndef MATRIX_UTIL
#define MATRIX_UTIL
#include <stdlib.h>
#include <string.h>
#include <complex.h>


void my_zgemm(const int M, const int N, const int K, const double complex alpha, const double complex  *A, const  double complex *B, const complex double beta, double complex * Out);

#endif
