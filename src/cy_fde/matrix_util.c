#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include "matrix_util.h"

/*
* A : M x K
* B : K x N
* C : M x N
*/
void my_zgemm(const int M, const int N, const int K, const double complex alpha, const double complex  *A, const  double complex *B, const double complex beta, double complex * Out){
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++, Out++) {
      *Out *= beta;
      complex double *A_ptr = A + K*m;
      complex double *B_ptr = B + n;
      complex double sum = 0;
      for (int  k = 0; k < K; k++, A_ptr++, B_ptr+= N) {
        sum+= *A_ptr * ( *B_ptr);
        /*
        printf("m=%d, n=%d\n",m,n);
        printf("A;%f\n", abs(*A_ptr - A[K*m + k] ));
        printf("B;%f\n", abs(*B_ptr - B[k*N + n] ));
        */
      }
      *Out += alpha*sum;
    }
  }
}
