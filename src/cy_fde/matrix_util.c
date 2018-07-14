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
    for (int n = 0; n < N; n++, Out++){
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


void my_zax(const int dim, const double complex alpha, double complex * o_vec){
for (int i = 0; i < dim; i++, o_vec++) {
  *o_vec *= alpha;
  }
}



void inv_ow(const int dim,  double complex *o_vec){
  for (int i = 0; i < dim; i++, o_vec++) {
    *o_vec = 1./ *o_vec;
    }
}



void inv2by2(const double complex *A, double complex *o_inv){
  double complex  det = 0;
  det = A[3]*A[0] - A[1]*A[2];
  o_inv[0] = A[3]/det;
  o_inv[1] = - A[1]/det;
  o_inv[2] = - A[2]/det;
  o_inv[3] = A[0]/det;
}



void inv2by2_overwrite(double complex *o_A){
  double complex  det = 0;
  det = o_A[3]*o_A[0] - o_A[1]*o_A[2];
  double complex temp = o_A[0]/det;
  o_A[0] = o_A[3]/det;
  o_A[1] *= - 1./det;
  o_A[2] *= - 1./det;
  o_A[3] *= temp;
}


void outer(const int dim, const double complex * v , const double complex  *w , double complex *o_mat){
  for (int m = 0; m < dim; m++, v++) {
    double complex *w_ptr = w;
    for (int n = 0; n < dim; n++, w_ptr++, o_mat++) {
      *o_mat = *v * conj(*w);
    }
  }
}



void my_zaxpy(const int dim, const double complex alpha, const double complex * v, double complex *o_vec){
  for (int i = 0; i < dim; i++, v++, o_vec++) {
    *o_vec +=  alpha*(*v);
  }
}

void my_zdot(const int dim, const double complex *v , double complex* o_vec){
  for (int i = 0; i < dim; i++, v++, o_vec++) {
    *o_vec *=  *v;
  }
}
