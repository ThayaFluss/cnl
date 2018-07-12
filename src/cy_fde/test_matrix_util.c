#include <stdlib.h>
#include <string.h>
#include <complex.h>

#include <cblas.h>

int test_blas(int result){
   int  d = 50;
   int  k = 2;
   int  l = 3;
   double *A, *B, *C;
   A = malloc(sizeof(double) *d * k);
   B = malloc(sizeof(double) *k * l);
   C = malloc(sizeof(double) *d * l);
   double alpha  = 1;
   double beta = 0;

   memset(A, 0, sizeof(double) * d * k);
   memset(B, 0, sizeof(double) * k * l);
   memset(C, 0, sizeof(double) * d * l);
   cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
       d, l, k, alpha, A, d, B, k, beta, C, d);

   return result;

}

int test_my_zgemm(int result){
    int M= 5;
    int N = 2;
    int K = 4;

    complex double *A, *B, *Out;
    A = malloc(sizeof(complex double) *M * K);
    B = malloc(sizeof(complex double) *K * N);
    Out = malloc(sizeof(complex double) *M * N);
    complex double alpha  = 1;
    complex double beta = 0;

    memset(A, 0, sizeof(complex double) * M * K);
    memset(B, 0, sizeof(complex double) * K * N);
    memset(Out, 0, sizeof(complex double) * M * N);

    for (int i = 0; i < M*K; i++) {
      A[i] = i;
    }
    for (int i = 0; i < K*N; i++) {
      B[i] = i;
    }
    my_zgemm(M,N,K,alpha, A ,B,beta, Out);

    return result;
}
