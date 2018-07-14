#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <assert.h>
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


int test_inv2by2(int result){
    complex double * A, *inv, *temp;
    A = malloc(sizeof(complex double) *4);
    inv = malloc(sizeof(complex double) *4);
    temp = malloc(sizeof(complex double) *4);

    A[0] = 1;
    A[1] = 2;
    A[2] = 3;
    A[3] = 4;
    inv2by2(A, inv);
    my_zgemm(2,2,2, 1.0, A, inv, 1.0, temp);
    assert(temp[0] == 1);
    assert(temp[1] == 0);
    assert(temp[2] == 0);
    assert(temp[3] == 1);

    return result;
}


int test_outer(int result){
  int dim = 2;
  complex double *v, *w, *mat;
  v = malloc(sizeof(complex double) *dim);
  w = malloc(sizeof(complex double) *dim);
  mat = malloc(sizeof(complex double) *dim*dim);
  v[0] = 1;
  v[1] = 0;
  w[0] = 0;
  w[1] = 1 + 2*I;
  outer(dim, v,w, mat);
  assert(mat[0] == 0);
  assert(mat[1] == 1 - 2*I);
  assert(mat[2] == 0);
  assert(mat[3] == 0);
  return result;

}


int test_my_zaxpy(int result){
  int dim = 2;
  complex double alpha = 3;
  complex double *v, *w;
  v = malloc(sizeof(complex double) *dim);
  w = malloc(sizeof(complex double) *dim);
  v[0] = 1;
  v[1] = 0;
  w[0] = 0;
  w[1] = 1 + 2*I;
  my_zaxpy(dim, alpha, v,w);
  assert ( w[0] == 3);
  assert ( w[1] == 1 + 2*I);

  return result;
}



int test_my_zdot(int result){
  int dim = 2;
  complex double *v, *w;
  v = malloc(sizeof(complex double) *dim);
  w = malloc(sizeof(complex double) *dim);
  v[0] = 1;
  v[1] = 2;
  w[0] = 0;
  w[1] = 1 + 2*I;
  my_zdot(dim, v, w);
  assert ( w[0] == 0);
  assert ( w[1] == 2 + 4*I);
  return result;
}
