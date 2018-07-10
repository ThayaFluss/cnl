#include <stdlib.h>
#include <string.h>
#include "spn.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

int c_cauchy_2by2(double complex* Z, double complex*  o_G, int max_iter, double thres, double sigma, int p_dim, int dim, long* o_forward_iter){
    int flag = 0;
    double complex sub_x = 0;
    double complex sub_y = 0;
    for(int n = 0; n < max_iter; n++){
      sub_x = 1./ (Z[0] - (pow(sigma,2)*o_G[1]*p_dim)/dim) - o_G[0];
      sub_y = 1./ (Z[1] - (pow(sigma,2)*o_G[0])) - o_G[1];
      sub_x *= 0.5;
      sub_y *= 0.5;
      if ( pow(cabs(sub_x), 2) + pow(cabs(sub_y),2) < pow(thres,2)){
        flag = 1;
      }
      o_G[0] += sub_x;
      o_G[1] += sub_y;
      if(flag == 1){
        o_forward_iter[0] +=n;
        //printf("iter = %d \n", n);
        return 1;
      }
      }
    o_forward_iter[0] +=max_iter;
    return 0;
}


int c_cauchy_subordination(double complex* B, double complex* o_omega,double complex* o_G_sc,int max_iter,double thres, \
double sigma, int p_dim, int dim, long* o_forward_iter,double* a,  double complex* o_omega_sc){
    my_zgemm();
    int flag = 0;
    int result = 0;
    for (int n = 0; n< max_iter; ++n){
        result |= c_cauchy_2by2(o_omega, o_G_sc, max_iter, thres, sigma,p_dim,dim, o_forward_iter);
        double complex W_x = 0;
        double complex W_y = 0;
        W_x  = 1./o_G_sc[0] - o_omega[0] + B[0];
        W_y =  1./o_G_sc[1] - o_omega[1] + B[1];

        double complex sum_inv_det = 0;
        for (int d = 0; d < dim; d++){
          sum_inv_det += 1./(W_x*W_y  - pow(a[d], 2) );
        }
        double complex sub_x = 0;
        double complex sub_y = 0;
        // omega_transform - old omega
        sub_x = dim / (W_y*sum_inv_det ) - W_x + B[0] - o_omega[0];
        sub_y = p_dim / (W_x*sum_inv_det + (p_dim - dim)/W_y)  - W_y  +  B[1]  - o_omega[1];
        // If subtraction is small, break the for-loop.
        if ( pow(cabs(sub_x), 2) + pow(cabs(sub_y), 2) < pow(thres, 2) ){
          flag = 1;
        }
        o_omega[0] += sub_x;
        o_omega[1] += sub_y;
        if (flag == 1){
          break;
        }
    }
    result |= c_cauchy_2by2(o_omega, o_G_sc, max_iter, thres, sigma,p_dim,dim, o_forward_iter);
    o_omega_sc[0] = 1./o_G_sc[0] - o_omega[0] + B[0];
    o_omega_sc[1] = 1./o_G_sc[1] - o_omega[1] + B[1];
    return result;
}

int test_blas(void){
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

   return 0;

}
/*
* A : M x K
* B : K x N
* C : M x N
*/
//void my_zgemm(M, N, K, const double complex  *A, const  double complex *B, double complex * Out){
void my_zgemm(void){
  int M= 5;
  int N = 2;
  int K = 4;

  double *A, *B, *Out;
  A = malloc(sizeof(double) *M * K);
  B = malloc(sizeof(double) *K * N);
  Out = malloc(sizeof(double) *M * N);
  double alpha  = 1;
  double beta = 0;

  memset(A, 0, sizeof(double) * M * K);
  memset(B, 0, sizeof(double) * K * N);
  memset(Out, 0, sizeof(double) * M * N);

  for (size_t i = 0; i < M*K; i++) {
    A[i] = i;
  }
  for (size_t i = 0; i < K*N; i++) {
    B[i] = i;
  }


  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++, Out++) {
      *Out *= beta;
      double *A_ptr = A + K*m;
      double *B_ptr = B + n;
      double sum = 0;
      for (int  k = 0; k < K; k++, A_ptr++, B_ptr+= N) {
        sum+= *A_ptr * ( *B_ptr);
        printf("m=%d, n=%d\n",m,n);
        printf("A;%f\n", *A_ptr - A[K*m + k]);
        printf("B;%f\n", *B_ptr - B[k*N + n]);
      }
      *Out += alpha*sum;
    }
  }
}
