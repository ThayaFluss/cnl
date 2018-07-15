#ifndef MATRIX_UTIL
#define MATRIX_UTIL
#include <stdlib.h>
#include <string.h>
#include <complex.h>

// matrix product
void my_zgemm(const int M, const int N, const int K,\
   const double complex alpha, const double complex  *A, \
   const  double complex *B, \
   const complex double beta, double complex * Out);

// inverse of 2-dim matrix
void inv2by2(const double complex *A, double complex *o_inv);

// inverse of 2-dim matrix
// overwrite
void inv2by2_overwrite(double complex *o_A);

// outer product of two vectors  =  v w^*
void outer(const int dim, const double complex * v , const double complex  *w , double complex *o_mat);


// sum of vectors
// overwrite second vector
// o_vec = alpha *x + o_vec
void my_zaxpy(const int dim, const double complex alpha, const double complex * v,  double complex *o_vec);



// sclar x  vector
// overwrite
void my_zax(const int dim, const double complex alpha, double complex * o_vec);


// entriewise product
// overwrite second vector
void my_zdot(const int dim, const double complex *v , double complex* o_vec);

// entriewise inverse
// overwrite
void inv_ow(const int dim,  double complex *o_vec);

#endif
