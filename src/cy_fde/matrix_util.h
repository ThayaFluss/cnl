#ifndef MATRIX_UTIL
#define MATRIX_UTIL
#include "init_cy_fde.h"


//MEM_RESULT create_dcomplex(MEM_RESULT ret, const int dim, const DCOMPLEX init_value, DCOMPLEX* o_vec);

// matrix product
void my_zgemm(const int M, const int N, const int K,\
   const DCOMPLEX alpha, const DCOMPLEX  *A, \
   const  DCOMPLEX *B, \
   const DCOMPLEX beta, DCOMPLEX * Out);

// inverse of 2-dim matrix
void inv2by2(const DCOMPLEX *A, DCOMPLEX *o_inv);

// inverse of 2-dim matrix
// overwrite
void inv2by2_overwrite(DCOMPLEX *o_A);

// outer product of two vectors  =  v w^*
void outer(const int dim, const DCOMPLEX * v , const DCOMPLEX  *w , DCOMPLEX *o_mat);


// sum of vectors
// overwrite second vector
// o_vec = alpha *x + o_vec
void my_zaxpy(const int dim, const DCOMPLEX alpha, const DCOMPLEX * v,  DCOMPLEX *o_vec);



// sclar x  vector
// overwrite
void my_zax(const int dim, const DCOMPLEX alpha, DCOMPLEX * o_vec);


// entriewise product
// overwrite second vector
void my_zdot(const int dim, const DCOMPLEX *v , DCOMPLEX* o_vec);

// entriewise inverse
// overwrite
void inv_ow(const int dim,  DCOMPLEX *o_vec);


#define isnanz( v) (isnan(creal(v)) ||  isnan(cimag(v) ))

void z_isnan(const int dim, const DCOMPLEX *v);

DCOMPLEX* create_zvec(const int dim, const DCOMPLEX* init_value);
#endif
