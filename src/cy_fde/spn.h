#ifndef SPN
#define SPN
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include "matrix_util.h"
//#include <cblas.h>
int c_cauchy_2by2(double complex*, double complex*, int, double, double, int, int, long*);

int c_cauchy_subordination(double complex* B, double complex* o_omega,double complex* o_G_sc,int max_iter,double thres, \
double sigma, int p_dim, int dim, long* o_forward_iter,double* a,  double complex* o_omega_sc);


int c_grad(void);


#endif
