#ifndef SPN
#define SPN
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include "matrix_util.h"
//#include <cblas.h>
int cauchy_sc(double complex*, double complex*, int, double, double, int, int, long*);

int cauchy_spn(double complex* B, double complex* o_omega,double complex* o_G_sc,int max_iter,double thres, \
double sigma, int p_dim, int dim, long* o_forward_iter,double* a,  double complex* o_omega_sc);

void grad_cauchy(int d, int p, const complex z,  const complex double *a, const complex double sigma, \
  const complex double *G, const complex double *omega, const complex double *omega_sc,\
    complex double *o_grad_a, complex double *o_grad_sigma);



// transpose of derivation of Ge
// G : 2
// o_DGe: 2 x 2
void DGe( const int p, const int d, const double sigma, const complex double *G, complex double *o_DGe);


// transpose of derivation of cauchy_sc
// G: 2
// DG: 2 x 2
void DG( const int p, const int d, const double sigma, const complex double *G, complex double *o_DG);

void T_eta(const int p, const int d, complex double *o_T_eta);

void Dh(const double complex* DG, const double complex *T_eta, const double sigma,double complex *o_Dh);


void des_DG( int p, int d, const double *a, const complex double *W,complex double*o_DG);

#endif
