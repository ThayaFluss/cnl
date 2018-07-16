#ifndef SPN
#define SPN
#include "init_cy_fde.h"
#include "matrix_util.h"
//#include <cblas.h>

/** Compute Cauchy transform of SemiCircular( returns total iterations)
* @param Z : input matrix
* @param o_G : out_put Cauchy transform
*
*/
int
cauchy_sc( int p,  int d,  double sigma, DCOMPLEX* Z, \
   int max_iter, double thres,\
   DCOMPLEX* o_G);

 /** Compute Cauchy transform of Signal-Plus-Noise model( returns total iterations)
 * @param Z : input matrix
 * @param o_G_sc : out_put Cauchy transform
 *
 */
int
cauchy_spn(int p_dim, int dim, double* a, double sigma,\
     DCOMPLEX* B,\
     int max_iter,double thres, \
     DCOMPLEX* o_G_sc, DCOMPLEX* o_omega, DCOMPLEX* o_omega_sc);

void
grad_cauchy_spn(int p, int d,  double  *a, double  sigma, \
  DCOMPLEX z, DCOMPLEX *G, DCOMPLEX *omega, DCOMPLEX *omega_sc,\
  DCOMPLEX *o_grad_a, DCOMPLEX *o_grad_sigma);



// transpose of derivation of Ge
// G : 2
// o_DGe: 2 x 2
void TG_Ge( const int p, const int d, const double sigma, \
  const DCOMPLEX *G, DCOMPLEX *o_DGe);


// transpose of derivation of cauchy_sc
// G: 2
// DG: 2 x 2
void DG(const DCOMPLEX *G,  const  DCOMPLEX *DGe,  DCOMPLEX *o_DG);

void T_eta(const int p, const int d, DCOMPLEX *o_T_eta);

void Dh(const DCOMPLEX* DG, const DCOMPLEX *T_eta, const double sigma,DCOMPLEX *o_Dh);

void Psigma_G(const int p,const int d, const double sigma, const DCOMPLEX *G, const DCOMPLEX *DGe, DCOMPLEX *o_Psigma_G);

void Psigma_h(const int p, const int d, const double sigma, const DCOMPLEX * G, const DCOMPLEX* P_sigma_G, const DCOMPLEX *T_eta,\
DCOMPLEX* o_Psigma_h);

//// Descrete

void des_DG( int p, int d, const double *a, const DCOMPLEX *W,DCOMPLEX*o_DG);


void des_Dh( const DCOMPLEX *DG, const DCOMPLEX *F,DCOMPLEX*o_Dh);


void des_Pa_h( int p, int d, const double *a, const DCOMPLEX *W, DCOMPLEX *F, DCOMPLEX *Pa_h);



/** compute gradient and loss of likelihood
* return total number of forward_iter
*
*/
int
grad_loss_cauchy_spn(  int p, int d, double  *a, double  sigma, double scale, \
  int  num_sample, double *sample, \
  double *o_grad_a, double *o_grad_sigma, double *o_loss);


#endif
