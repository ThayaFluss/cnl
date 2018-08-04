#ifndef SPN_H
#define SPN_H
#include "init_cy_fde.h"
#include "matrix_util.h"
//#include <cblas.h>



typedef struct SemiCircularNet{
  int p;
  int d;
  DCOMPLEX *Pa_h_A;         // d
  DCOMPLEX *Pa_omega;       // 2*d


  DCOMPLEX F_A[2];         //
  DCOMPLEX h_A[2];         //

  DCOMPLEX TG_Ge_sc[4] ;   //
  DCOMPLEX DG_sc[4];       //

  DCOMPLEX temp_T_eta[4];  //
  DCOMPLEX Dh_sc[4];       //


  DCOMPLEX Psigma_G_sc[2]; //

  DCOMPLEX Psigma_h_sc[2] ;//


  DCOMPLEX DG_A[4];        //
  DCOMPLEX Dh_A[4];        //

  DCOMPLEX S[4];           //

  DCOMPLEX temp_mat[4];    //

  DCOMPLEX Psigma_omega[2];//


}SCN;

void
SCN_construct(SCN* self, int p , int d);

void
SCN_init(SCN* self);

void
SCN_init_forward(SCN* self);
void
SCN_init_backward(SCN* self);


void
SCN_destroy(SCN* self);


int
SCN_cauchy(SCN* self);

void
SCN_grad(SCN* self,  int p, int d, double  *a,  double  sigma, \
  DCOMPLEX z,DCOMPLEX *G, DCOMPLEX *omega, DCOMPLEX *omega_sc,\
  DCOMPLEX *o_grad_a, DCOMPLEX *o_grad_sigma);



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


/*
Only for debug
*/
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
  int  batch_size, double *batch, \
  double *o_grad_a, double *o_grad_sigma, double *o_loss);


#endif
