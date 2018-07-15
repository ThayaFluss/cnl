#include <stdlib.h>
#include <string.h>
#include "spn.h"
#include <complex.h>
#include <math.h>
//#include <cblas.h>
#include "matrix_util.h"

int cauchy_sc(double complex* Z, double complex*  o_G, int max_iter, double thres, double sigma, int p_dim, int dim, long* o_forward_iter){
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


int cauchy_spn(double complex* B, double complex* o_omega,double complex* o_G_sc,int max_iter,double thres, \
double sigma, int p_dim, int dim, long* o_forward_iter,double* a,  double complex* o_omega_sc){
    int flag = 0;
    int result = 0;
    for (int n = 0; n< max_iter; ++n){
        result |= cauchy_sc(o_omega, o_G_sc, max_iter, thres, sigma,p_dim,dim, o_forward_iter);
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
    result |= cauchy_sc(o_omega, o_G_sc, max_iter, thres, sigma,p_dim,dim, o_forward_iter);
    o_omega_sc[0] = 1./o_G_sc[0] - o_omega[0] + B[0];
    o_omega_sc[1] = 1./o_G_sc[1] - o_omega[1] + B[1];
    return result;
}


void grad_cauchy_spn(int d, int p, const complex z,  const complex double *a, const complex double sigma, \
  const complex double *G, const complex double *omega, const complex double *omega_sc,\
    complex double *o_grad_a, complex double *o_grad_sigma){
  double complex* i_mat;
  malloc( sizeof(double complex)*4);
  i_mat[0] = z;
  i_mat[1] = 0;
  i_mat[2] = 0;
  i_mat[3] = z;

  //P_sigma_omega = omega_sc

  //my_zgemm( P_sigma_omega,TG_Sc, P_sigma_G);

}



void TG_Ge( const int p, const int d, const double sigma, const complex double *G, complex double *o_DGe){
  // init diagonal matrix
  o_DGe[0] = 0;
  o_DGe[1] = sigma*sigma*(((double)p )/d + 0*I);
  o_DGe[2] = 0;
  o_DGe[3] = sigma*sigma;
  my_zdot(4, G, o_DGe);
  my_zdot(4, G, o_DGe);
}


void DG( const int p, const int d, const double sigma, const complex double *G, complex double *o_DG){
  TG_Ge( p,d, sigma , G, o_DG);
  // o_DG = eye -  TG_Ge
  o_DG[0] = 1 - o_DG[0];
  o_DG[1] = 0 - o_DG[1];
  o_DG[2] = 0 - o_DG[2];
  o_DG[3] = 1 - o_DG[3];
  // S = ( eye- TG_Ge)^{-1}
  inv2by2_overwrite(o_DG);
  // o_DG = Tz_Ge @ S
  // where Tz_Ge =  - G[0]**2, 0,
  //                  0, -G[1]**2
  o_DG[0] *= -G[0]*G[0];
  o_DG[1] *= -G[0]*G[0];
  o_DG[2] *= -G[1]*G[1];
  o_DG[3] *= -G[1]*G[1];

}

void T_eta(const int p, const int d, complex double *o_T_eta){
  o_T_eta[0] = 0;
  o_T_eta[1] = 1;
  o_T_eta[2] = (double)p/d;
  o_T_eta[3] = 0;
}

void Dh(const double complex* DG, const double complex *T_eta, const double sigma,double complex *o_Dh){
  my_zgemm(2,2,2,1, DG, T_eta, 1, o_Dh);
  int temp = -sigma*sigma;
  my_zax(4, temp, o_Dh);
}



/*
Derivations of Descrete
a: d
o_DG; 2 x 2
W: 2
*/
void des_DG( int p, int d, const double *a, const complex double *W,complex double*o_DG){
  complex double sum1 = 0;
  complex double sum2 = 0;
  for (int i = 0; i < d; i++, a++) {
    complex double inv;
    inv = 1./(W[1]*W[0] - *a* (*a));
    sum1 -= inv*inv;
    sum2 -= inv*inv*(*a)*(*a);
  }
  o_DG[0] = (1./d)*W[1]*W[1]*sum1;
  o_DG[1] = (1./p)*sum2;
  o_DG[2] = (1./d)*sum2;
  o_DG[3] = (1./p)*( W[0]*W[0]*sum1 - (double)(p-d)/W[1]**2 ) ;
}


//
// 2 x 2
//
//
void des_Dh( const complex double *DG, const complex double *F,complex double*o_Dh){
  for (int m = 0; m < 2; m++) {
    for (int n = 0; n < 2; n++, o_Dh++, tpTG++) {
      *o_Dh  = - F[n]* (*tpTG)*F[n];
      if (m == n ){
      *o_Dh -= 1.;
      }
    }
  }
}

/*
partial differencial of Descrete
a: d
o_DG; 2 x 2
W: 2
F: 2
Pa_h : d x 2
*/

void des_Pa_h( int p, int d, const double *a, const complex double *W, complex double *F, complex double *Pa_h){
  for (int m = 0; m < d; m++, a++, out++) {
    complex double inv = 1./ ( W[0]*W[1]- (*a)*(*a) );
    complex double temp = 2*(*a)*inv*inv;
    // out[m][0]
    *out = - temp*F[0]*F[0]*W[1]/d;
    // out[m][1]
    out += 1;
    *out = - temp*F[1]*F[1]*W[0]/p;
  }
}
