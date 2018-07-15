#include "spn.h"

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
  i_mat = malloc( sizeof(double complex)*4);
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
  o_DGe[1] = pow(sigma,2)*(((double)p )/d + 0*I);
  o_DGe[2] = 0;
  o_DGe[3] = pow(sigma,2);
  my_zdot(4, G, o_DGe);
  my_zdot(4, G, o_DGe);
}


void DG(const complex double *G,  const  complex double *TG_Ge,  complex double *o_DG){
  // o_DG = eye -  TG_Ge
  o_DG[0] = 1 - TG_Ge[0];
  o_DG[1] = 0 - TG_Ge[1];
  o_DG[2] = 0 - TG_Ge[2];
  o_DG[3] = 1 - TG_Ge[3];
  // S = ( eye- TG_Ge)^{-1}
  inv2by2_overwrite(o_DG);
  // o_DG = Tz_Ge @ S
  // where Tz_Ge =  - G[0]**2, 0,
  //                  0, -G[1]**2
  o_DG[0] *= -pow(G[0],2);
  o_DG[1] *= -pow(G[0],2);
  o_DG[2] *= -pow(G[1],2);
  o_DG[3] *= -pow(G[1],2);

}

void T_eta(const int p, const int d, complex double *o_T_eta){
  o_T_eta[0] = 0;
  o_T_eta[1] = 1;
  o_T_eta[2] = (double)p/d;
  o_T_eta[3] = 0;
}

void Dh(const double complex* DG, const double complex *T_eta, const double sigma,double complex *o_Dh){
  my_zgemm(2,2,2,1.0, DG, T_eta, 1.0, o_Dh);
  int temp = -pow(sigma,2);
  my_zax(4, temp, o_Dh);
}

void Psigma_G(const int p,const int d, const double sigma, const complex double *G, const complex double *DGe, complex double *o_Psigma_G){
  complex double *Psigma_Ge;
  Psigma_Ge =malloc(sizeof(complex double)*2);
  Psigma_Ge[0] = 2*sigma*pow(G[0],2)*(G[1]* (double)p/d);
  Psigma_Ge[1] = 2*sigma*pow(G[1],2)*(G[0]);


  complex double *S;
  S =malloc( sizeof(complex double)*4);
  S[0] = 1 - DGe[0];
  S[1] = 0 - DGe[1];
  S[2] = 0 - DGe[2];
  S[3] = 1 - DGe[3];

  inv2by2_overwrite(S);
  my_zgemm(1,2,2, 1.0, Psigma_Ge,  S,1.0, o_Psigma_G);


}

// o_Psigma_h ; 1 x 2
void Psigma_h(const int p, const int d, const double sigma, const complex double * G, const complex double* P_sigma_G, const double complex *T_eta,\
complex double* o_Psigma_h){
    // -2*sigma*eta_2by2(G)
    o_Psigma_h[0] = -2*sigma*(G[1]* (double)p/d);
    o_Psigma_h[1] = -2*sigma*(G[0]);
    // 1 x 2 mat @ 2 x 2 mat
    my_zgemm(1,2,2, -sigma*sigma, P_sigma_G, T_eta, 1.0, o_Psigma_h);
}




//////////////////////////////
// Total and Partial Derivations of Descrete
///////////////////////////////
// total derivation
//@a: d
//@o_DG; 2 x 2
//@W: 2
void des_DG( int p, int d, const double *a, const complex double *W,complex double*o_DG){
  complex double sum1 = 0;
  complex double sum2 = 0;
  for (int i = 0; i < d; i++, a++) {
    complex double inv;
    inv = 1./pow(W[1]*W[0] - *a* (*a), 2);
    sum1 -= inv;
    sum2 -= inv*pow(*a,2);
  }
  o_DG[0] = (1./d)*pow(W[1],2)*sum1;
  o_DG[1] = (1./p)*sum2;
  o_DG[2] = (1./d)*sum2;
  o_DG[3] = (1./p)*( pow(W[0],2)*sum1 - (double)(p-d)/ (pow(W[1],2) ) ) ;
}


//
// 2 x 2
//
//
void des_Dh( const complex double *DG, const complex double *F,complex double*o_Dh){
  for (int m = 0; m < 2; m++) {
    for (int n = 0; n < 2; n++, o_Dh++, DG++) {
      *o_Dh  = - pow(F[n],2)* (*DG);
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

void des_Pa_h( int p, int d, const double *a, const complex double *W, complex double *F, complex double *o_Pa_h){
  for (int m = 0; m < d; m++, a++, o_Pa_h++) {
    complex double inv = 1./ pow( W[0]*W[1]- (*a)*(*a),2);
    complex double temp = 2*(*a)*inv;
    // o_Pa_h[m][0]
    *o_Pa_h = - temp*pow(F[0],2)*W[1]/d;
    // o_Pa_h[m][1]
    o_Pa_h += 1;
    *o_Pa_h = - temp*pow(F[1],2)*W[0]/p;
  }
}
