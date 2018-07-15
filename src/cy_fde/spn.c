#include "spn.h"

int cauchy_sc(DCOMPLEX* Z, DCOMPLEX*  o_G, int max_iter, double thres, double sigma, int p_dim, int dim, long* o_forward_iter){
    int flag = 0;
    DCOMPLEX sub_x = 0;
    DCOMPLEX sub_y = 0;
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


int cauchy_spn(DCOMPLEX* B, DCOMPLEX* o_omega,DCOMPLEX* o_G_sc,int max_iter,double thres, \
double sigma, int p_dim, int dim, long* o_forward_iter,double* a,  DCOMPLEX* o_omega_sc){
    int flag = 0;
    int result = 0;
    for (int n = 0; n< max_iter; ++n){
        result |= cauchy_sc(o_omega, o_G_sc, max_iter, thres, sigma,p_dim,dim, o_forward_iter);
        DCOMPLEX W_x = 0;
        DCOMPLEX W_y = 0;
        W_x  = 1./o_G_sc[0] - o_omega[0] + B[0];
        W_y =  1./o_G_sc[1] - o_omega[1] + B[1];

        DCOMPLEX sum_inv_det = 0;
        for (int d = 0; d < dim; d++){
          sum_inv_det += 1./(W_x*W_y  - pow(a[d], 2) );
        }
        DCOMPLEX sub_x = 0;
        DCOMPLEX sub_y = 0;
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


void grad_cauchy_spn(int p, int d, const complex z,  const double  *a, const double  sigma, \
  const DCOMPLEX *G, const DCOMPLEX *omega, const DCOMPLEX *omega_sc,\
    DCOMPLEX *o_grad_a, DCOMPLEX *o_grad_sigma){

  /// init ///
  DCOMPLEX *F_A =  malloc( sizeof(DCOMPLEX)*2);
  DCOMPLEX *h_A = malloc( sizeof(DCOMPLEX)*2);

  DCOMPLEX *TG_Ge_sc = malloc( sizeof(DCOMPLEX)*4);
  DCOMPLEX *DG_sc = malloc( sizeof(DCOMPLEX)*4);

  DCOMPLEX *temp_T_eta = malloc( sizeof(DCOMPLEX)*4);

  DCOMPLEX *Dh_sc =  malloc( sizeof(DCOMPLEX)*4);


  DCOMPLEX *Psigma_G_sc = o_grad_sigma;
  //Psigma_G_sc = malloc( sizeof(DCOMPLEX)*2);

  DCOMPLEX *Psigma_h_sc= malloc( sizeof(DCOMPLEX)*2);

  DCOMPLEX *DG_A = malloc( sizeof(DCOMPLEX)*4);
  DCOMPLEX *Dh_A =malloc( sizeof(DCOMPLEX)*4);

  // O(d)
  DCOMPLEX * Pa_h_A = malloc( sizeof(DCOMPLEX)*d*2);

  DCOMPLEX * S = malloc( sizeof(DCOMPLEX)*4);

  // O(d)
  DCOMPLEX * Pa_omega = malloc( sizeof(DCOMPLEX)*d*2);
  DCOMPLEX * temp_mat = malloc( sizeof(DCOMPLEX)*4);
  DCOMPLEX * Psigma_omega = malloc( sizeof(DCOMPLEX)*2);



  /// run ///
  printf("-----(grad_cauchy_spn)------\n");

  F_A[0] = omega[0]+omega_sc[0]-z;
  F_A[1] = omega[1]+omega_sc[1]-z;

  assert ( cabs(F_A[0]*G[0] - 1) < 1e-8 );
  assert ( cabs(F_A[1]*G[1] - 1) < 1e-8 );

  h_A[0] = omega[0] -z;
  h_A[1] = omega[1] -z;

  printf("F_A[0]=%f\n", cabs(F_A[0]));
  printf("G=%f\n", cabs(G[0]));
  printf("G=%f\n", cabs(G[1]));
  TG_Ge(p,d,sigma,G, TG_Ge_sc);

  printf("abs TG_Ge_sc=%f\n", cabs(TG_Ge_sc[0]));
  printf("abs TG_Ge_sc=%f\n", cabs(TG_Ge_sc[1]));
  printf("abs TG_Ge_sc=%f\n", cabs(TG_Ge_sc[2]));
  printf("abs TG_Ge_sc=%f\n", cabs(TG_Ge_sc[3]));
  DG(G, TG_Ge_sc, DG_sc);
  printf(" DG_sc= %f \n", cabs(DG_sc[0]));
  printf(" DG_sc= %f \n", cabs(DG_sc[1]));
  printf(" DG_sc= %f \n", cabs(DG_sc[2]));
  printf(" DG_sc= %f \n", cabs(DG_sc[3]));
  T_eta(p,d,temp_T_eta);
  printf("T_eta=%f\n", cabs(temp_T_eta[0]));
  printf("T_eta=%f\n", cabs(temp_T_eta[1]));
  printf("T_eta=%f\n", cabs(temp_T_eta[2]));

  Dh(DG_sc, temp_T_eta, sigma, Dh_sc);
  printf("Dh_sc=%f\n", cabs(Dh_sc[0]));
  printf("Dh_sc=%f\n", cabs(Dh_sc[1]));
  printf("Dh_sc=%f\n", cabs(Dh_sc[2]));
  printf("Dh_sc=%f\n", cabs(Dh_sc[3]));

  Psigma_G(p,d,sigma, G, TG_Ge_sc, Psigma_G_sc);

  printf("P_sigma_G_sc=%f\n", cabs(Psigma_G_sc[0]));


  Psigma_h(p, d, sigma,G, Psigma_G_sc, temp_T_eta,\
  Psigma_h_sc);

  printf("Psigma_h=%f\n", cabs(Psigma_h_sc[0]));

  des_DG(p, d, a, omega_sc, DG_A);

  printf("abs  DG_A=%f\n", cabs(DG_A[0]));
  printf("abs  DG_A=%f\n", cabs(DG_A[1]));
  printf("abs  DG_A=%f\n", cabs(DG_A[2]));
  printf("abs  DG_A=%f\n", cabs(DG_A[3]));
  des_Dh(DG_A, F_A,  Dh_A);

  printf("abs  dha=%f\n", cabs(Dh_A[0]));
  printf("abs  dha=%f\n", cabs(Dh_A[1]));
  printf("abs  dha=%f\n", cabs(Dh_A[2]));
  printf("abs  dha=%f\n", cabs(Dh_A[3]));

  des_Pa_h(p, d,  a,  omega_sc, F_A,  Pa_h_A);
  printf("abs  Pa_h_A=%f\n", cabs(Pa_h_A[0]));
  printf("abs  Pa_h_A=%f\n", cabs(Pa_h_A[1]));
  printf("abs  Pa_h_A=%f\n", cabs(Pa_h_A[2]));
  printf("abs  Pa_h_A=%f\n", cabs(Pa_h_A[3]));
  ////tpS =  np.linalg.inv(np.eye(2,dtype=np.complex128) -  tpThsc  @ tpThA

  S[0] = 1.;
  S[1] = 0.;
  S[2] = 0.;
  S[3] = 1.;
  my_zgemm(2,2,2 , -1.0, Dh_sc, Dh_A, 1.0, S );
  ///*
  printf("pre:abs  S=%f\n", cabs(S[0]));
  printf("abs  S=%f\n", cabs(S[1]));
  printf("abs  S=%f\n", cabs(S[2]));
  printf("abs  S=%f\n", cabs(S[3]));
  //*/
  inv2by2_overwrite(S);
  ///*
  printf("inv:abs  S=%f\n", cabs(S[0]));
  printf("abs  S=%f\n", cabs(S[1]));
  printf("abs  S=%f\n", cabs(S[2]));
  printf("abs  S=%f\n", cabs(S[3]));
  //*/
  my_zgemm(d,2,2 , 1.0, Pa_h_A, S, 0, Pa_omega );
  z_isnan(d*2, Pa_omega);

  printf("abs  Pa_omega=%f\n", cabs(Pa_omega[0]));
  printf("abs  Pa_omega=%f\n", cabs(Pa_omega[1]));
  printf("abs  Pa_omega=%f\n", cabs(Pa_omega[2]));
  printf("abs  Pa_omega=%f\n", cabs(Pa_omega[3]));

  my_zgemm(d,2,2 , 1.0, Pa_omega, DG_sc, 0,  o_grad_a);

  z_isnan(d*2, o_grad_a);

  my_zgemm(2,2,2, 1.0, Dh_A, S, 0, temp_mat);
  my_zgemm(1,2,2, 1.0, Psigma_h_sc, temp_mat, 0, Psigma_omega);

  printf("abs  Psigma_omega=%f\n", cabs(Psigma_omega[0]));
  printf("abs  Psigma_omega=%f\n", cabs(Psigma_omega[1]));

  // tpPsigmaG =  tpPsigmaOmega @ tpTGsc + tpPsigmaG
  my_zgemm(1,2,2, 1.0, Psigma_omega, DG_sc, 1.0, Psigma_G_sc);

  printf("abs  Psigma_G_sc=%f\n", cabs(Psigma_G_sc[0]));
  printf("abs  Psigma_G_sc=%f\n", cabs(Psigma_G_sc[1]));

  printf("abs  o_grad_sigma=%f\n", cabs(o_grad_sigma[0]));
  printf("abs  o_grad_sigma=%f\n", cabs(o_grad_sigma[1]));

  z_isnan(2, o_grad_sigma);

}



void
TG_Ge( const int p, const int d, const double sigma, const DCOMPLEX *G, \
  DCOMPLEX *o_DGe){
  // init diagonal matrix
  o_DGe[0] = 0;
  o_DGe[1] = G[1]*G[1]*pow(sigma,2);
  o_DGe[2] = G[0]*G[0]*pow(sigma,2)*(((double)p )/d + 0*I);
  o_DGe[3] = 0;
}


void
DG(const DCOMPLEX *G,  const  DCOMPLEX *TG_Ge,  DCOMPLEX *o_DG){
  // o_DG = eye -  TG_Ge
  o_DG[0] = 1. - TG_Ge[0];
  o_DG[1] = 0. - TG_Ge[1];
  o_DG[2] = 0. - TG_Ge[2];
  o_DG[3] = 1. - TG_Ge[3];
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

void T_eta(const int p, const int d, DCOMPLEX *o_T_eta){
  o_T_eta[0] = 0;
  o_T_eta[1] = 1;
  o_T_eta[2] = (double)p/d;
  o_T_eta[3] = 0;
}

void Dh(const DCOMPLEX* DG, const DCOMPLEX *T_eta, const double sigma,DCOMPLEX *o_Dh){
  my_zgemm(2,2,2,1.0, DG, T_eta, 0, o_Dh);
  DCOMPLEX temp = -pow(sigma,2) + 0.*I;
  my_zax(4, temp, o_Dh);
}

void Psigma_G(const int p,const int d, const double sigma, const DCOMPLEX *G, const DCOMPLEX *TG_Ge, DCOMPLEX *o_Psigma_G){
  DCOMPLEX *Psigma_Ge;
  Psigma_Ge =malloc(sizeof(DCOMPLEX)*2);
  Psigma_Ge[0] = 2*sigma*G[0]*G[0]*(G[1]* (double)p/d);
  Psigma_Ge[1] = 2*sigma*G[1]*G[1]*(G[0]);


  DCOMPLEX *S;
  S =malloc( sizeof(DCOMPLEX)*4);
  S[0] = 1 - TG_Ge[0];
  S[1] = 0 - TG_Ge[1];
  S[2] = 0 - TG_Ge[2];
  S[3] = 1 - TG_Ge[3];

  inv2by2_overwrite(S);
  my_zgemm(1,2,2, 1.0, Psigma_Ge,  S,1.0, o_Psigma_G);


}

// o_Psigma_h ; 1 x 2
void Psigma_h(const int p, const int d, const double sigma, const DCOMPLEX * G, const DCOMPLEX* P_sigma_G, const DCOMPLEX *T_eta,\
DCOMPLEX* o_Psigma_h){
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
void des_DG( int p, int d, const double *a, const DCOMPLEX *W,DCOMPLEX*o_DG){
  DCOMPLEX sum1 = 0;
  DCOMPLEX sum2 = 0;
  for (int i = 0; i < d; i++, a++) {
    DCOMPLEX inv;
    inv = 1./(W[1]*W[0] - *a* (*a));
    inv *= inv;
    sum1 -= inv;
    sum2 -= inv*(*a)*(*a);
  }
  o_DG[0] = (1./d)*W[1]*W[1]*sum1;
  o_DG[1] = (1./p)*sum2;
  o_DG[2] = (1./d)*sum2;
  o_DG[3] = (1./p)*( W[0]*W[0] *sum1 - (double)(p-d)/ (W[1]*W[1]) )  ;
}


//
// 2 x 2
//
//
void des_Dh( const DCOMPLEX *DG, const DCOMPLEX *F,DCOMPLEX*o_Dh){
  for (int m = 0; m < 2; m++) {
    for (int n = 0; n < 2; n++, o_Dh++, DG++) {
      *o_Dh  = - F[n]*F[n]* (*DG);
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

void des_Pa_h( int p, int d, const double *a, const DCOMPLEX *W, DCOMPLEX *F, DCOMPLEX *o_Pa_h){
  for (int m = 0; m < d; m++, a++, o_Pa_h++) {
    DCOMPLEX den = W[0]*W[1]- (*a)*(*a) ;
    den *= den;
    assert(den != 0);
    DCOMPLEX inv = 1./den;
    DCOMPLEX temp = 2*(*a)*inv;
    // o_Pa_h[m][0]
    *o_Pa_h = - temp*F[0]*F[0]*W[1]/d;
    assert (isnan( creal(*o_Pa_h)) );
    // o_Pa_h[m][1]
    o_Pa_h += 1;
    *o_Pa_h = - temp*F[1]*F[1]*W[0]/p;
    assert (isnan( creal(*o_Pa_h)) );

  }
}
