#include "spn.h"

int
cauchy_sc(int p_dim, int dim, double sigma, \
  DCOMPLEX* Z, \
  int max_iter, double thres,\
  DCOMPLEX*  o_G){
    if ( !(thres> 0 )){
      printf("(cauchy_sc)ERROR:Must be thres > 0\n");
      exit(EXIT_FAILURE);
    }
    if (cimag(Z[0]) < 0 || cimag(Z[1]) < 0){
       printf("(cauchy_sc)ERROR:Must be Im Z > 0 \n");
       exit(EXIT_FAILURE);
     }
    int flag = 0;
    int num_iter = 0;
    DCOMPLEX sub_x = 0;
    DCOMPLEX sub_y = 0;
    for(int n = 0; n < max_iter; ++n){
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
        num_iter =n;
        break;
      }
    }
    if (flag == 0){
      num_iter = max_iter;
      printf("(cauchy_sc)reached max_iter.\n");
    }
    return num_iter;
}


int
cauchy_spn(int p_dim, int dim, double* a,double sigma,\
  DCOMPLEX* B,\
  int max_iter,double thres, \
  DCOMPLEX* o_G_sc, DCOMPLEX* o_omega, DCOMPLEX* o_omega_sc){
    int flag = 0;
    int num_total_iter = 0;
    if ( !(thres > 0)){
      printf("(cauchy_spn)ERROR: Must be thres > 0 : %e\n" , thres);
      exit(EXIT_FAILURE);
    }
    for (int n = 0; n< max_iter; ++n){
        num_total_iter += cauchy_sc(p_dim,dim, sigma, o_omega,\
           max_iter, thres,\
          o_G_sc);
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
    num_total_iter += cauchy_sc(p_dim,dim, sigma,o_omega, \
      max_iter, thres, \
      o_G_sc);
    o_omega_sc[0] = 1./o_G_sc[0] - o_omega[0] + B[0];
    o_omega_sc[1] = 1./o_G_sc[1] - o_omega[1] + B[1];
    return num_total_iter;
}




void
grad_cauchy_spn(int p, int d, double  *a,  double  sigma, \
  DCOMPLEX z,DCOMPLEX *G, DCOMPLEX *omega, DCOMPLEX *omega_sc,\
  DCOMPLEX *o_grad_a, DCOMPLEX *o_grad_sigma){

  MEM_RESULT ret = MEM_OK;
  /// init ///
  DCOMPLEX *F_A =  (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*2);
  DCOMPLEX *h_A=  (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*2);
  memset(F_A, 0,2);
  memset(h_A, 0,2);

  DCOMPLEX *TG_Ge_sc =  (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*4);
  DCOMPLEX *DG_sc =  (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*4);
  memset(TG_Ge_sc, 0,4);
  memset(DG_sc, 0,4);

  DCOMPLEX *temp_T_eta =  (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*4);
  DCOMPLEX *Dh_sc =  (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*4);
  memset(temp_T_eta, 0,4);
  memset(Dh_sc, 0,4);


  DCOMPLEX *Psigma_G_sc = (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*2);
  memset(Psigma_G_sc, 0, 2);
  //Psigma_G_sc = malloc( sizeof(DCOMPLEX)*2);

  DCOMPLEX *Psigma_h_sc = (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*2);
  memset(Psigma_h_sc, 0,2);


  DCOMPLEX *DG_A, *Dh_A;
  DG_A =  (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*4);
  Dh_A =  (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*4);
  memset(DG_A, 0,4);
  memset(Dh_A, 0,4);

  // O(d)
  DCOMPLEX * Pa_h_A=  (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*d*2);
  memset(Pa_h_A, 0,d*2);


  DCOMPLEX * S = (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*4);
  memset(S, 0, 4);

  // O(d)
  DCOMPLEX * Pa_omega = (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*d*2);
  memset(Pa_omega, 0,d*2);
  DCOMPLEX * temp_mat=(DCOMPLEX*)malloc(sizeof(DCOMPLEX)*4);
  memset(temp_mat, 0, 4);

  DCOMPLEX * Psigma_omega= (DCOMPLEX*)malloc(sizeof(DCOMPLEX)*2);
  memset(Psigma_omega, 0, 2);



  /// run ///
  //printf("-----(grad_cauchy_spn)------\n");

  F_A[0] = omega[0]+omega_sc[0]-z;
  F_A[1] = omega[1]+omega_sc[1]-z;

  // TODO separate check list:
  assert ( cabs(F_A[0]*G[0] - 1) < 1e-8 );
  assert ( cabs(F_A[1]*G[1] - 1) < 1e-8 );


  if (!(  cabs(F_A[0]*G[0] - 1) < 1e-8 || ( cabs(F_A[1]*G[1] - 1) < 1e-8 ) ) ){
    printf("omega error\n");
  }

  h_A[0] = omega[0] -z;
  h_A[1] = omega[1] -z;

  //printf("F_A=%e\n", cabs(F_A[0]));
  //printf("F_A=%e\n", cabs(F_A[1]));

  //printf("G=%e\n", cabs(G[0]));
  //printf("G=%e\n", cabs(G[1]));
  TG_Ge(p,d,sigma,G, TG_Ge_sc);
  /*
  printf("abs TG_Ge_sc=%e\n", cabs(TG_Ge_sc[0]));
  printf("abs TG_Ge_sc=%e\n", cabs(TG_Ge_sc[1]));
  printf("abs TG_Ge_sc=%e\n", cabs(TG_Ge_sc[2]));
  printf("abs TG_Ge_sc=%e\n", cabs(TG_Ge_sc[3]));
  //*/
  DG(G, TG_Ge_sc, DG_sc);
  //printf(" DG_sc= %e \n", cabs(DG_sc[0]));
  //printf(" DG_sc= %e \n", cabs(DG_sc[1]));
  //printf(" DG_sc= %e \n", cabs(DG_sc[2]));
  //printf(" DG_sc= %e \n", cabs(DG_sc[3]));
  T_eta(p,d,temp_T_eta);
  //printf("T_eta=%e\n", cabs(temp_T_eta[0]));
  //printf("T_eta=%e\n", cabs(temp_T_eta[1]));
  //printf("T_eta=%e\n", cabs(temp_T_eta[2]));

  Dh(DG_sc, temp_T_eta, sigma, Dh_sc);

  //printf("Dh_sc=%e\n", cabs(Dh_sc[0]));
  //printf("Dh_sc=%e\n", cabs(Dh_sc[1]));
  //printf("Dh_sc=%e\n", cabs(Dh_sc[2]));
  //printf("Dh_sc=%e\n", cabs(Dh_sc[3]));

  Psigma_G(p,d,sigma, G, TG_Ge_sc, Psigma_G_sc);

  /*
  printf("abs P_sigma_G_sc=%e\n", cabs(Psigma_G_sc[0]));
  printf("abs P_sigma_G_sc=%e\n", cabs(Psigma_G_sc[1]));
  //*/

  Psigma_h(p, d, sigma,G, Psigma_G_sc, temp_T_eta,\
  Psigma_h_sc);
  /*
  printf("abs Psigma_h %e\n", cabs(Psigma_h_sc[0]));
  printf("abs Psigma_h %e\n", cabs(Psigma_h_sc[1]));
  //*/
  des_DG(p, d, a, omega_sc, DG_A);

  //printf("abs  DG_A=%e\n", cabs(DG_A[0]));
  //printf("abs  DG_A=%e\n", cabs(DG_A[1]));
  //printf("abs  DG_A=%e\n", cabs(DG_A[2]));
  //printf("abs  DG_A=%e\n", cabs(DG_A[3]));
  des_Dh(DG_A, F_A,  Dh_A);

  /*
  printf("abs  dha=%e\n", cabs(Dh_A[0]));
  printf("abs  dha=%e\n", cabs(Dh_A[1]));
  printf("abs  dha=%e\n", cabs(Dh_A[2]));
  printf("abs  dha=%e\n", cabs(Dh_A[3]));
  //*/
  des_Pa_h(p, d,  a,  omega_sc, F_A,  Pa_h_A);
  /*
  for (int i = 0; i < d*2; i++) {
  printf("abs  Pa_h_A=%e\n", cabs(Pa_h_A[i]));
  }
  //*/
  ////tpS =  np.linalg.inv(np.eye(2,dtype=np.complex128) -  tpThsc  @ tpThA

  S[0] = 1.;
  S[1] = 0.;
  S[2] = 0.;
  S[3] = 1.;
  my_zgemm(2,2,2 , -1.0, Dh_sc, Dh_A, 1.0, S );
  /*
  printf("pre:abs  S=%e\n", cabs(S[0]));
  printf("abs  S=%e\n", cabs(S[1]));
  printf("abs  S=%e\n", cabs(S[2]));
  printf("abs  S=%e\n", cabs(S[3]));
  //*/
  inv2by2_overwrite(S);
  /*
  printf("inv:abs  S=%e\n", cabs(S[0]));
  printf("abs  S=%e\n", cabs(S[1]));
  printf("abs  S=%e\n", cabs(S[2]));
  printf("abs  S=%e\n", cabs(S[3]));
  //*/
  my_zgemm(d,2,2 , 1.0, Pa_h_A, S, 0, Pa_omega );
  z_isnan(d*2, Pa_omega);
  /*
  printf("abs  Pa_omega=%e\n", cabs(Pa_omega[0]));
  printf("abs  Pa_omega=%e\n", cabs(Pa_omega[1]));
  printf("abs  Pa_omega=%e\n", cabs(Pa_omega[2]));
  printf("abs  Pa_omega=%e\n", cabs(Pa_omega[3]));
  */
  my_zgemm(d,2,2 , 1.0, Pa_omega, DG_sc, 0,  o_grad_a);

  z_isnan(d*2, o_grad_a);

  my_zgemm(2,2,2, 1.0, Dh_A, S, 0, temp_mat);

  /*
  printf("abs  temp_mat=%e\n", cabs(temp_mat[0]));
  printf("abs  temp_mat=%e\n", cabs(temp_mat[1]));
  //*/

  my_zgemm(1,2,2, 1.0, Psigma_h_sc, temp_mat, 0, Psigma_omega);
  /*
  printf("abs  Psigma_omega=%e\n", cabs(Psigma_omega[0]));
  printf("abs  Psigma_omega=%e\n", cabs(Psigma_omega[1]));
  //*/
  // tpPsigmaG =  tpPsigmaOmega @ tpTGsc + tpPsigmaG
  my_zgemm(1,2,2, 1.0, Psigma_omega, DG_sc, 1.0, Psigma_G_sc);

  /*
  printf("abs  Psigma_G_sc=%e\n", cabs(Psigma_G_sc[0]));
  printf("abs  Psigma_G_sc=%e\n", cabs(Psigma_G_sc[1]));
  //*/
  o_grad_sigma[0] = Psigma_G_sc[0];
  o_grad_sigma[1] = Psigma_G_sc[1];
  /*
  printf("abs  o_grad_sigma=%e\n", cabs(o_grad_sigma[0]));
  printf("abs  o_grad_sigma=%e\n", cabs(o_grad_sigma[1]));
  //*/
  z_isnan(2, o_grad_sigma);




  // destroy
  free(F_A);
  F_A =NULL;
  free(h_A);
  h_A =NULL;

  free(TG_Ge_sc);
  TG_Ge_sc =NULL;
  free(DG_sc);
  DG_sc =NULL;

  free(temp_T_eta);
  temp_T_eta =NULL;
  free(Dh_sc);
  Dh_sc=NULL;

  free(Psigma_h_sc);
  Psigma_h_sc=NULL;

  free(DG_A);
  DG_A =NULL;
  free(Dh_A);
  Dh_A =NULL;
  free(Pa_h_A);
  Pa_h_A=NULL;
  free( S);
  S=NULL;

  free( Pa_omega);
  Pa_omega =NULL;
  free( temp_mat);
  temp_mat =NULL;

  free( Psigma_omega);
  Psigma_omega =NULL;
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
  memset(o_Psigma_G, 0 ,2);
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
  my_zgemm(1,2,2, 1.0, Psigma_Ge,  S,0, o_Psigma_G);

  free(S);
  free(Psigma_Ge);
  S =NULL;
  Psigma_Ge = NULL;
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
  for (int m = 0; m < d; ++m, ++a, ++o_Pa_h) {
    DCOMPLEX den = W[0]*W[1]- (*a)*(*a) ;
    den *= den;
    if (den == 0)printf("(des_Pa_h)0 division\n");
    assert(den != 0);
    DCOMPLEX inv = 1./den;
    DCOMPLEX temp = 2*(*a)*inv;
    // o_Pa_h[m][0]
    *o_Pa_h = - temp*F[0]*F[0]*W[1]/d;
    // o_Pa_h[m][1]
    o_Pa_h += 1;
    *o_Pa_h = - temp*F[1]*F[1]*W[0]/p;
  }
}



int
grad_loss_cauchy_spn(  int p, int d, \
  double  *a, double  sigma, double scale,\
  int  num_sample, double *sample, \
  double *o_grad_a, double *o_grad_sigma, double *o_loss){

    assert(num_sample > 0);


    DCOMPLEX*omega;
    omega = malloc(sizeof(DCOMPLEX)*2);
    omega[0] = I;
    omega[1] = I;


    DCOMPLEX*omega_sc;
    omega_sc = malloc(sizeof(DCOMPLEX)*2);
    omega_sc[0] = I;
    omega_sc[1] = I;

    DCOMPLEX*G;
    G = malloc(sizeof(DCOMPLEX)*2);
    G[0] = -I;
    G[1] = -I;

    int total_f_iter = 0;
    const int max_iter = 1000;
    const double thres = 1e-8;

    //printf("(grad_loss_cauchy_spn)before for loop\n" );
    DCOMPLEX*B;
    B = malloc(sizeof(DCOMPLEX)*2);
    memset(B, 0, 2);

    DCOMPLEX*t_grad_a;
    t_grad_a = malloc(sizeof(DCOMPLEX)*d*2);
    DCOMPLEX*t_grad_sigma;
    t_grad_sigma = malloc(sizeof(DCOMPLEX)*2);

    memset(t_grad_a, 0, d*2);
    memset(t_grad_sigma, 0, 2);

    /*
    for (size_t i = 0; i < 2; i++) {
      printf("(grad_loss_cauchy_spn)grad_sigma.imag %e\n", cimag(t_grad_sigma[i]));
    }
    printf("(grad_loss_cauchy_spn)t_grad_sigma:%p\n", t_grad_sigma);
    printf("(grad_loss_cauchy_spn)t_grad_a:%p\n", t_grad_a);

    printf("(grad_loss_cauchy_spn)----------for loop about sample----------\n");
    //*/

    for (int n = 0; n < num_sample; n++) {
      DCOMPLEX z = sample[n] + scale*I;
      DCOMPLEX w = csqrt(z);
      B[0] = w;
      B[1] = w;

      total_f_iter += cauchy_spn(p, d,a,sigma,\
        B, \
        max_iter,thres, \
        G,omega, omega_sc);

      /*
      for (size_t i = 0; i < 2; i++) {
          printf("(grad_loss_cauchy_spn)G.imag: %e\n", cimag(G[i]));
      }

      printf("(grad_loss_cauchy_spn)after cauchy_spn\n" );
      //*/

      double den = -cimag(G[0]*1./w);
      double rho = den/M_PI;


      if (!(rho>0)){
        printf("(grad_loss_cauchy_spn)Must rho > 0: %e\n",rho );
        exit(EXIT_FAILURE);
      }

      assert(rho > 0);
      *o_loss -= log(rho);

      memset(t_grad_a, 0, d*2);
      memset(t_grad_sigma, 0, 2);

      /*
      for (size_t i = 0; i < 2; i++) {
        printf("(grad_loss_cauchy_spn)t_grad_sigma.imag %e\n", cimag(t_grad_sigma[i]));
      }
      printf("(grad_loss_cauchy_spn)t_grad_sigma:%p\n", t_grad_sigma);
      printf("(grad_loss_cauchy_spn)t_grad_a:%p\n", t_grad_a);
      printf("(grad_loss_cauchy_spn)--------------begin grad_cauchy_spn ...\n");
      //*/
      grad_cauchy_spn(p, d, a,sigma, \
          w, G, omega,  omega_sc,\
          t_grad_a, t_grad_sigma);
      /*
      printf("(grad_loss_cauchy_spn)--------------end grad_cauchy_spn ...\n");

      for (size_t i = 0; i < 2; i++) {
        printf("(grad_loss_cauchy_spn)t_grad_sigma.imag %e\n", cimag(t_grad_sigma[i]));
      }
      printf("(grad_loss_cauchy_spn)t_grad_sigma:%p\n", t_grad_sigma);
      printf("(grad_loss_cauchy_spn)t_grad_a:%p\n", t_grad_a);
      for (size_t i = 0; i < 2*d; i++) {
        printf("grad_a %e\n", t_grad_a[i]);
      }
      //*/

      for (int i = 0; i < d; i++) {
      o_grad_a[i] += cimag(t_grad_a[2*i]/w)/den;
      }
      /*
      printf("(grad_loss_cauchy_spn)after grad_cauchy\n" );
      //*/
      o_grad_sigma[0] += cimag(t_grad_sigma[0]/w)/den;


    }
    /*
    for (size_t i = 0; i < d; i++) {
        printf("o_grad_a: %e\n", o_grad_a[i]);
    }
    */

    my_dax(d*2, 1./num_sample,  o_grad_a);
    my_dax(2, 1./num_sample,  o_grad_sigma);
    *o_loss /=  num_sample;

    //printf("before free\n" );

    free(B);
    B=NULL;
    free(t_grad_a);
    t_grad_a=NULL;
    free(t_grad_sigma);
    t_grad_sigma=NULL;


    free(omega);
    free(omega_sc);
    free(G);
    omega= NULL;
    omega_sc= NULL;
    G= NULL;
    return total_f_iter;
}
