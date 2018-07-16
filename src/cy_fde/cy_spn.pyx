cimport numpy as np

cdef extern from "test_all.h":
    int test_all(int result)

def cy_test_all(int result):
    return test_all(result)

cdef extern from "spn.h":
    int cauchy_sc(int p_dim, int dim, double sigma, double complex* Z, \
    int max_iter, float thres,\
    double complex* o_G )

def cy_cauchy_2by2(int p_dim, int dim, double sigma,\
 np.ndarray[double complex, ndim=1] Z, \
 int max_iter, float thres, \
 np.ndarray[double complex, ndim=1] o_G):
    return cauchy_sc(p_dim, dim, sigma, &Z[0], \
     max_iter, thres,  \
     &o_G[0])


cdef extern from "spn.h":
    int cauchy_spn( int p_dim, int dim,double* a, double sigma,\
    double complex* B,\
    int max_iter,double thres, \
    double complex* o_G_sc,double complex* o_omega, double complex* o_omega_sc)



def cy_cauchy_subordination(int p_dim, int dim,\
  np.ndarray[double, ndim=1] a, double sigma,\
  np.ndarray[double complex, ndim=1] B, \
  int max_iter,double thres,\
  np.ndarray[double complex, ndim=1] o_G_sc,\
  np.ndarray[double complex, ndim=1] o_omega,\
  np.ndarray[double complex, ndim=1] o_omega_sc):
    return cauchy_spn( p_dim, dim, &a[0], sigma, \
    &B[0],\
    max_iter,thres, \
    &o_G_sc[0],&o_omega[0],&o_omega_sc[0])





cdef extern from "spn.h":
    void grad_cauchy_spn(int p, int d,   const double  *a, const double  sigma, \
      const  double complex z,  double complex *G, const double complex *omega, const double complex *omega_sc,\
        double complex *o_grad_a, double complex *o_grad_sigma)


def cy_grad_cauchy_spn(int p, int d, \
 np.ndarray[double, ndim=1] a, const double  sigma, \
 const double complex z,\
 np.ndarray[double complex, ndim=1] G,   np.ndarray[double complex, ndim=1] omega,   np.ndarray[double complex, ndim=1] omega_sc,\
 np.ndarray[double complex, ndim=1] o_grad_a,   np.ndarray[double complex, ndim=1] o_grad_sigma):
     return grad_cauchy_spn(p, d,  &a[0], sigma, \
           z,&G[0], &omega[0], &omega_sc[0],\
            &o_grad_a[0], &o_grad_sigma[0])





cdef extern from "spn.h":
    int grad_loss_cauchy_spn( \
      int p, int d,  double  *a, double sigma, double scale,\
      int  num_sample, double *sample, \
      double *o_grad_a, double *o_grad_sigma, double *o_loss)

def cy_grad_loss_cauchy_spn(\
  int p, int d, np.ndarray[double, ndim=1] a,  double  sigma, double scale,\
  int  num_sample,  np.ndarray[double, ndim=1] sample, \
  np.ndarray[double, ndim=1] o_grad_a, \
  np.ndarray[double, ndim=1] o_grad_sigma,\
  np.ndarray[double, ndim=1] o_loss):
      return  grad_loss_cauchy_spn( p, d,&a[0], sigma, scale,\
                num_sample, &sample[0], \
                &o_grad_a[0], &o_grad_sigma[0], &o_loss[0])
