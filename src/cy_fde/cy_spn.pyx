cimport numpy as np

cdef extern from "test_all.h":
    int test_all(int result)

def cy_test_all(int result):
    return test_all(result)

cdef extern from "spn.h":
    int cauchy_sc(double complex* Z, double complex* o_G, int max_iter, float thres, double sigma, int p_dim, int dim, long* o_forward_iter)


cdef extern from "spn.h":
    int cauchy_spn(double complex* B, double complex* o_omega,double complex* o_G_sc,int max_iter,double thres, \
    double sigma, int p_dim, int dim, long* o_forward_iter,double* a,  double complex* o_omega_sc)



def cy_cauchy_2by2(np.ndarray[double complex, ndim=1] Z, np.ndarray[double complex, ndim=1] o_G, int max_iter, float thres, double sigma, int p_dim, int dim, np.ndarray[long, ndim=1] o_forward_iter):
    return cauchy_sc(&Z[0], &o_G[0], max_iter, thres, sigma, p_dim, dim, &o_forward_iter[0])


def cy_cauchy_subordination(np.ndarray[double complex, ndim=1] B, np.ndarray[double complex, ndim=1] o_omega,np.ndarray[double complex, ndim=1] o_G_sc,int max_iter,double thres, \
double sigma, int p_dim, int dim, np.ndarray[long, ndim=1] o_forward_iter, np.ndarray[double, ndim=1] a,  np.ndarray[double complex, ndim=1] o_omega_sc):
    return cauchy_spn(&B[0], &o_omega[0],&o_G_sc[0],max_iter,thres, \
    sigma, p_dim, dim, &o_forward_iter[0], &a[0],  &o_omega_sc[0])





cdef extern from "spn.h":
    void grad_cauchy_spn(int p, int d, const  double complex z,  const double  *a, const double  sigma, \
      const double complex *G, const double complex *omega, const double complex *omega_sc,\
        double complex *o_grad_a, double complex *o_grad_sigma)


def cy_grad_cauchy_spn(int p, int d, \
const double complex z, np.ndarray[double, ndim=1] a, const double  sigma, \
np.ndarray[double complex, ndim=1] G,   np.ndarray[double complex, ndim=1] omega,   np.ndarray[double complex, ndim=1] omega_sc,\
 np.ndarray[double complex, ndim=1] o_grad_a,   np.ndarray[double complex, ndim=1] o_grad_sigma):
     return grad_cauchy_spn(p, d, z,  &a[0], sigma, \
           &G[0], &omega[0], &omega_sc[0],\
            &o_grad_a[0], &o_grad_sigma[0])
