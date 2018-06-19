cimport numpy as np

cdef extern from "cython_fde_sc_c2.h":
    double complex c_algo(double complex *arr_a, double complex *arr_b, int size_a, int size_b)

cdef extern from "cython_fde_sc_c2.h":
    int c_cauchy_2by2(double complex* Z, double complex* o_G, int max_iter, float thres, double sigma, int p_dim, int dim, long* o_forward_iter)


cdef extern from "cython_fde_sc_c2.h":
    int c_cauchy_subordination(double complex* B, double complex* o_omega,double complex* o_G_sc,int max_iter,double thres, \
    double sigma, int p_dim, int dim, long* o_forward_iter,double* a,  double complex* o_omega_sc)



def cy_algo(np.ndarray[double complex, ndim=1] arr_a, np.ndarray[double complex, ndim=1] arr_b):
    return c_algo(&arr_a[0], &arr_b[0], len(arr_a), len(arr_b))



def cy_cauchy_2by2(np.ndarray[double complex, ndim=1] Z, np.ndarray[double complex, ndim=1] o_G, int max_iter, float thres, double sigma, int p_dim, int dim, np.ndarray[long, ndim=1] o_forward_iter):
    return c_cauchy_2by2(&Z[0], &o_G[0], max_iter, thres, sigma, p_dim, dim, &o_forward_iter[0])


def cy_cauchy_subordination(np.ndarray[double complex, ndim=1] B, np.ndarray[double complex, ndim=1] o_omega,np.ndarray[double complex, ndim=1] o_G_sc,int max_iter,double thres, \
double sigma, int p_dim, int dim, np.ndarray[long, ndim=1] o_forward_iter, np.ndarray[double, ndim=1] a,  np.ndarray[double complex, ndim=1] o_omega_sc):
    return c_cauchy_subordination(&B[0], &o_omega[0],&o_G_sc[0],max_iter,thres, \
    sigma, p_dim, dim, &o_forward_iter[0], &a[0],  &o_omega_sc[0])
