#include "cython_c_code.h"
#include <complex.h>
double complex c_algo(double complex* arr_a, double complex* arr_b, int size_a, int size_b){
    int res = 0;
    for(int i=0; i < size_a; i++){
        for(int j=0; j < size_b; j++){
            res = res + arr_a[i]+arr_b[j];
        }
    }
    return res;
}


/*
def cauchy_subordination(self, B, \
init_omega,init_G_sc, max_iter=1000,thres=1e-7):
    des = self.des
    omega = init_omega
    flag = False;
    sc_g = init_G_sc
    for n in range(max_iter):
        assert omega.imag[0] > 0
        assert omega.imag[1] > 0
        sc_g = self.cauchy_2by2(omega, sc_g)
        sc_h = 1/sc_g - omega
        omega_transform = des.h_transform(sc_h + B) + B
        sub = omega_transform - omega
        if np.linalg.norm(sub) < thres:
            flag = True
        omega += sub
        if flag :
            break
    out = self.cauchy_2by2(omega, sc_g)
    omega_sc = 1/out - omega + B
    return out, omega, omega_sc

def cauchy_2by2(self,Z,  G_init, max_iter=1000, thres=1e-7):
    G = G_init
    sigma = self.sigma
    flag = False

    for d in range(max_iter):
        eta = np.copy(G[::-1])
        eta[0] *=float(self.p_dim)/self.dim ### for recutangular matrix
        sub = 1/(Z - sigma**2*eta) -G
        sub *= 0.5
        if np.linalg.norm(sub) < thres:
            flag = True
        G += sub
        if flag:
            self.forward_iter += d
            return G
    #logging.info("cauchy_2by2: sub = {} @ iter= {}".format(np.linalg.norm(sub),d))
    loggin.info("cauchy_2by2: reahed max_iter")
    self.forward_iter += d
    return G_init


### descrete
def cauchy_transform(self,W):
    #assert np.allclose(W.shape, [2,2])
    a = self.a

    sum_inv_det = np.sum( 1/(W[1]*W[0] - a*a) )
    G = [ (1/self.dim)*W[1]*sum_inv_det,\
     (1./self.p_dim)*(W[0]*sum_inv_det + (self.p_dim -self.dim)/W[1] ) ]

    return np.asarray(G)


def h_transform(self,W):
    return 1/(self.cauchy_transform(W)) - W

*/
