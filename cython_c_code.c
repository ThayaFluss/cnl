#include "cython_c_code.h"

long c_algo(long* arr_a, long* arr_b, int size_a, int size_b){
    int res = 0;
    for(int i=0; i < size_a; i++){
        for(int j=0; j < size_b; j++){
            res = res + arr_a[i]+arr_b[j];
        }
    }
    return res;
}


"""
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
"""
