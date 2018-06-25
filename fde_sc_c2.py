import numpy as np
import scipy as sp


from scipy import stats
from matrix_util import *
from random_matrices import *
import matplotlib.pyplot as plt
from timer import Timer

from itertools import chain #for ESD

import time
import logging

### To spped up forward iteration.
from cython_fde_sc_c2 import cy_cauchy_2by2, cy_cauchy_subordination

E = np.zeros([2,2,2,2])
for i in range(2):
    for j in range(2):
        E[i][j][i][j] = 1
matrix_units = np.asarray( E, np.complex128)
eye2 = np.eye(2,dtype=np.complex128)
d00eta =  matrix_units[1][1]
d01eta =  np.zeros([2,2], np.complex128)
d10eta =  np.zeros([2,2], np.complex128)
d11eta =  matrix_units[0][0]

J_eta = np.asarray([[d00eta, d01eta], [d10eta, d11eta]])

i_TEST_MODE= False

class SemiCircular(object):
    """Matrix valued SemiCircular."""
    def __init__(self,dim=1,p_dim=-1, scale=1e-1):
        super(SemiCircular, self).__init__()
        self.diag_A = np.asarray([0])
        self.sigma = 0
        self.scale= scale
        self.test_grads = False
        self.dim = dim
        ### rectangular
        ### p_dim \times dim
        if p_dim > 0:
            self.p_dim = p_dim
        else:
            self.p_dim = dim
        self.G= np.eye(2*self.dim)*(1-1j)
        self.grads = np.zeros( (self.dim+1, 2*self.dim, 2*self.dim), dtype=np.complex128)

        ### for subordination
        self.des = Descrete(self.diag_A)
        self.G2 = np.ones(2)*(-1j)
        self.grads2 = np.zeros((self.dim+1, 2,2),dtype = np.complex128)
        self.omega = np.ones(2)*1j
        self.omega_sc = np.ones(2)*1j
        self.forward_iter = np.asarray([0], dtype=np.int)

    def set_params(self, a,sigma):
        assert self.dim == a.shape[0]
        self.diag_A = a
        self.des = Descrete(self.diag_A, p_dim=self.p_dim)
        self.sigma = sigma

    def update_params(self, a,sigma):
        self.diag_A = a
        self.des.__init__(a,p_dim=self.p_dim)
        self.sigma = sigma

    def eta(self, in_mat):
        M = in_mat.shape[0]
        assert  M % 2 == 0 and  M == in_mat.shape[1]
        half_M = int(M/2)
        t2 = ntrace(in_mat[half_M:,half_M:])
        t1 = ntrace(in_mat[:half_M,:])
        #assert t2 + t1 == np.trace(in_mat)/(half_M)
        out = np.zeros(M, dtype=np.complex128)
        for i in range(half_M):
            out[i]= t2
        for i in range(half_M, M):
            out[i]= t1

        return np.diag(out)
    
    def eta_array(self, in_mat):
        M = in_mat.shape[0]
        #assert  M % 2 == 0 and  M == in_mat.shape[1]
        half_M = int(M/2)
        t2 = np.trace(in_mat[half_M:,half_M:])/half_M
        t1 = np.trace(in_mat[:half_M,:])/half_M
        #assert t2 + t1 == np.trace(in_mat)/(half_M)
        out = np.empty(M, dtype=np.complex128)
        for i in range(half_M):
            out[i]= t2
        for i in range(half_M, M):
            out[i]= t1

        return out






    ###  G^{-1} = b - \eta(G)
    ### -jbW + \eta(W)W = 1
    ###  VW + \eta(W)W = 1

    def fixed_point(self, init_mat, var_mat , max_iter=100, thres=1e-8):
        W = init_mat
        size = W.shape[0]
        sub = thres + 1
        #timer = Timer()
        #timer.tic()
        flag = False
        for it in range(max_iter):
            sub = np.linalg.inv( self.eta(W)+ var_mat) - W
            sub*= 0.5
            if it > 1 and np.linalg.norm(sub) < thres*np.linalg.norm(W):
                flag = True
            W += sub
            if flag:
                break
        #timer.toc()
        #logging.info("cauchy time={}/ {}-iter".format(timer.total_time, it))
        return W

    def cauchy(self, init_G, var_mat,sigma):
        #assert init_G.shape == var_mat.shape
        #assert sigma > 0 or sigma ==0
        if abs(sigma) == 0:
            print(sigma)
            G = np.linalg.inv(var_mat)
        else:
            init_W = 1j*init_G*sigma
            var_mat *= -1j/sigma
            W = self.fixed_point(init_W, var_mat)
            G = -1j*W/sigma
        return G

    ##TODO move this to matrix_util
    def diag_nondiag(self,z,A):
        size = A.shape[0]
        e_param_mat = np.zeros(4*size**2, dtype=np.complex128).reshape([2*size, 2*size])
        for k in range(size):
            for l in range(size):
                e_param_mat[k][size+l] = A.H[k,l]
                e_param_mat[size+k][l] = A[k,l]

        L = z*np.eye(2*size)
        #L = Lambda(z,  2*size, -1)
        out =L - e_param_mat

        return out



    def Lambda(self, z,  size, scale=-1, test=0):
        assert z.imag > 0
        if scale < 0:
            #If not using Linearizaion Trick
            out = z*np.eye(size)
        elif test==1:
            #For Linearizaion TricK
            out = np.zeros((size,size), dtype=np.complex128)
            half_size=int(size/2)
            for i in range(half_size):
                out[i][i]=z
            for i in range(half_size, size):
                out[i][i] = scale*1j
        else:
            out = np.zeros((size,size), dtype=np.complex128)
            out[0][0] = z
            for i in range(1, size):
                out[i][i] = scale*1j
        return out

    def ESD(self, num_shot, dim_cauchy_vec=0,COMPLEX=False):
        evs_list = []
        param_mat = rectangular_diag(self.diag_A, self.p_dim, self.dim)
        for n in range(num_shot):
            W = info_plus_noise(param_mat, self.sigma, COMPLEX)
            evs =  np.linalg.eigh(W)[0]

            c_noise =  sp.stats.cauchy.rvs(loc=0, scale=self.scale, size=dim_cauchy_vec)
            if dim_cauchy_vec >0:
                for k in range(dim_cauchy_vec):
                    evs_list.append( (evs - c_noise[k]).tolist())
            else:
                evs_list.append(evs.tolist())
        out = list(chain.from_iterable(evs_list))

        return out



    def ESD_symm(self, num_shot, dim_cauchy_vec=0,COMPLEX=False):
        evs_list = []
        param_mat = rectangular_diag(self.diag_A, self.p_dim, self.p_dim)
        for n in range(num_shot):
            W = info_plus_noise_symm(self.p_dim, self.dim, param_mat, self.sigma, COMPLEX)
            evs =  np.linalg.eigh(W)[0]

            c_noise =  sp.stats.cauchy.rvs(loc=0, scale=self.scale, size=dim_cauchy_vec)
            if dim_cauchy_vec >0:
                for k in range(dim_cauchy_vec):
                    evs_list.append( (evs - c_noise[k]).tolist())
            else:
                evs_list.append(evs.tolist())
        out = list(chain.from_iterable(evs_list))

        return out



    def density(self, x_array):
        size = self.dim
        param_mat = np.diag(self.diag_A)
        assert param_mat.shape[0] == size
        param_mat = np.matrix(param_mat)
        e_param_mat = np.zeros(4*size**2, dtype=np.complex128).reshape([2*size, 2*size])
        for k in range(size):
            for l in range(size):
                e_param_mat[k][size+l] = param_mat.H[k,l]
                e_param_mat[size+k][l] = param_mat[k,l]
        e_param_mat = np.matrix(e_param_mat)
        G = np.eye(2*size)*(1-1j)
        G = np.matrix(G)
        num = len(x_array)
        rho_list = []
        for i in range(num):
            x = x_array[i]
            z = sp.sqrt(x+1j*self.scale)
            L = z*np.eye(2*size)
            #L = Lambda(z,  2*size, -1)
            L = np.matrix(L)
            var_mat = L - e_param_mat
            G = self.cauchy(G, var_mat, self.sigma)
            self.G = G
            G_2 = G / z   ### zG_2(z^2) = G(z)
            rho =  -ntrace(G_2).imag/sp.pi
            #logging.debug( "(density_info_plus_noise)rho(", x, ")= " ,rho
            rho_list.append(rho)

        return np.array(rho_list)



    ### G^{-1} = b - v^2 \eta(G)
    ### - G^{-1} dG G^{-1} = db -v^2\eta(dG) -dv^2 \eta(G)
    ### dG = G(-db +   v^2\eta(dG)  + dv^2 \eta(G))G

    #if use_numba:
    def grad_by_iteration(self, G, sigma, grads_init,  max_iter=500, base_thres = 1e-6, use_numba=False):
    #    return sc_grad_by_iteration_fast(G, var_mat,sigma, grads_init,  max_iter, base_thres)
    #else:
        grads = grads_init
        ### For multiplication of matrix and diagonal matrix
        G = np.asarray(G)

        temp_max_iter =  200
        size = int(G.shape[0]/2)
        thres = base_thres

        #linalg.init()

        num_coord = size + 1

        monitor_step = 10
        #diag_nondiag = -sigma**(-2)*var_mat
        timer = Timer()
        timer.tic()
        #E = np.zeros((2*size, 2*size))
        flag = 1

        #s_gpu = gpuarray.to_gpu(np.asarray(sigma))
        M = grads.shape[1]
        half_M = int(M/2)
        t_out = np.empty(M, dtype=np.complex128)
        def _eta_array(in_mat):
                t2 = np.trace(in_mat[half_M:,half_M:])/half_M
                t1 = np.trace(in_mat[:half_M,:])/half_M
                #assert t2 + t1 == np.trace(in_mat)/(half_M)
                for i in range(half_M):
                    t_out[i]= t2
                for i in range(half_M, M):
                    t_out[i]= t1
                return t_out
        for i in range(num_coord):
          if flag == 1:
            grad = grads[i]
            E = np.zeros((2*size,2*size),np.complex128)
            if i < size:
                E[i][size+i] = 1.
                E[size+i][i] = 1.
            else:
                E= 2*sigma*self.eta(G)

            C = (G @ E) @ G

            #G_gpu = gpuarray.to_gpu(G)
            #E_gpu = gpuarray.to_gpu(E)
            #out_gpu = gpuarray.to_gpu(out)

            #C_gpu =  linalg.dot(linalg.dot(G_gpu, E_gpu), G_gpu)
            if use_numba:
                grad= iterate_grad(max_iter, C, sigma, grad, G,thres)
            else:
                sub_flag = False
                for n in range(max_iter+1):
                    ### Broadcast: return the same result as
                    ### out = C + sigma**2*G @ self.eta(out) @ G
                    ### Pay attention: Does not work for np.matrix
                    sub  = -grad + C + sigma**2*(_eta_array(grad)*G ) @ G
                    #grad = Tau_transform(C, sigma , grad.shape[0], grad, G)
                    #eta = gpuarray.to_gpu(sigma**2*self.eta(out_gpu.get()))
                    #out_gpu = C_gpu  + linalg.dot(G_gpu, linalg.dot(eta, G_gpu))
                    #print (n,np.linalg.norm(sub), np.linalg.norm(grad))
                    if n > 1 and np.linalg.norm(sub) < thres*np.linalg.norm(grad):
                        sub_flag = True
                    grad += sub
                    if sub_flag:
                        break

                    ### Moninitoring convergence
                    """
                    if n % monitor_step == monitor_step -1:
                        old_grad = np.copy(grad)
                        #old_out = out_gpu.get()
                    elif n % monitor_step == 0 and n >0:
                        #out = out_gpu.get()
                        sub = np.linalg.norm(grad- old_grad)
                        logging.debug("{} grads sub={}".format(n,sub ))
                        if sub < thres:
                            #logging.debug( "break grads at {}, sub={}".format(n, sub))
                            grads[i] = grad
                            logging.debug("coord {} : break at {}".format(i,n))
                            break
                        elif  n > temp_max_iter:
                            if sub < 1e-1 and temp_max_iter < max_iter:
                                logging.info("Continue computing grads...at i={}, n={} : sub = {}".format(i,n,sub))
                                temp_max_iter += 50
                            else:
                                logging.error( "::::::error grads is ignored.:::::at i={}, n={} : sub = {}".format(i, n, sub))
                                grads[i]= np.zeros((2*size,2*size), np.complex128)
                                flag = -1
                                break
                    """
                timer.toc()
                logging.debug("grads:time={}, {}-iter".format(timer.total_time, n))

        return grads



        ###SLow, use only for debug
    def grad_by_inverse(self,G, var_mat,sigma):
                L = G.shape[0]
                size = int(L/2)
                tp_TGe = np.zeros((L,L, L, L),np.complex128)
                for i in range(L):
                    for j in range(L):
                        E = np.zeros((L, L), np.complex128)
                        E[i][j] = 1.
                        tp_TGe[i][j] = sigma**2*G @ self.eta(E) @ G
                tpTGe = tp_TGe.reshape([L**2,L**2])

                tpS = sp.sparse.linalg.inv(sp.sparse.csc_matrix(np.eye(L**2) - tpTGe))
                """
                num = 15
                temp =  np.eye(L**2, dtype=np.complex128128)
                out = temp
                for n in range(num):
                    temp = temp @ tpTGe
                    out += temp
                tpS_Neumann = out
                norm = np.sum(np.abs(tpS - tpS_Neumann))
                logging.info("inv -Neumann={}".format(norm))
                """
                PAGe = np.zeros((size, L,L), np.complex128)

                for d in range(size):
                    E = np.zeros((L, L), np.complex128)
                    E[d][size+d] = 1.
                    E[size+d][d] = 1.
                    PAGe[d] = G@E@G
                PAGe= PAGe.reshape([size, L**2])

                out = PAGe @ tpS
                out = out.reshape([size,L,L])
                return out

    def loss(self,sample):
        density = self.density(sample)
        loss = -np.average(sp.log(density))
        return loss


    def grad_loss(self, param_array, sigma, sample):
            size = np.shape(param_array)[0]
            param_mat = np.matrix(np.diag(param_array))

            #e_param_mat = np.zeros(4*size**2, dtype=np.complex128).reshape([2*size, 2*size])
            #for k in range(size):
            #    for l in range(size):
            #        e_param_mat[k][size+l] = param_mat.H[k,l]
            #        e_param_mat[size+k][l] = param_mat[k,l]
            #e_param_mat = np.matrix(e_param_mat)
            e_param_mat = self.diag_nondiag(0, param_mat)
            num_sample = len(sample)
            rho_list = []
            num_coord = size + 1
            grads = np.zeros(num_coord)
            G = self.G
            scale = self.scale
            for i in range(num_sample):
                x = sample[i]
                z = sp.sqrt(x+1j*self.scale)
                #L = z*np.eye(2*size)
                #L = Lambda(z,  2*size, -1)
                #L = np.matrix(L)
                #var_mat = L - e_param_mat
                var_mat = z*np.eye(2*size) + e_param_mat
                G = self.cauchy(self.G, var_mat, sigma)
                ### Update initial value of G
                self.G  = G

                G_2 = G / z   ### zG_2(z^2) = G(z)
                rho =  -ntrace(G_2).imag/sp.pi
                rho_list.append(rho)

                grads_G = self.grad_by_iteration(G,sigma, grads_init=self.grads)
                #grads_G = sc_grad_by_iteration_fast(G, var_mat, sigma, self.grads)
                ### Update initial value of gradients of G
                self.grads = grads_G


                if self.test_grads:
                    timer=Timer()
                    timer.tic()
                    grad_by_inv = self.grad_by_inverse(G,  var_mat, sigma)
                    timer.toc()
                    logging.info("grad_by_inver time={}".format(timer.total_time))
                    norm = np.linalg.norm(grads[:size, :,:] - grad_by_inv)
                    logging.info("fixed_point - inverse:\n{}".format(norm) )

                ### (-log \rho)' = - \rho' / \rho
                for n in range(num_coord):
                    grads[n] += (ntrace(grads_G[n])/z).imag/(sp.pi*rho)

            loss = np.average(-sp.log(rho_list))
            grads/= num_sample
            return grads , loss



    def regularization_grad_loss(self, diag_A, sigma,reg_coef, TYPE="L1"):
        if TYPE == "L1":
            loss =  np.sum(np.abs(diag_A)) #+ abs(sigma)
            loss *= reg_coef
            grads = np.empty( self.dim + 1)
            grads[:self.dim] = np.sign(diag_A.real)
            grads[-1] = 0#np.sign(sigma)
            grads *= reg_coef
            #logging.info("LASSO: grads={}, loss={}".format(grads,loss))
        elif TYPE == "L2":
            loss =  np.sum(diag_A**2) #+ sigma**2
            loss *= reg_coef
            grads = np.empty( self.dim + 1)
            grads[:self.dim] = 2*diag_A
            grads[-1] = 0#2*sigma
            grads *= reg_coef

        return grads, loss


    ##########################
    ###### Subordinatioin ####
    ##########################
    def cauchy_subordination(self, B, \
    init_omega,init_G_sc, max_iter=1000,thres=1e-8, TEST_MODE=i_TEST_MODE, CYTHON=True):
        if not CYTHON or TEST_MODE:
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
                    #logging.debug("subo:{}-iter".format(n))
                    break
            out = self.cauchy_2by2(omega, sc_g)
            omega_sc = 1/out - omega + B
            if TEST_MODE:
                G1 = out
                G2 = des.cauchy_transform(omega_sc)
                G3 = 1/(omega + omega_sc - B)
                assert ( np.allclose(G1, G2))
                assert ( np.allclose(G1, G3))
                assert ( np.allclose(G2, G3))
            if not CYTHON:
                return out, omega, omega_sc


        if CYTHON:
            a = np.asarray(self.des.a, dtype=np.float)
            sigma = float(self.sigma)
            cy_omega_sc = np.zeros(2, dtype=np.complex)
            result = cy_cauchy_subordination(B, init_omega,init_G_sc,max_iter,thres, \
            sigma, self.p_dim, self.dim, self.forward_iter, a, cy_omega_sc)
            if result == 0:
                logging.info("cy_cauchy_subordination: reached max iter")
            if TEST_MODE:
                cy_G1 = init_G_sc
                cy_G2 = des.cauchy_transform(cy_omega_sc)
                cy_G3 = 1/(init_omega + cy_omega_sc - B)
                assert ( np.allclose(cy_G1, cy_G2))
                assert ( np.allclose(cy_G1, cy_G3))
                assert ( np.allclose(cy_G2, cy_G3))
                assert ( np.allclose(init_omega, omega))
                assert ( np.allclose(cy_omega_sc,omega_sc ))
                assert ( np.allclose(init_G_sc, out))

            return init_G_sc, init_omega, cy_omega_sc



    def rho(self, x, G, omega):
        z = x+1j*self.scale
        L = sp.sqrt(z)*np.ones(2)
        G,omega, omgega_sc = self.cauchy_subordination(B=L, init_omega=omega, init_G_sc=G)
        self.G2 = G
        G_out = G[0]/ sp.sqrt(z)
        rho =  - G_out.imag/sp.pi
        return rho, G, omega


    def rho_symm(self, x, G, omega):
        z = x+1j*self.scale
        L = z*np.ones(2)
        G,omega, omgega_sc = self.cauchy_subordination(B=L, init_omega=omega, init_G_sc=G)
        rho =- ntrace(G).imag/sp.pi
        return rho, G, omega



    def density_subordinaiton(self, x_array):
        num = len(x_array)
        omega = 1j*np.ones(2)
        G = -1j*np.ones(2)
        rho_list = []
        for i in range(num):
            rho, G, omega = self.rho(x_array[i], G, omega)
            if rho < 0:
                print(rho)
            #assert rho > 0
            rho_list.append(rho)

        return np.array(rho_list)


    def density_subordinaiton_symm(self, x_array):
        num = len(x_array)
        omega = 1j*np.eye(2,dtype=np.complex128)
        G = -1j*np.eye(2,dtype=np.complex128)
        rho_list = []
        for i in range(num):
            rho, G, omega = self.rho_symm(x_array[i], G, omega)
            if rho < 0:
                print(rho)
            #assert rho > 0
            rho_list.append(rho)

        return np.array(rho_list)

    def cauchy_2by2(self,Z,  G_init, max_iter=1000, thres=1e-8, CYTHON=True, TEST_MODE=False):
        if not CYTHON or TEST_MODE:
            G = np.copy(G_init) ### copy to avoid overlapping with cython
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
                    break
            #logging.info("cauchy_2by2: sub = {} @ iter= {}".format(np.linalg.norm(sub),d))
            if d == max_iter:
                loggin.info("cauchy_2by2: reahed max_iter")
            self.forward_iter += d
            if not TEST_MODE:
                return G
        if CYTHON:
            sigma = float(self.sigma)
            result = cy_cauchy_2by2(Z, G_init,max_iter,thres, sigma, self.p_dim, self.dim,self.forward_iter)
            if result == 0:
                loggin.info("cy_cauchy_2by2: reahed max_iter")

            G_cy = G_init
            if TEST_MODE:
                assert( np.allclose(G, G_init))

            return G_cy



    ######## Derivations of SemiCircular
    ### transpose of tangent
    ### 2 x 2
    ### i  k
    ### \part f_k / \part x_i
    def eta_2by2(self,G):
        eta = np.copy(G[::-1])
        eta[0]*=float(self.p_dim)/self.dim ### for recutangular matrix
        return eta

    def tp_T_eta(self):
        return  np.asarray([[0, 1], [float(self.p_dim)/self.dim, 0]])

    def tp_TG_Ge(self, G, TEST_MODE=i_TEST_MODE):
        eye2 = np.eye(2,dtype=np.complex128)
        out = np.empty([2,2], dtype=np.complex128)
        for i in range(2):
                out[i] = self.sigma**2*G *self.eta_2by2(eye2[i])*G

        if TEST_MODE:
            out_2 = np.empty([2,2], dtype=np.complex128)
            out_2[0] = self.sigma**2*G*eye2[1]*G
            out_2[1] = self.sigma**2*G*(self.p_dim/self.dim)*eye2[0]*G
            assert np.allclose(out, out_2)

        return out


    def tp_Tz_Ge(self, G, TEST_MODE=i_TEST_MODE):
        temp= - np.diag(1*G**2)
        out = temp
        if TEST_MODE:
            Tz_Ge = []
            eye2 = np.eye(2,dtype=np.complex128)
            for i in range(2):
                    di = - G * eye2[i] * G
                    Tz_Ge.append(di)
            Tz_Ge =  np.asarray(Tz_Ge)
            #import pdb; pdb.set_trace()
            assert np.allclose(temp, Tz_Ge)

        return out

    def tp_T_G( self, G):
        eye2 = np.eye(2,dtype=np.complex128)
        tp_S = np.linalg.inv( eye2 - self.tp_TG_Ge(G))
        T = self.tp_Tz_Ge(G) @ tp_S
        return T


    def tp_T_h( self,G):
        T= - self.sigma**2*self.tp_T_G(G) @ self.tp_T_eta()
        return T

    ######## Derivations by parameters
    def tp_Psigma_Ge(self,G):
        D= 2*self.sigma*G * self.eta_2by2(G)* G
        #D = np.reshape(D, [1,2])
        return D
    def tp_Psigma_G(self, G):
        eye2 = np.eye(2,dtype=np.complex128)
        tpS = np.linalg.inv(eye2 - self.tp_TG_Ge(G) )
        D  =  self.tp_Psigma_Ge(G) @ tpS
        return D
    def tp_Psigma_h(self,G):
        tpPsigmaG = self.tp_Psigma_G(G)
        D = -2*self.sigma*self.eta_2by2(G) \
        - self.sigma**2*tpPsigmaG @ self.tp_T_eta()
        #print("c2_tp_Psigma_h:", D)
        return D



    def squre_density_from_G(G,z):
        #0.5  for  normalized trace
        rho = -np.imag(ntrace(G)/z)/sp.pi
        return rho

    ### Requires:
    ### G_out(B)= G_sc(omega) = G_A(omega_sc) = (omega + omega_sc - B)^{-1}
    ### F_A = G_A(omega_sc)^{-1} =
    def grad_subordination(self,  z, G_out, omega, omega_sc,\
     TEST_MODE=i_TEST_MODE):
            #import pdb; pdb.set_trace()
            #print("c2:G_out:", G_out)
            self.G2 = G_out
            self.omega = omega
            self.omega_sc = omega_sc
            i_mat = z*np.ones(2,dtype=np.complex128)
            des = self.des
            if TEST_MODE:
                assert (np.allclose( G_out, self.cauchy_2by2(omega, G_init=G_out)))
                assert (np.allclose( G_out, 1/(omega + omega_sc - i_mat ) ))
                assert (np.allclose( G_out, des.cauchy_transform(omega_sc)))

            ### f and h transform of A @ omega_sc
            F_A = omega + omega_sc - i_mat
            #assert (np.allclose( 1./des.cauchy_transform(omega_sc), F_A))
            h_A = F_A - omega_sc
            #print("omega, omega_sc", omega,omega_sc)
            #print("c2:F_A", F_A)
            #import pdb; pdb.set_trace()

            ### transposed derivation of g, h of sc @ omega
            tpTGsc = self.tp_T_G( G=G_out)
            tpThsc = self.tp_T_h( G=G_out)
            tpPsigmah = self.tp_Psigma_h(G=G_out)
            tpPsigmaG = self.tp_Psigma_G(G=G_out)
            #print("c2:tpTGsc:", tpTGsc )
            #print("c2:tpThsc:", tpThsc )
            #print("c2:tpPsigmaG:", tpPsigmaG )
            #print("c2:tpPsigmah:", tpPsigmah )

            ### transposed derivation of h of A @ omega_sc
            tpTGA = des.tp_T_G(W=omega_sc)
            #print("c2:tpTGA:", tpTGA )

            tpThA  = des.tp_T_h(W=omega_sc, F=F_A)
            tpPah = des.tp_Pa_h(W=omega_sc, F=F_A)
            ### 2x2
            tpS =  np.linalg.inv(np.eye(2,dtype=np.complex128) -  tpThsc  @ tpThA )

            ### partial derivation of omega
            tpPAOmega = tpPah @  tpS
            tpPAG =  tpPAOmega @ tpTGsc

            tpPsigmaOmega = tpPsigmah @ tpThA @ tpS
            tpPsigmaG =  tpPsigmaOmega @ tpTGsc + tpPsigmaG

            if TEST_MODE:
                tpS_sc =  np.linalg.inv(np.eye(2,dtype=np.complex128) -  tpThA @ tpThsc  )
                tpPsigmaOmega_2 = tpPsigmah @  tpS_sc
                tpPsigmaG_2 =  tpPsigmaOmega_2 @ tpTGA
                assert (np.allclose(tpPsigmaG_2, tpPsigmaG))

            #assert (tpPAG.shape , [self.dim, 2])
            tpPsigmaG = np.reshape(tpPsigmaG,  [1,2])
            grad = np.append(tpPAG, tpPsigmaG, axis=0)

            return grad


    def grad_loss_subordination(self,  sample):
            num_sample = len(sample)
            rho_list = []
            num_coord = self.dim + 1
            grad = np.zeros(num_coord)
            scale = self.scale
            omega = 1j*np.ones(2)
            G = -1j*np.ones(2)


            timerF = Timer()
            timerB = Timer()


            for i in range(num_sample):
                x = sample[i]
                z = x+1j*self.scale
                w = sp.sqrt(z)
                timerF.tic()
                L = w*np.ones(2)
                G, omega, omega_sc =  self.cauchy_subordination(\
                B=L, init_omega = omega, init_G_sc=G)
                ### Update initial value of G
                timerF.toc()
                self.G2  = G
                G_out = G[0]/w  ### zG_2(z^2) = G(z)
                rho =  - G_out.imag/sp.pi
                if rho < 0:
                    import pdb; pdb.set_trace()
                assert rho > 0
                rho_list.append(rho)
                timerB.tic()
                grad_G = self.grad_subordination(w, G, omega, omega_sc)
                timerB.toc()

                self.grads2 = grad_G
                ### (-log \rho)' = - \rho' / \rho
                for n in range(num_coord):
                    grad[n] += (grad_G[n][0]/w).imag/(sp.pi*rho)


            loss = np.average(-sp.log(rho_list))
            grad/= num_sample

            logging.debug("Forward: {} sec".format(timerF.total_time))
            logging.debug("Backward: {} sec".format(timerB.total_time))

            return grad, loss



    def loss_subordination(self,  sample):
            num_sample = len(sample)
            rho_list = []
            num_coord = self.dim + 1
            grad = np.zeros(num_coord)
            scale = self.scale
            omega = 1*1j*np.eye(2,dtype=np.complex128)
            G = -1j*np.eye(2,dtype=np.complex128)


            for i in range(num_sample):
                x = sample[i]
                z = sp.sqrt(x+1j*self.scale)
                G, omega, omega_sc =  self.cauchy_subordination(\
                B=[[z,0],[0,z]], init_omega = omega, init_G_sc=G)
                ### Update initial value of G
                timerF.toc()
                self.G2  = G

                G_2 = G / z   ### zG_2(z^2) = G(z)
                rho =  -ntrace(G_2).imag/sp.pi
                rho_list.append(rho)

            loss = np.average(-sp.log(rho_list))


            return loss





class Descrete(object):
    """docstring for Descrete."""
    def __init__(self, a, p_dim=-1):
        super(Descrete, self).__init__()
        self.a = a
        self.dim = a.shape[0]
        if p_dim > 0:
            assert p_dim >= self.dim
            self.p_dim = p_dim
        else:
            self.p_dim = self.dim
        self.G = 0
        self.f = 0
        self.h = 0

    def cauchy_transform(self,W):
        #assert np.allclose(W.shape, [2,2])
        a = self.a

        sum_inv_det = np.sum( 1/(W[1]*W[0] - a*a) )
        G = [ (1/self.dim)*W[1]*sum_inv_det,\
         (1./self.p_dim)*(W[0]*sum_inv_det + (self.p_dim -self.dim)/W[1] ) ]

        """
        T = [ [W[1][1]*np.ones(self.dim), a - W[0][1]],\
            [a - W[1][0], W[0][0]*np.ones(self.dim)] ] \
              / (W[1][1]*W[0][0] - (W[0][1]-a)*(W[1][0]-a) )
        G = np.mean(T, axis=2)
        """
        return np.asarray(G)


    def h_transform(self,W):
        return 1/(self.cauchy_transform(W)) - W


    ### Transpose of total_derivation
    ### (i , j , k, l)
    ### ij -th derivation of kl entry
    ### 2 x 2
    def tp_T_G(self, W):
        a = self.a
        d = self.dim
        p = self.p_dim
        invdet = 1./ (W[1]*W[0] - a*a)
        ones = np.ones(self.dim,dtype=np.complex128)

        alpha  = np.sum(invdet**2)

        out  = [ \
        [  (1./d)*(-W[1]**2)*alpha,\
         (1./p)*np.sum(-a**2*invdet**2)],\
        [ (1./d)*np.sum(-a**2*invdet**2),\
         (1./p)*(-W[0]**2*alpha - (p-d)/W[1]**2)]\
        ]
        out = np.asarray(out).reshape([2,2])
        #print("c2:tp_T_G", out)
        return out

    ### 2 x 2
    def tp_T_h(self, W,  F):
        t_G = self.tp_T_G(W)
        T_h = np.empty([2,2], dtype=np.complex128)
        eye2 = np.eye(2,dtype=np.complex128)
        for i in range(2):
            T_h[i] = - F * t_G[i] * F - eye2[i]
        return T_h


    ### d  x 2
    def tp_Pa_CT(self, W):
        a  = self.a
        d = self.dim
        p = self.p_dim
        temp = 2*a/ ( W[0]*W[1] - a**2)**2
        P =  np.asarray([ (1./d)*W[1]*temp, (1./p)*W[0]*temp])
        return P.T

    ### d x 2
    def tp_Pa_h(self, W, F):
        tpPaCT = self.tp_Pa_CT(W)
        tpPah = np.empty( [self.dim, 2], dtype=np.complex128)
        for k in range(self.dim):
                tpPah[k] = -F*tpPaCT[k]*F
        return tpPah


class Laplace(object):
    """docstring for Laplace."""
    def __init__(self, arg):
        super(Laplace, self).__init__()
        self.d = d
        self.p = p
        self.xi = xi
