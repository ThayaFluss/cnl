import tensorflow as tf
import numpy as np
import scipy as sp
from timer import Timer
from cauchy import SemiCircular
import matplotlib.pyplot as plt

import logging
##################################
#### test and summary ############
##################################
test_mode = False
gather_summary = False



###################################
#### local variables ##############
with tf.name_scope("matrix_units"):
    E = np.zeros([2,2,2,2])
    for i in range(2):
        for j in range(2):
            E[i][j][i][j] = 1
    matrix_units = tf.constant( E, tf.complex128)


inv_matrix_units=\
tf.convert_to_tensor(\
[[matrix_units[1][1], -matrix_units[0][1]], [-matrix_units[1][0], matrix_units[0][0]]])


with tf.name_scope("T_eta"):
    d00eta =  matrix_units[1][1]
    d01eta =  tf.zeros([2,2], tf.complex128)
    d10eta =  tf.zeros([2,2], tf.complex128)
    d11eta =  matrix_units[0][0]

    J_eta = tf.convert_to_tensor([[d00eta, d01eta], [d10eta, d11eta]])
    T_eta = tf.reshape(J_eta, [4,4])

eye4 = tf.eye(4, dtype=tf.complex128)
eye2 = tf.eye(2, dtype=tf.complex128)

####################################
#### Defintions of parts
####################################

####################################
###### For SemiCircular ############
####################################

def sc_cauchy_2by2(i_mat, sigma, G_init, depth=20):
    with tf.name_scope("sc_cauchy"):
        G = G_init
        for d in range(depth):
            with tf.name_scope("sc_iteration"):
                eta =  sigma**2*sc_eta_2by2(G)
                G = G + tf.matrix_inverse(i_mat - eta,name="Ge")
                G*= 0.5
    return G
def sc_eta_2by2(G):
    eta = tf.matrix_diag(tf.diag_part(G)[::-1], name="eta")
    return eta

def sc_h_2by2(i_mat, sigma, G_init,depth):
    ### h(z) = G(z)^{-1} - z = sigma**2*eta(G(z))
    with tf.name_scope("sc_h"):
        G = sc_cauchy_2by2(i_mat, sigma, G, depth)
        h = - sigma**2*sc_eta_2by2(G)
    return h


def A_subblock(A, dim_A, dim_o=2):
    with tf.name_scope("A_subblock"):
        dim = int(dim_A/dim_o)
        A_sub = tf.reshape(A, [dim_o,dim,dim_o,dim])
        A_sub = tf.transpose(A_sub, [0,2,1,3])
    return A_sub


######## Derivations of SemiCircular
### transpose of tangent
### 4 x 4
### i  k
### \part f_k / \part x_i
def tp_TG_Ge(G, sigma):
    with tf.name_scope("TG_Ge"):
        out = []
        for i in range(2):
            for j in range(2):
                entry = sigma**2*G @ sc_eta_2by2(matrix_units[i][j]) @ G
                out.append(entry)
        out = tf.convert_to_tensor(out)
        out = tf.reshape(out, [4,4])
    return out



def tp_Tz_Ge(G, sigma):
    with tf.name_scope("Tz_Ge"):
        """
        Tz_Ge = []
        for i in range(2):
            for j in range(2):
                dij = - G @ matrix_units[i][j] @ G
                Tz_Ge.append(dij)
        Tz_Ge = tf.convert_to_tensor(Tz_Ge)
        Tz_Ge = tf.reshape(Tz_Ge, [4,4])
        """
        ###
        G_tile = tf.tile(tf.reshape(G, [4]), [4])
        G_tile = tf.reshape(G_tile, [2,2,2,2])
        Tz_new = - G_tile @ matrix_units @ G_tile
        Tz_new = tf.reshape(Tz_new, [4,4])
        #L1_tester(Tz_Ge, Tznew)
    return Tz_new

def tp_T_G_sc( G, sigma):
    with tf.name_scope("T_G_sc"):
        tp_S = tf.matrix_inverse( eye4 - tp_TG_Ge(G,sigma))
        T = tp_Tz_Ge(G, sigma) @ tp_S
    return T


def tp_T_h_sc( G, sigma):
    with tf.name_scope("T_h_sc"):
        T= - sigma**2*tp_T_G_sc(G,sigma)@T_eta
    return T

######## Derivations by parameters
def tp_Psigma_Ge(G, sigma):
    with  tf.name_scope("Psigma_Ge"):
        D= 2*sigma*G @ sc_eta_2by2(G)@ G
        D = tf.reshape(D, [1,4])
    return D
def tp_Psigma_G_sc(G, sigma):
    with tf.name_scope("Psigma_G_sc"):
        tpS = tf.matrix_inverse(eye4 - tp_TG_Ge(G, sigma) )
        D  =  tp_Psigma_Ge(G,sigma) @ tpS
    return D
def tp_Psigma_h_sc(G,sigma):
    with tf.name_scope("Psigma_h_sc"):
        tpPsigmaG = tp_Psigma_G_sc(G,sigma)
        D = -2*sigma*tf.reshape(sc_eta_2by2(G), [4]) \
        - sigma**2* tpPsigmaG @ T_eta
    return D


###########################################
###### For deterministic matrices ########
##########################################
def Cauchy_from_sub(i_mat, A_sub, dim_A, dim_o=2):
    with tf.name_scope("enlarge"):
        dim = int(dim_A/dim_o)
        entries = []
        ones = tf.ones(dim,dtype=tf.complex128)
        for i in range(dim_o):
            row = []
            for j in range(dim_o):
                row.append(i_mat[i][j]*ones)
            entries.append(row)
        W_enlarge = tf.matrix_diag(entries,name="enlarge")
    with tf.name_scope("Cauchy_from_sub"):
        concated = tf.transpose( W_enlarge - A_sub,perm=[0,2,1,3])
        concated = tf.reshape(concated, [dim_A, dim_A])
        inv = tf.matrix_inverse(concated, name="for_cauchy_A")
        inv = tf.reshape(inv, [dim_o, dim, dim_o, dim])
        inv = tf.transpose(inv, [0,2,1,3])

        G = tf.trace(inv,name="subtrace")/dim
    return  G
def Cauchy(i_mat, A, dim_A, dim_o=2):
    return Cauchy_from_sub(i_mat, A_subblock(A, dim_A, dim_o), dim_A, dim_o)

### (id \otimes tr)[ (W-A)^{-1} ]
def det_mat_valued_from_array(i_mat, array, dim):
    with tf.name_scope("det"):
        one = tf.ones(dim, dtype=tf.complex128)
        det = i_mat[0][0]*i_mat[1][1]*one - (i_mat[0][1]*one-tf.conj(array))*(i_mat[1][0]*one-array)
    return det
def Cauchy_from_array(i_mat, array,dim):
    with tf.name_scope("cauchy_from_array"):
        inv_det = tf.reciprocal(det_mat_valued_from_array(i_mat,array,dim))
        tr_inv_det = tf.reduce_mean(inv_det)
        t_one = tf.ones(dim, dtype=tf.complex128)
        with tf.name_scope("direct"):
            G00 = i_mat[1][1]*tr_inv_det
            G01 = -i_mat[0][1]*tr_inv_det  + tf.reduce_mean(tf.conj(array)*inv_det )
            G10 = - i_mat[1][0]*tr_inv_det + tf.reduce_mean(array*inv_det)
            G11 = i_mat[0][0]*tr_inv_det
            G = tf.convert_to_tensor([[G00, G01], [G10, G11]])
    return G




###### Derivations of deterministic matrices
### T_det: 2 x 2  x d
def tp_J_det(i_mat, array, dim):
    with tf.name_scope("tp_J_det"):
        J = tf.convert_to_tensor(\
        [[i_mat[1][1]*tf.ones(dim, dtype=tf.complex128), -i_mat[1][0]+ array],\
        [- i_mat[0][1] + array, i_mat[0][0]*tf.ones(dim, dtype=tf.complex128)]])
    return J


### 2 x 2 x 2 x2
def tp_J_V(i_mat):
    with tf.name_scope("tp_T_V"):
        P00V = matrix_units[1][1]
        P01V = -matrix_units[0][1]
        P10V = -matrix_units[1][0]
        P11V = matrix_units[0][0]
        T = tf.convert_to_tensor([P00V, P01V, P10V, P11V], tf.complex128)
        J = tf.reshape(T, [2,2,2,2])
    return J
###### 2 x 2 x 2 x 2
###### ( i,j,k,l)
###### (i,j)-th partial derivatin of (G_A)_kl

def tp_J_G_A(i_mat, array, dim ):
    with tf.name_scope("tp_J_G_A"):
        ### det: d x 1
        det = det_mat_valued_from_array(i_mat, array,dim)
        tpJdet = tp_J_det(i_mat, array, dim)
        tpJinvdet = - tpJdet/det**2
        ### tr_T_invdet : (4)
        V = tf.convert_to_tensor(\
        [[i_mat[1][1], -i_mat[0][1]],\
         [-i_mat[1][0], i_mat[0][0]]])
        tpJV = tp_J_V(i_mat)

        C = matrix_units[0][1] + matrix_units[1][0]

        tr_invdet = tf.reduce_mean(1./det)
        ### subtrace: 2 x 2
        subtr_J_invdet = tf.reduce_mean(tpJinvdet, axis=2)

        subtr_J_array_invdet = tf.reduce_mean(array*tpJinvdet, axis=2)
        J = []
        for i in range(2):
            for j in range(2):
                P = tr_invdet*tpJV[i][j]\
                + subtr_J_invdet[i][j]*V\
                + subtr_J_array_invdet[i][j]*C
                J.append(P)
        J = tf.convert_to_tensor(J)
        J = tf.reshape(J, [2,2,2,2])
    return J

### 4 x 4
### (ij) x (kl)
### T_G_A : \part(G_A)_ij / \part z_kl
### tp_T_G_A : \part(G_A)_kl / \part z_ij
def tp_T_G_A(i_mat, array, dim):
    with tf.name_scope("tp_T_G_A"):
        T = tf.reshape(tp_J_G_A(i_mat, array, dim), [4,4])
    return T


### F_A(mat) = G_A(mat)^{-1}
def tp_T_h_A(i_mat, array, dim, F_A):
    with tf.name_scope("tp_T_h_A"):
        t_J = tp_J_G_A(i_mat, array, dim)
        """
        T_h_list = []
        for i in range(2):
            for j in range(2):
                entry = - F_A @ t_J[i][j] @ F_A - matrix_units[i][j]
                T_h_list.append(entry)
        T_h = tf.convert_to_tensor(T_h_list, dtype=tf.complex128)
        T_h = tf.reshape(T_h, [4,4])
        """
        F_A_tile = tf.tile(tf.reshape(F_A, [4]), [4])
        F_A_tile = tf.reshape(F_A_tile, [2,2,2,2])
        tp_T_h_new = - F_A_tile @ t_J @ F_A_tile - matrix_units
        tp_T_h_new = tf.reshape(tp_T_h_new, [4,4])
        #L1_tester(T_h, T_h_new)
    return tp_T_h_new

###### Derivations by parameters
### dim x dim
### m x n
### \part det_m/ \part d_n
def diag_PA_det(W, array):
    with tf.name_scope("diag_PA_det"):
        ones = tf.ones(36,tf.complex128)
        x = W[1][0]*ones -  array + W[0][1]*ones - tf.conj(array)
    return x


def PA_det(i_mat, array):
    with tf.name_scope("PA_det"):
        out = tf.matrix_diag(diag_PA_det(i_mat, array))
    return out


### d x 4
def tp_PA_G_A(i_mat, array, dim):
    with tf.name_scope("PA_G_A"):
        det = det_mat_valued_from_array(i_mat, array,dim)
        i_PA_det = PA_det(i_mat, array)
        D_list = []
        one = tf.ones(dim, dtype=tf.complex128)
        eye = tf.eye(dim,dtype=tf.complex128)
        for j in range(dim):
            D_invdet = - i_PA_det[j]/det**2
            d00 = i_mat[1][1]*tf.reduce_mean(D_invdet)
            d01 = - tf.reduce_mean( - eye[j]/det + (i_mat[0][1]*one - tf.conj(array))*(D_invdet) )
            d10 = - tf.reduce_mean( - eye[j]/det + (i_mat[1][0]*one - array)*(D_invdet) )
            d11 = i_mat[0][0]*tf.reduce_mean(D_invdet)
            D_list.append([d00, d01, d10, d11])
        D = tf.convert_to_tensor(D_list, tf.complex128)

        """
        ###4 x d-version
        det = det_mat_valued_from_array(i_mat, array,dim)
        dgPAdet = diag_PA_det(i_mat, array)
        ### (d)
        PA_invdet = - dgPAdet/det**2
        ### PA tr(array/det)
        PA_x = 1./(det*dim) + tf.reduce_mean(array*PA_invdet)
        tr_inv_det = tf.reduce_mean(PA_invdet)

        PA_G00 = i_mat[1][1]*PA_invdet
        PA_G01 = -i_mat[0][1]*PA_invdet + PA_x
        PA_G10 = -i_mat[1][0]*PA_invdet + PA_x
        PA_G11 = i_mat[0][0]*PA_invdet

        ### shape = (4, d)
        D = tf.convert_to_tensor([PA_G00, PA_G01,PA_G10, PA_G11], tf.complex128)
        """
    return D

### d x 4
def tp_PA_h_A(i_mat, array, dim, h):
    with tf.name_scope("PA_h_A"):
        tpPAG = tp_PA_G_A(i_mat, array, dim)
        tpPAG = tf.reshape(tpPAG, [dim,2,2])
        """
        PA_h= []
        for j in range(dim):
            PA_h.append(- h @ PA_G[j] @  h ) #
        PA_h =  tf.convert_to_tensor(PA_h, tf.complex128)
        PA_h = tf.reshape(PA_h, [dim, 4])
        L1_tester(PA_h, PA_h_new)
        """
        #@TODO not hT , but fT ?
        h_tile = tf.tile(tf.reshape(h, [4]), [dim])
        h_tile = tf.reshape(h_tile,[dim,2,2])
        PA_h_new = - h_tile @ tpPAG @ h_tile
        PA_h_new = tf.reshape(PA_h_new, [dim, 4])
    return PA_h_new



#################################################
#### Network definiton (subordinations) #########
#################################################
def sc_to_omega(i_mat,A, dim_A, omega, G_sc, depth):
    A_sub = A_subblock(A, dim_A)
    omega_op = omega
    omega_sc_op = tf.matrix_inverse(G_sc,name="F_sc") - omega_op + i_mat
    omega_op  = tf.matrix_inverse(Cauchy_from_sub(omega_sc_op, A_sub, dim_A ), name="for_h") \
    - omega_sc_op + i_mat
    return omega_op

def omega_net(i_mat,A, dim_A, sigma, omega, G_sc, depth, depth_sc):
    with tf.name_scope("omega_net"):
        A_sub = A_subblock(A, dim_A)
        G_sc_new = G_sc
        omega_new = omega
        for d in range(depth):
            with tf.name_scope("omega"):
                with tf.name_scope("h_sc-B"):
                    G_sc_new = sc_cauchy_2by2(omega_new,sigma, G_sc_new, depth_sc)
                    omega_sc =  -sigma**2*sc_eta_2by2(G_sc_new,name="F_sc") + i_mat
                with tf.name_scope("h_A-B"):
                    #omega_new =  i_mat
                    omega_new = tf.matrix_inverse(Cauchy_from_sub(omega_sc, A_sub, dim_A ), name="h_A") \
                    - omega_sc + i_mat
                """
                with tf.name_scope("summary_omega"):
                    tf.summary.scalar("omega_{}".format(d),tf.imag(tf.trace(omega_new)/2))
                    tf.summary.scalar("G_sc_{}".format(d),tf.imag(tf.trace(G_sc_new)/2))
                    tf.summary.scalar("omega_sc_{}".format(d),tf.imag(tf.trace(omega_sc)/2))
                """
    return omega_new, omega_sc

def recurrent_cauchy_net_MtoM(i_mat,A, dim_A, sigma, omega, G_sc, depth, depth_sc):
    omega_cell, omega_sc = omega_net(i_mat,A, dim_A, sigma, omega, G_sc, depth, depth_sc)
    with tf.name_scope("last_cauchy"):
        o_G = sc_cauchy_2by2(omega_cell, sigma, G_sc,depth_sc)
    return o_G, omega_cell, omega_sc


def omega_net_from_array(i_mat,array, dim, sigma, omega, G_sc, depth, depth_sc):
    with tf.name_scope("omega_net"):
        G_sc_new = G_sc
        omega_new = omega
        for d in range(depth):
            with tf.name_scope("omega"):
                with tf.name_scope("h_sc-B"):
                    G_sc_new = sc_cauchy_2by2(omega_new,sigma, G_sc_new, depth_sc)
                    omega_sc =  tf.matrix_inverse(G_sc_new,name="F_sc") - omega_new + i_mat
                with tf.name_scope("h_A-B"):
                    #omega_new =  i_mat
                    omega_new = tf.matrix_inverse(Cauchy_from_array(omega_sc, array,dim), name="h_A") \
                    - omega_sc + i_mat
                """
                with tf.name_scope("summary_omega"):
                    tf.summary.scalar("omega_{}".format(d),tf.imag(tf.trace(omega_new)/2))
                    tf.summary.scalar("G_sc_{}".format(d),tf.imag(tf.trace(G_sc_new)/2))
                    tf.summary.scalar("omega_sc_{}".format(d),tf.imag(tf.trace(omega_sc)/2))
                """
            #omega_list.append ()
    return omega_new, omega_sc


def square_density_from_G(G,z):
    with tf.name_scope("rho_from_G"):
        rho = -tf.imag(0.5*tf.trace(G)/z)/tf.constant(sp.pi, dtype=tf.float64)
    return rho

### G : 2 x 2  x dim
def square_density_from_G_reduce(G,z):
    with tf.name_scope("rho_from_G_reduce"):
        rho = -tf.imag(0.5*(G[0][0]+G[1][1])/z)/tf.constant(sp.pi, dtype=tf.float64)
    return rho


def recurrent_cauchy_net_from_array_MtoM(i_mat,array, dim, sigma, omega_init, G_sc_init, depth, depth_sc):
    with tf.name_scope("RCN"):
        omega_cell, omega_sc = omega_net_from_array(i_mat,array,dim, sigma, omega_init, G_sc_init, depth, depth_sc)
        with tf.name_scope("last_cauchy"):
            o_G = sc_cauchy_2by2(omega_cell, sigma, G_sc_init,depth_sc)
    return o_G, omega_cell, omega_sc



#############################################
###### Loss function ########################
#############################################
def likelyhood(x,array, sigma, omega, G_sc, depth, depth_sc, scale):
    z = tf.sqrt(x+1j*scale)
    i_mat = z*eye2
    G, omega_cell, omega_sc = recurrent_cauchy_net_from_array_MtoM(i_mat,array, dim, sigma, omega, G_sc, depth, depth_sc)
    rho = square_density_from_G(G, z)
    return rho


############################################
##### Gradient of loss #####################
############################################
def grad_loss(dim, z, array, sigma, G_out, omega, omega_sc):
    with tf.name_scope("grad_loss"):
        ### G_out(B)= G_sc(omega) = G_A(omega_sc) = (omega + omega_sc - B)^{-1}
        ### F_A = G_A(omega_sc)^{-1} = G(B)^{-1}
        i_mat = z*eye2

        F_A = omega + omega_sc - i_mat
        h_A = F_A + i_mat

        if test_mode:
            with tf.name_scope("test_omega_sc"):
                G_init_temp = -1j*eye2
                G_sc_temp = sc_cauchy_2by2(omega, sigma, G_init_temp, 20)
                omega_sc_temp = tf.matrix_inverse(G_sc_temp) - omega + i_mat
                L1_tester(omega_sc, omega_sc_temp)
            with tf.name_scope("test_G_sc"):
                L1_tester(G_sc_temp , G_out)
            with tf.name_scope("test_G_A"):
                G_A = tf.matrix_inverse(F_A)
                L1_tester(G_A , G_out)
            with tf.name_scope("test_FA"):
                G_A_temp = Cauchy_from_array(omega_sc, array, dim)
                F_A_temp = tf.matrix_inverse(G_A)
                L1_tester(F_A, F_A_temp)

        tpTGsc = tp_T_G_sc( G_out, sigma)
        tpThsc = tp_T_h_sc( G_out, sigma)
        tpThA  = tp_T_h_A(omega_sc, array, dim, F_A)
        ### 4x4
        with tf.name_scope("impricit_thm"):
            tpS =  tf.matrix_inverse(eye4 -  tpThsc  @ tpThA )

            #tf.summary.histogram("real_S", tf.real(S))
            #tf.summary.histogram("imag_S", tf.imag(S))

        with tf.name_scope("Psigma_G"):
            ### 1 x 4
            tpPsigmaOmega = tp_Psigma_h_sc(G_out, sigma) @ tpThA @ tpS
            PsigmaG =  tpPsigmaOmega @ tpTGsc + tp_Psigma_G_sc(G_out,sigma)

        with tf.name_scope("PA_G"):
            ### d x 4
            tpPAOmega = tp_PA_h_A(omega_sc, array, dim, h_A) @  tpThsc @ tpS
            tpPAG =  tpPAOmega @ tpTGsc


        ### rho and grad rho
        with tf.name_scope("loss"):
            rho = square_density_from_G(G_out, z)
            loss = - tf.log(rho)

        with tf.name_scope("Psigma"):
            Psigma_G = tf.reshape(PsigmaG, [2,2])
            Psigma_rho = square_density_from_G(Psigma_G, z)
            Psigma_loss = -Psigma_rho/ rho

        with tf.name_scope("PA"):
            tpPAG = tf.reshape(tpPAG, [dim,2,2])
            PA_rho = square_density_from_G(tpPAG, z)
            PA_loss = -PA_rho/ rho

        """
        with tf.name_scope("last_summary"):
            tf.summary.scalar("rho",rho)
            tf.summary.scalar("loss", loss)
            #tf.summary.histogram("Psigma_G", Psigma_G)
            tf.summary.scalar("Psigma_rho", Psigma_rho)
            tf.summary.scalar("Psigma_loss", Psigma_loss)
            #tf.summary.histogram("PA_G", PA_G)
            for d in range(dim):
                tf.summary.scalar("{}_PA_rho".format(d), PA_rho[d])
                tf.summary.scalar("{}_PA_loss".format(d), PA_loss[d])
        """

        if gather_summary:
            tf.summary.scalar("loss", loss)
    return loss, Psigma_loss, PA_loss


###### Minibatch x Cauchy seeds
def Cauchy_noise(num, location, scale):
    with tf.name_scope("Cauchy_noise"):
        standard = tf.tan(tf.constant(sp.pi, dtype=tf.float64)*(tf.random_uniform(shape=[num], minval=-0.5, maxval=0.5,dtype=tf.float64)))
        zero = tf.zeros(num, dtype=tf.float64)
        standard = tf.complex(standard,zero)
        seeds =  location + scale*standard
    return seeds

#@minibatch : minibatch_size x dim
def grad_Cauchy_noise_loss(dim , minibatch, minibatch_size, num_Cauchy_seeds,scale, \
array, sigma,  omega_init, G_sc_init, depth, depth_sc, use_tf_seed):
    with tf.name_scope("grad_CNL"):
        z_list = []
        loss_list = []
        Psigma_list = []
        PA_list = []
        if use_tf_seed:
            seeds = Cauchy_noise(num_Cauchy_seeds, 0, scale)
            if gather_summary:
                tf.summary.histogram("cauchy_nise", tf.real(seeds))
            for m in range(num_Cauchy_seeds):
                for n in range(minibatch_size):
                    with tf.name_scope("Each_loss"):
                        with tf.name_scope("x_to_i_mat"):
                            x = minibatch[n] - seeds[m]
                            z = tf.sqrt(x + 1j*scale)
                            i_mat = z*eye2
                        G_out, omega, omega_sc = recurrent_cauchy_net_from_array_MtoM(i_mat, array,dim, sigma, omega_init, G_sc_init, depth, depth_sc)
                        loss, Psigma, PA = grad_loss(dim, z, array, sigma, G_out,omega,omega_sc)
                        loss_list.append(loss)
                        Psigma_list.append(Psigma)
                        PA_list.append(PA)
        else:
            minibatch_size*= num_Cauchy_seeds
            for n in range(minibatch_size):
                with tf.name_scope("Each_loss"):
                    with tf.name_scope("x_to_i_mat"):
                        x = minibatch[n]
                        z = tf.sqrt(x + 1j*scale)
                        i_mat = z*eye2
                    G_out, omega, omega_sc = recurrent_cauchy_net_from_array_MtoM(i_mat, array,dim, sigma, omega_init, G_sc_init, depth, depth_sc)
                    loss, Psigma, PA = grad_loss(dim, z, array, sigma, G_out,omega,omega_sc)
                    loss_list.append(loss)
                    Psigma_list.append(Psigma)
                    PA_list.append(PA)

        with tf.name_scope("Average"):
            loss_average = tf.reduce_mean(tf.convert_to_tensor(loss_list))
            Psigma_average = tf.reduce_mean(tf.convert_to_tensor(Psigma_list))
            ### (num_Cauchy_seeds*minibatch_size) x d
            PA = tf.convert_to_tensor(PA_list)
            PA_average = tf.reduce_mean(PA, 0)
    return loss_average, Psigma_average, PA_average


def L1_tester(X,Y):
    L1_norm= tf.reduce_mean(tf.abs(X-Y))
    tf.summary.scalar("L1_norm", L1_norm)


########################################
###### SGD ###########
########################################
