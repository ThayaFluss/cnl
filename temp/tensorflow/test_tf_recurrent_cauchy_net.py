import tensorflow as tf
import numpy as np
import scipy as sp
from timer import Timer
from tf_recurrent_cauchy_net import *
from matrix_util import *
from random_matrices import *

from cauchy import SemiCircular


import matplotlib.pyplot as plt
"""
###Test cauchy_mat_valued
size = 100
bysize = 2*size
A_value = np.arange(bysize**2).reshape([bysize,bysize])
A = tf.constant(A_value, tf.complex128)
W_value = 1j*np.identity(2) + 0.1*np.arange(4).reshape([2,2])
W = tf.constant(W_value, tf.complex128)

G = cauchy_mat_valued(W,A, bysize)

timer = Timer()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    timer.tic()
    sess.run(G)
    print(G)

timer.toc()
print(timer.total_time)



###Test sc_cauchy_2by2


scale = 1e-6
x = 10.
sigma = 1.
i_mat_value = (x+1j*scale)*np.identity(2)
i_mat = tf.constant(i_mat_value, tf.complex128)
#G_init = tf.placeholder(tf.complex64,shape=(2,2))


G_init_value = -1j * np.identity(2)
G = tf.Variable(G_init_value, tf.complex128)

net = sc_cauchy_2by2(i_mat, sigma, G, depth=10)

timer = Timer()
with tf.Session() as sess:
    timer.tic()
    sess.run(tf.global_variables_initializer())
    #out = sess.run(G, feed_dict={G:G_init_value)
    value = sess.run(net)
    print(value)


timer.toc()
print(timer.total_time)

### Test omega
size = 36
bysize = 2*size
A_value = (1+1j)*np.random.randn(bysize,bysize)/bysize
A_value = A_value + np.matrix(A_value).H
A_value /=2.
A_value = np.array(A_value) + np.identity(bysize)
A = tf.Variable(A_value, tf.complex128, name = "A")

sigma = 1
scale = 1e-6
x = 1.
i_mat_value = (x+1j*scale)*np.identity(2)
i_mat = tf.constant(i_mat_value, tf.complex128, name="i_mat")

omega = tf.Variable( 1j*np.identity(2), tf.complex128, name= "omega")
omega_sc = tf.Variable( 1j*np.identity(2), tf.complex128, name = "omega_sc")

G_sc = tf.Variable(-1j*np.identity(2), tf.complex128, name="G_sc")

###graph
net = omega_net(i_mat, A, bysize, sigma, omega, omega_sc,G_sc, 40,20)

timer = Timer()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('./tb_log_rcn', graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    timer.tic()
    out = sess.run(net)
    #out = sess.run(omega, feed_dict={init_G:init_G_value})
    print (out)

timer.toc()
print(timer.total_time)

"""


def square_density_tf(x_array, param_mat, scale):
    size = param_mat.shape[1]
    assert param_mat.shape[0] == size

    ###offdiagonal block matrix
    e_param_mat = np.zeros(4*size**2, dtype=np.complex).reshape([2*size, 2*size])
    for k in range(size):
        for l in range(size):
            e_param_mat[k][size+l] = np.matrix(param_mat).H[k,l]
            e_param_mat[size+k][l] = param_mat[k,l]
    e_param_mat = np.array(e_param_mat)


    ###Variables
    A = tf.Variable(e_param_mat, tf.complex128, name = "A")
    sigma = tf.Variable(1.+0*1j, tf.complex128, name = "sigma")
    ###placeholders
    i_mat = tf.placeholder( tf.complex128, name="i_mat")
    omega = tf.placeholder(  tf.complex128, name= "omega")
    G_sc = tf.placeholder(tf.complex128, name="G_sc")

    omega_init = 1j*np.identity(2)
    G_sc_value = -1j*np.identity(2)

    ###graph
    #sc_cell = sc_cauchy_2by2(i_mat, sigma, G_sc,depth=20)
    net, omega_cell = recurrent_cauchy_net(i_mat, A, 2*size, sigma,\
    omega, G_sc, depth=10,depth_sc=20)
    #omega_test = omega_net(i_mat, A, 2*size, sigma,\
    #omega, omega_sc,G_sc, depth=2,depth_sc=10)
    #G_sc_out = tf.placeholder(tf.complex128, name= "G_sc_out")
    #sc_to_omega_cell = sc_to_omega(i_mat, A, 2*size, sigma, omega, G_sc_out, depth=100)


    ###Session
    sess = tf.InteractiveSession()

    ###Summary
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./tb_log_rcn', graph=sess.graph)

    ###Initialize
    sess.run(tf.global_variables_initializer())

    ###main
    timer = Timer()
    timer.tic()
    num = len(x_array)
    rho_list = []
    for i in range(num):
        x = x_array[i]
        z = sp.sqrt(x+1j*scale)
        var_mat = z*np.identity(2)
        summary, net_out,omega_init = sess.run([merged, net, omega_cell], \
        feed_dict={G_sc:G_sc_value, i_mat:var_mat, omega:omega_init})
        summary_writer.add_summary(summary, i)
        #print ("omega=", omega_out)
        #print ("i_mat=", var_mat)

        #for n in range(num_iter):
        #    sc_cell_out= sess.run(sc_cell, feed_dict={G_sc:G_sc_value, i_mat:var_mat})
        #    out = sess.run(sc_to_omega_cell, feed_dict={G_sc_out:sc_cell_out, i_mat:var_mat,omega:out})
        #out = G_sc_value
        G_2 = net_out / z   ### zG_2(z^2) = G(z)
        rho =  -ntrace(G_2).imag/sp.pi
        print ("rho({})={}".format(x,rho))
        #logging.debug( "(density_info_plus_noise)rho(", x, ")= " ,rho
        rho_list.append(rho)
    timer.toc()
    print ("total_time=", timer.total_time)



    return np.array(rho_list)



def square_density_tf_from_array(x_array, param_array, scale):
    size = param_array.shape[0]
    ###Variables
    array = tf.Variable(param_array, tf.complex128, name = "singular")
    sigma = tf.Variable(1.+0*1j, tf.complex128, name = "sigma")
    ###placeholders
    i_mat = tf.placeholder( tf.complex128, name="i_mat")
    omega = tf.placeholder(  tf.complex128, name= "omega")
    G_sc = tf.placeholder(tf.complex128, name="G_sc")

    omega_init = 1j*np.identity(2)
    G_sc_value = -1j*np.identity(2)

    ###graph
    #sc_cell = sc_cauchy_2by2(i_mat, sigma, G_sc,depth=20)
    net, omega_cell, omega_sc = recurrent_cauchy_net_from_array_MtoM(i_mat, array, sigma,\
    omega, G_sc, depth=20,depth_sc=20)
    #omega_test = omega_net(i_mat, A, 2*size, sigma,\
    #omega, omega_sc,G_sc, depth=2,depth_sc=10)
    #G_sc_out = tf.placeholder(tf.complex128, name= "G_sc_out")
    #sc_to_omega_cell = sc_to_omega(i_mat, A, 2*size, sigma, omega, G_sc_out, depth=100)


    ###Session
    sess = tf.InteractiveSession()

    ###Summary
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./tb_log_rcn', graph=sess.graph)

    ###Initialize
    sess.run(tf.global_variables_initializer())

    ###main
    timer = Timer()
    timer.tic()
    num = len(x_array)
    rho_list = []
    for i in range(num):
        x = x_array[i]
        z = sp.sqrt(x+1j*scale)
        var_mat = z*np.identity(2)
        summary, net_out,omega_init, omega_sc_out = sess.run([merged, net, omega_cell, omega_sc], \
        feed_dict={G_sc:G_sc_value, i_mat:var_mat, omega:omega_init})
        summary_writer.add_summary(summary, i)
        #print ("omega=", omega_out)
        #print ("i_mat=", var_mat)

        #for n in range(num_iter):
        #    sc_cell_out= sess.run(sc_cell, feed_dict={G_sc:G_sc_value, i_mat:var_mat})
        #    out = sess.run(sc_to_omega_cell, feed_dict={G_sc_out:sc_cell_out, i_mat:var_mat,omega:out})
        #out = G_sc_value
        G_2 = net_out / z   ### zG_2(z^2) = G(z)
        rho =  -ntrace(G_2).imag/sp.pi
        print ("rho({})={}".format(x,rho))
        #logging.debug( "(density_info_plus_noise)rho(", x, ")= " ,rho
        rho_list.append(rho)
    timer.toc()
    print ("total_time=", timer.total_time)



    return np.array(rho_list)


PLOTEVS = False

size = 100
x_array = np.linspace(0.01, 40, 101) ## Modify
a = np.zeros(size, np.complex128)
half_size = int(size/2)
for i in range(half_size):
  a[i] = 2.
  a[i+half_size]=5.
sigma = 1
scale = 1e-4

plt.figure()

#A = np.diag(a)

if PLOTEVS:
    A = np.diag(a)
    evs_list =[]
    num_sample = 100
    bins=100
    num_cauchy = 100
    for i  in range(num_sample):
        evs= np.linalg.eigh(info_plus_noise(size, A,sigma, COMPLEX=True))[0]
        c_noise =  sp.stats.cauchy.rvs(loc=0, scale=scale, size=num_cauchy)
        for n in range(num_cauchy):
            new_evs = evs-c_noise[n]
            evs_list += new_evs.tolist()

    plt.hist(evs_list, bins=bins, normed=True, label="empirical eigenvalues")

#rho_array = square_density_tf(x_array, A,  scale)
rho_array = square_density_tf_from_array(x_array, a,  scale)

plt.plot(x_array,rho_array)
plt.savefig("./rho_array_tf.png")

G = tf.constant(-2j*np.identity(2), tf.complex128)
sigma = tf.constant(1., tf.complex128)
