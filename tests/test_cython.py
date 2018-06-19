# -*-encode: utf-8-*-
from timer import *
import numpy as np
from cython_code import *

timer = Timer()

Z = np.asarray( [0.1j, 0.1j], dtype=np.complex)
o_G = np.asarray( [-1j, -1j], dtype=np.complex)
max_iter = 1000
thres = 1e-7
p_dim = 50
dim = 50
sigma = 0.1
o_forward_iter = np.asarray( [0], dtype=np.int)
timer.tic()
result = cy_cauchy_2by2(Z, o_G,max_iter,thres, sigma, p_dim, dim,o_forward_iter)
timer.toc()
print("result={}".format(result))
print("forward_iter={}".format(o_forward_iter))
print("o_G={}".format(o_G))
print("{} sec".format(timer.total_time))
#import pdb; pdb.set_trace()

B = np.asarray( [0.1j, 0.1j], dtype=np.complex)
o_omega = np.asarray( [0.1j, 0.1j], dtype=np.complex)
o_omega_sc = np.asarray( [0.1j, 0.1j], dtype=np.complex)

o_G_sc = np.asarray( [-1j, -1j], dtype=np.complex)
max_iter = 1000
thres = 1e-7
p_dim = 50
dim = 50
sigma = 0.1
o_forward_iter = np.asarray( [0], dtype=np.int)
a = 0.2*np.ones(dim)
a = np.asarray(a, dtype=float)

timer.tic()
result = cy_cauchy_subordination(B, o_omega,o_G_sc,max_iter,thres, \
sigma, p_dim, dim, o_forward_iter, a,  o_omega_sc)
timer.toc()
print("result={}".format(result))
print("forward_iter={}".format(o_forward_iter))
print("o_G_sc={}".format(o_G_sc))
print("{} [sec]".format(timer.total_time))
import pdb; pdb.set_trace()
