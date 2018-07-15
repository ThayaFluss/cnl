# -*-encode: utf-8-*-
import unittest
from timer import *
import numpy as np
from cy_fde_spn import *
from spn_c2 import *


class TestCython(unittest.TestCase):
    def test_cy_test_all(self):
        result = 1
        result = cy_test_all(result)
        assert result

    def test_cy_cauchy_subordianation(self):
        timer = Timer()

        Z = np.asarray( [0.1j, 0.1j], dtype=np.complex)
        o_G = np.asarray( [-1j, -1j], dtype=np.complex)
        max_iter = 1000
        thres = 1e-8
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
        thres = 1e-8
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

    def test_cy_grad_cauchy_spn(self):
        timer = Timer()

        Z = np.asarray( [0.1j, 0.1j], dtype=np.complex)
        o_G = np.asarray( [-1j, -1j], dtype=np.complex)

        B = np.asarray( [0.1j, 0.1j], dtype=np.complex128)
        o_omega = np.asarray( [0.1j, 0.1j], dtype=np.complex128)
        o_omega_sc = np.asarray( [0.1j, 0.1j], dtype=np.complex128)

        o_G_sc = np.asarray( [-1j, -1j], dtype=np.complex128)
        max_iter = 1000
        thres = 1e-8
        p_dim = 50
        dim = 50
        sigma = 0.1
        o_forward_iter = np.asarray( [0], dtype=np.int)
        a = 0.2*np.ones(dim)
        a = np.asarray(a, dtype=float)

        result = cy_cauchy_subordination(B, o_omega,o_G_sc,max_iter,thres, \
        sigma, p_dim, dim, o_forward_iter, a,  o_omega_sc)


        o_grad_sigma = np.zeros(2, dtype=np.complex128)
        o_grad_a = np.zeros(2*dim, dtype=np.complex128)

        z = Z[0]
        cy_timer = Timer()
        cy_timer.tic()
        cy_grad_cauchy_spn(p_dim ,dim,\
         z, a, sigma, o_G_sc, o_omega, o_omega_sc, o_grad_a, o_grad_sigma)
        cy_timer.toc()
        print("c:Backward= {} sec".format(cy_timer.total_time))


        sc = SemiCircular(dim=dim, p_dim=p_dim)
        sc.set_params(a,sigma)
        sc.TEST_MODE = False
        py_timer = Timer()
        py_timer.tic()
        py_grad = sc.grad_subordination(z, o_G_sc, o_omega, o_omega_sc, CYTHON=False)
        py_timer.toc()
        print("python:Backward= {} sec".format(py_timer.total_time))

        py_grad_a = py_grad[:dim, :].flatten()
        py_grad_sigma = py_grad[-1, :].flatten()

        assert( np.allclose(py_grad_a, o_grad_a))
        assert( np.allclose(py_grad_sigma, o_grad_sigma))


        sc.TEST_MODE = True
        py_grad = sc.grad_subordination(z, o_G_sc, o_omega, o_omega_sc, CYTHON=True)
