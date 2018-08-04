import unittest
import numpy as np
from psvd import *
from random_matrices import *

class TestPSVD(unittest.TestCase):
    def test_psvd_cnl(self):
        p = 50
        d = 50
        sigma = 0.2
        zero_dim = 40
        min_singular = 0.3
        sample_mat = random_from_diag(p,d,zero_dim, min_singular)
        sample_mat += sigma*Ginibre(p,d)
        U,D, V, sigma = psvd_cnl(sample_mat)
        #import pdb; pdb.set_trace()

    def test_rank_estimation(self):
        p = 50
        d = 50
        sigma = 0.1
        zero_dim = 10
        min_singular = 0.3
        sample_mat = random_from_diag(p,d,zero_dim, min_singular)
        sample_mat += sigma*Ginibre(p,d)

        ranks, a, s = rank_estimation(sample_mat)

    def test_z_value(self):
        p = 50
        d = 25
        sigma = 0.2
        sample_mat = sigma*Ginibre(p,d)
        a = np.zeros(d)
        z  = z_value(sample_mat, a, sigma)
